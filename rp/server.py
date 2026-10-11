"""OpenAI-compatible role-play server that learns from chatting.

    python -m rp.server [--port 8619] [--data-dir rp_data] [--model Qwen/Qwen2.5-0.5B-Instruct]

SillyTavern: API = Chat Completion, source = Custom (OpenAI-compatible), endpoint = http://127.0.0.1:8619/v1
"""
from __future__ import annotations

import argparse
import hmac
import logging
import re
import signal
from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager
import html
import json
import os
import subprocess
import sys
import threading
import time
import uuid

import uvicorn
import psutil
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from . import DEFAULT_DATA_DIR, DEFAULT_MODEL, DEFAULT_PORT
from .settings import LearningSettings
from .learn import Analysis, clean_messages
from .engine import list_adapters, read_current, write_current
from .learn import ACK_PREFIX, Learner
from .store import Store
from .ui.journal import JobJournal


class SleepManager:
    def __init__(self, cfg, engine):
        self.cfg, self.engine = cfg, engine
        self.proc = None
        self.started = self.finished = None
        self.last_result = None
        self.metrics = []
        self.progress = 0
        self.checkpoint = None
        self.record = None
        self.log_path = os.path.join(cfg.data_dir, "sleep.log")
        self._lock = threading.RLock()
        self.journal = JobJournal(cfg.data_dir)
        self.state_path = Path(cfg.data_dir) / "sleep-status.json"
        if self.state_path.is_file():
            try:
                previous = json.loads(self.state_path.read_text())
                self.last_result = previous.get("last_result")
                self.finished = previous.get("finished")
                self.checkpoint = previous.get("checkpoint")
                if previous.get("running"):
                    self.last_result = {"status": "interrupted", "reason": "Server restarted during training. Resume its saved checkpoint from the trainer."}
            except (OSError, ValueError):
                pass

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, extra: dict | None = None, reason="manual", resume=None):
        with self._lock:
            if self.running or self.started and self.finished is None:
                return self.status()
            for record in self.journal.history():
                if record.get('state') in ('queued','running','stopping') and (record.get('pid') is None or self._record_alive(record)):
                    raise RuntimeError('Another training job is already running. Stop it or wait for its checkpoint.')
            cmd = [sys.executable, "-m", "rp.sleep", "--data-dir", self.cfg.data_dir, "--model", self.cfg.model]
            options = dict(extra or {})
            if self.cfg.sleep_threads:
                options.setdefault('threads', self.cfg.sleep_threads)
            allowed = {"epochs", "lr", "rank", "accum", "max_minutes", "threads", "guard_max_drift", "no_activate"}
            if set(options) - allowed:
                raise ValueError("Unsupported training setting.")
            bounds = {"epochs": (1,20), "lr": (1e-6,.005), "rank": (1,128), "accum": (1,64), "max_minutes": (.1,120), "threads": (1,64), "guard_max_drift": (0,10)}
            integers = {'epochs', 'rank', 'accum', 'threads'}
            for key, value in options.items():
                if key == 'no_activate':
                    if not isinstance(value, bool):
                        raise ValueError('no_activate must be true or false.')
                elif isinstance(value, bool) or not isinstance(value, (int,float)) or not bounds[key][0] <= value <= bounds[key][1] or (key in integers and not isinstance(value, int)):
                    raise ValueError(f"Invalid training setting: {key}.")
                flag = "--" + key.replace("_", "-")
                if value is True:
                    cmd.append(flag)
                elif value is not False:
                    cmd += [flag, str(value)]
            if resume:
                checkpoint = Path(resume).resolve()
                if not checkpoint.is_dir() or not checkpoint.is_relative_to(Path(self.cfg.data_dir).resolve()):
                    raise ValueError("The checkpoint must exist inside project storage.")
                cmd += ["--resume",str(checkpoint)]
                options["resume"] = str(checkpoint)
            identifier = uuid.uuid4().hex[:16]
            self.log_path = str(self.journal.log_path(identifier))
            logf = open(self.log_path, 'w', encoding='utf-8')
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['TOKENIZERS_PARALLELISM'] = 'false'
            if getattr(self.cfg, 'auth_token', ''):
                env['RP_AUTH_TOKEN'] = self.cfg.auth_token
            try:
                self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                             cwd=Path(__file__).resolve().parents[1])
            except OSError:
                logf.close()
                raise RuntimeError('The trainer could not start. Check Python dependencies and storage.')
            self.started, self.finished, self.last_result = time.time(), None, None
            self.metrics, self.progress, self.checkpoint = [], 0, None
            self.record = {'id':identifier, 'kind':'sleep', 'state':'running', 'progress':0,
                           'log_tail':[], 'result':None, 'error':None, 'started_at':self.started,
                           'finished_at':None, 'metrics':[], 'checkpoint':None, 'config':options,
                           'pid':self.proc.pid, 'process_created':psutil.Process(self.proc.pid).create_time(),
                           'source':reason}
            self._persist()
            threading.Thread(target=self._wait, args=(self.proc, logf), daemon=True).start()
            return self.status()

    @staticmethod
    def _record_alive(record):
        try:
            process = psutil.Process(record['pid'])
            return process.is_running() and abs(process.create_time() - record.get('process_created',-1)) < 1
        except (KeyError, TypeError, psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def _persist(self):
        payload = {"running": self.running, "started": self.started, "finished": self.finished,
                   "last_result": self.last_result, "checkpoint":self.checkpoint}
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False))
        os.replace(temporary, self.state_path)
        if self.record:
            self.record.update(progress=self.progress, checkpoint=self.checkpoint,
                               metrics=self.metrics[-1000:], finished_at=self.finished,
                               result=self.last_result, log_tail=self._tail(100))
            self.journal.save(self.record)

    def stop(self):
        with self._lock:
            if not self.running:
                return False
            self.record['state'] = 'stopping'
            self.proc.send_signal(signal.SIGTERM)
            self._persist()
            return True

    def _wait(self, proc, logf):
        result = None
        with open(self.log_path, encoding='utf-8', errors='replace') as source:
            while True:
                line = source.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(.1)
                    continue
                with self._lock:
                    try:
                        if line.startswith('METRIC '):
                            metric = json.loads(line[7:])
                            self.metrics.append(metric)
                            self.progress = max(0, min(.99, float(metric.get('progress',self.progress))))
                            self.checkpoint = metric.get('checkpoint') or self.checkpoint
                        elif line.startswith('RESULT '):
                            result = json.loads(line[7:])
                            self.checkpoint = result.get('checkpoint') or self.checkpoint
                    except (ValueError, TypeError):
                        pass
                    self._persist()
        code = proc.wait()
        logf.close()
        result = result or {"status": "failed", "exit_code": code, 'reason':'The trainer exited without a completion record.'}
        if code != 0:
            result = {"status": "failed", "exit_code": code, "reason": "Training did not finish. Inspect the saved log; previous adapters remain available."}
        if result.get("activated"):
            try:
                self.engine.activate(read_current(self.cfg.data_dir, self.cfg.model))
                result["reloaded"] = True
            except Exception:
                result["reload_error"] = 'The saved adapter could not be loaded. Select an earlier version in Models.'
        with self._lock:
            self.last_result, self.finished = result, time.time()
            self.record['state'] = 'paused' if result.get('status') == 'paused' else 'done' if result.get('status') in ('ok','skipped') else 'error'
            self.record['error'] = result.get('reason') if self.record['state'] == 'error' else None
            if self.record['state'] == 'done':
                self.progress = 1
            self._persist()

    def _tail(self, n=30):
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, errors="replace") as source:
            from collections import deque
            return [line.rstrip("\n") for line in deque(source, maxlen=n)]

    def wait(self, timeout=None):
        started = time.time()
        while self.running or (self.started and not self.finished):
            if timeout and time.time() - started > timeout:
                break
            time.sleep(0.1)
        return self.status()

    def status(self):
        return {"running": self.running, "started": self.started, "finished": self.finished,
                "last_result": self.last_result, "log_tail": self._tail(12),
                'progress':self.progress, 'metrics':self.metrics[-1000:], 'checkpoint':self.checkpoint,
                'job_id':self.record['id'] if self.record else None}


class Message(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str = Field(min_length=1, max_length=100000)


class ChatInput(BaseModel):
    model_config = ConfigDict(extra='allow')
    messages: list[Message] = Field(min_length=1, max_length=512)
    temperature: float = Field(default=.7, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=2048)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=2048)
    top_p: float = Field(default=.9, gt=0, le=1)
    stream: bool = False
    user: str | None = Field(default=None, max_length=200)
    session_id: str | None = Field(default=None, max_length=200)
    metadata: dict = Field(default_factory=dict)
    repetition_penalty: float | None = Field(default=None, ge=1, le=2)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    top_k: int = Field(default=0, ge=0, le=100)
    seed: int | None = Field(default=None, ge=0, le=2147483647)
    stop: str | list[str] | None = None


def create_app(cfg, engine=None, start_scheduler=True) -> FastAPI:
    auth_token = getattr(cfg, "auth_token", "") or os.environ.get("RP_AUTH_TOKEN", "")
    if cfg.host not in ("127.0.0.1", "localhost", "::1") and len(auth_token) < 24:
        raise ValueError("Set RP_AUTH_TOKEN before exposing the model server.")
    store = Store(cfg.data_dir)
    if engine is None:
        from .engine import Engine
        engine = Engine(cfg.model, cfg.data_dir, dtype=cfg.dtype, threads=cfg.threads, max_context=cfg.max_context)
    learning = LearningSettings(cfg.data_dir)
    if not learning.path.is_file():
        learning.update({"min_signals": max(1, cfg.auto_sleep_signals), "idle_seconds": max(10, cfg.auto_sleep_idle), "auto_sleep": cfg.auto_sleep_signals > 0})
    learner = Learner(store)
    sleeper = SleepManager(cfg, engine)
    state = {"last_request": time.time(), "next_auto": 0, "busy": 0}
    shutdown = threading.Event()

    @asynccontextmanager
    async def lifespan(app):
        if start_scheduler:
            threading.Thread(target=auto_sleep_loop, daemon=True).start()
        yield
        shutdown.set()
        sleeper.stop()
        store.close()

    app = FastAPI(title="Roleplay chat learning API", lifespan=lifespan)
    auth_token = getattr(cfg, 'auth_token', '') or os.environ.get('RP_AUTH_TOKEN', '')
    if cfg.host not in ('127.0.0.1', 'localhost', '::1') and len(auth_token) < 24:
        raise ValueError('Set RP_AUTH_TOKEN before exposing the model server.')

    @app.middleware('http')
    async def require_auth(request, call_next):
        if auth_token and not hmac.compare_digest(request.headers.get('Authorization','').encode('utf-8'),('Bearer '+auth_token).encode('utf-8')):
            return JSONResponse({'detail': 'Authentication required.'}, status_code=401)
        return await call_next(request)

    @app.get('/health')
    def health():
        return {'status': 'ready', 'model': cfg.model, 'adapter': engine.version}

    app.state.store = store
    app.state.engine = engine
    app.state.sleeper = sleeper
    model_id = cfg.model_id

    def model_label():
        return f"{model_id}@{engine.version}"


    def auto_sleep_check():
        options = learning.get()
        if not options['enabled'] or not options['auto_sleep'] or options['mode'] == 'collect_only' or sleeper.running:
            return False
        now = time.time()
        if now < state['next_auto']:
            return False
        fresh = store.stats()['new_signals_since_sleep']
        idle = now - state['last_request']
        if not any(item.get('chosen') for item in store.fresh_signals()):
            return False
        if fresh < options['min_signals'] or (options['mode'] == 'idle' and (idle < options['idle_seconds'] or state['busy'])):
            return False
        state['next_auto'] = now + max(300, options['idle_seconds'])
        previous = sleeper.last_result or {}
        checkpoint = sleeper.checkpoint if previous.get('status') in ('paused','interrupted') else None
        sleeper.start({key:options[key] for key in ('epochs','max_minutes','threads','guard_max_drift')}, reason='automatic ' + options['mode'], resume=checkpoint)
        return True

    def auto_sleep_loop():
        while not shutdown.wait(10):
            try:
                auto_sleep_check()
            except (ValueError, RuntimeError):
                continue

    app.state.auto_sleep_check = auto_sleep_check
    app.state.learning_state = state
    app.state.learning = learning


    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "rp",
             "adapter": engine.version, "base_model": cfg.model}]}

    def ack_text(a):
        if a.command in ("good", "bad", "fix"):
            msg = learner.apply_command(a)
        elif a.command == "sleep":
            options = learning.get()
            try:
                st = sleeper.start({key:options[key] for key in ("epochs","max_minutes","threads","guard_max_drift")},reason="chat command")
            except RuntimeError:
                return ACK_PREFIX + "另一个训练任务正在运行，请等待检查点完成。"
            msg = "开始睡眠（训练 LoRA），完成后自动切换到新版本。" if st["running"] else "睡眠没有启动。"
        else:
            s = store.stats()
            msg = (f"当前版本 {engine.version}；信号 {s['signals']}；自上次睡眠以来新增 {s['new_signals_since_sleep']} 条；"
                   f"睡眠中：{'是' if sleeper.running else '否'}")
        return f"{ACK_PREFIX}{msg}"

    def chunk(cid, created, delta, finish=None):
        return "data: " + json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": created, "model": model_label(),
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}, ensure_ascii=False) + "\n\n"

    def completion(cid, created, text, finish, usage, gid=None):
        return {"id": cid, "object": "chat.completion", "created": created, "model": model_label(),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                "usage": usage, "rp_generation_id": gid}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        try:
            body = ChatInput.model_validate(await request.json()).model_dump()
        except ValueError:
            raise HTTPException(422, 'Use valid user/assistant/system messages and bounded generation settings.')
        if sum(len(m['content']) for m in body['messages']) > 1000000:
            raise HTTPException(413, 'The conversation is too large. Start a new scene or shorten its history.')
        if any(not message['content'].strip() for message in body['messages']):
            raise HTTPException(422,'Messages cannot be empty.')
        state["last_request"] = time.time()
        msgs = body.get("messages") or []
        if not msgs:
            raise HTTPException(400, "messages is required")
        session_id = body.get('session_id') or body.get('metadata', {}).get('conversation_id') or body.get('user') or request.headers.get('X-RP-Session')
        if session_id is not None and (not isinstance(session_id, str) or len(session_id) > 200):
            raise HTTPException(422, 'Conversation ID must be a short string.')
        options = learning.get()
        if options['enabled']:
            a = await run_in_threadpool(learner.analyze, msgs, session_id)
        else:
            a = Analysis('other', clean_messages(msgs))
        for ev in a.events:
            print(f"[learn] {ev[0]} turn={ev[1]}", flush=True)
        cid, created, stream = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time()), bool(body.get("stream"))

        if a.kind == "command":
            text = await run_in_threadpool(ack_text, a)
            print(f"[learn] command {a.command}: {text}", flush=True)
            if not stream:
                return completion(cid, created, text, "stop", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

            async def acks():
                yield chunk(cid, created, {"role": "assistant", "content": text})
                yield chunk(cid, created, {}, "stop")
                yield "data: [DONE]\n\n"
            return StreamingResponse(acks(), media_type="text/event-stream")

        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or cfg.max_tokens
        temperature = body.get("temperature", cfg.temperature)
        rep = body.get("repetition_penalty")
        if rep is None:
            fp = max(float(body.get("frequency_penalty") or 0), float(body.get("presence_penalty") or 0))
            rep = max(1, min(2, cfg.repetition_penalty + 0.25 * fp))
        stop = body.get("stop") or []
        if isinstance(stop, list) and (len(stop) > 8 or any(len(item) > 1000 for item in stop)) or isinstance(stop, str) and len(stop) > 1000:
            raise HTTPException(422, "Use at most eight short stop strings.")
        params = dict(max_tokens=min(int(max_tokens), cfg.max_tokens_cap), temperature=temperature,
                      top_p=body.get("top_p", cfg.top_p), top_k=body.get("top_k", 0) or 0,
                      repetition_penalty=rep, seed=body.get("seed"),
                      stop=[stop] if isinstance(stop, str) else list(stop))
        cancel = threading.Event()
        info: dict = {}

        def record(text):
            if not options['enabled']:
                return None
            gid = store.add_generation(
                a.turn_key, a.messages, text, info.get("adapter", engine.version), params, prefill=a.prefill,
                latency_s=info.get("latency_s"), first_token_s=info.get("first_token_s"),
                prompt_tokens=info.get("prompt_tokens"), completion_tokens=info.get("completion_tokens"),
                session_id=session_id, scope_id=a.scope_id)
            if a.kind == "other":
                store.set_status(gid, "other")
            n = info.get("completion_tokens") or 0
            lat = info.get("latency_s") or 0
            print(f"[gen] {gid} {a.kind} adapter={info.get('adapter')} prompt={info.get('prompt_tokens')} "
                  f"out={n} first={info.get('first_token_s') or 0:.2f}s total={lat:.2f}s "
                  f"({n / max(1e-6, lat - (info.get('first_token_s') or 0)):.1f} tok/s)", flush=True)
            return gid

        def finish_reason():
            return "length" if (info.get("completion_tokens") or 0) >= params["max_tokens"] else "stop"

        if not stream:
            state['busy'] += 1
            try:
                text, _ = await run_in_threadpool(lambda: engine.generate(a.messages, a.prefill, info=info, **params))
            except (ValueError, RuntimeError):
                logging.getLogger("rp.server").exception("Generation failed")
                raise HTTPException(503, 'Generation could not complete. Check the active adapter and available memory.')
            finally:
                state['busy'] -= 1
            gid = await run_in_threadpool(record, text)
            pt, ct = info.get("prompt_tokens", 0), info.get("completion_tokens", 0)
            return completion(cid, created, text, finish_reason(),
                              {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}, gid)

        async def events():
            state["busy"] += 1
            produced = []
            succeeded = False
            disconnected = False
            try:
                yield chunk(cid, created, {"role": "assistant", "content": ""})
                async for delta in iterate_in_threadpool(engine.stream(a.messages, a.prefill, cancel=cancel, info=info, **params)):
                    produced.append(delta)
                    yield chunk(cid, created, {"content": delta})
                    if await request.is_disconnected():
                        disconnected = True
                        cancel.set()
                        break
                succeeded = not disconnected
                yield chunk(cid, created, {}, finish_reason())
                yield "data: [DONE]\n\n"
            except (ValueError, RuntimeError):
                logging.getLogger("rp.server").exception("Streaming generation failed")
                yield "data: " + json.dumps({"error":{"message":"Generation could not complete. Check the active adapter and available memory.", "type":"generation_error"}}) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                state["busy"] -= 1
                cancel.set()
                if succeeded:
                    await run_in_threadpool(record, "".join(produced))

        return StreamingResponse(events(), media_type="text/event-stream")


    @app.get("/rp/stats")
    def stats():
        return {"adapter": engine.version, "model": cfg.model, **store.stats(),
                "sleep": sleeper.status(), "auto_sleep": learning.get(), "busy": state["busy"]}

    @app.get("/rp/signals")
    def signals(n: int = 20):
        return store.recent_signals(n)

    @app.post("/rp/feedback")
    async def feedback(request: Request):
        try:
            b = await request.json()
            if not isinstance(b, dict):
                raise ValueError()
        except ValueError:
            raise HTTPException(422, "Send a JSON object.")
        if not learning.get()["enabled"]:
            raise HTTPException(409, "Learning is disabled. Enable feedback collection in Settings first.")
        if b.get("gen_id") is not None and (not isinstance(b["gen_id"],str) or len(b["gen_id"]) > 100):
            raise HTTPException(422,"gen_id must be a short string.")
        g = store.get_generation(b["gen_id"]) if b.get("gen_id") else store.last_generation()
        if not g:
            raise HTTPException(404, "no generation to give feedback on")
        full = (g["prefill"] or "") + g["output"]
        rating, corr = b.get("rating"), b.get("correction")
        if corr is not None and (not isinstance(corr,str) or not corr.strip() or len(corr) > 100000):
            raise HTTPException(422, 'A correction must contain 1–100,000 characters.')
        if corr:
            command = 'fix'
        elif rating in ('good','bad'):
            command = rating
        else:
            raise HTTPException(400, 'give rating=good|bad or a correction')
        analysis = Analysis('command', [], command=command, command_arg=corr or '',
                            target={'gen':g, 'content':full, 'turn_key':g['turn_key'], 'messages':g['messages']},
                            session_id=g.get('session_id'), scope_id=g.get('scope_id',''))
        await run_in_threadpool(learner.apply_command,analysis)
        sid = store.find_signals(g['turn_key'],kind=command,gen_id=g['id'],active_only=False)[-1]['id']
        return {"ok": True, "signal_id": sid, "gen_id": g["id"]}

    @app.post("/rp/sleep")
    async def sleep(request: Request):
        try:
            b = await request.json()
        except Exception:
            b = {}
        try:
            options = learning.get()
            extra = {k: options[k] for k in ('epochs','max_minutes','threads','guard_max_drift')}
            if not isinstance(b, dict) or not isinstance(b.get("args",{}),dict):
                raise ValueError("Send training settings in an args object.")
            extra.update(b.get('args') or {})
            timeout = b.get("timeout", 60)
            if b.get("wait") and (isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not 1 <= timeout <= 3600):
                raise ValueError("timeout must be between 1 and 3600 seconds.")
            st = sleeper.start(extra, reason="api")
        except ValueError as error:
            raise HTTPException(422, str(error))
        except RuntimeError as error:
            raise HTTPException(409, str(error))
        if b.get("wait"):
            timeout = b.get("timeout", 60)
            if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not 1 <= timeout <= 3600:
                raise HTTPException(422,"timeout must be between 1 and 3600 seconds.")
            st = await run_in_threadpool(sleeper.wait, timeout)
        return st

    @app.post('/rp/sleep/stop')
    def stop_sleep():
        return {'stopped': sleeper.stop()}

    @app.get('/rp/settings')
    def get_settings():
        return learning.get()

    @app.patch('/rp/settings')
    async def update_settings(request: Request):
        try:
            return learning.update(await request.json())
        except ValueError as error:
            raise HTTPException(422, str(error))

    @app.get("/rp/sleep")
    def sleep_status():
        return sleeper.status()

    @app.get("/rp/adapters")
    def adapters():
        return {"active": engine.version, "current_file": read_current(cfg.data_dir, cfg.model),
                "versions": [{k: v for k, v in a.items() if k != "signal_ids"} for a in list_adapters(cfg.data_dir)]}

    @app.post("/rp/activate")
    async def activate(request: Request):
        try:
            value = await request.json()
            v = value.get("version", "base")
            if not isinstance(v,str) or not re.fullmatch(r"base|narration|v[1-9][0-9]*",v):
                raise ValueError()
        except (ValueError, AttributeError):
            raise HTTPException(422, "Choose base, narration or a numbered saved version.")
        try:
            await run_in_threadpool(engine.activate, v)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except (ValueError, RuntimeError):
            raise HTTPException(409, "This adapter could not be loaded. Check its foundation or choose an earlier version.")
        write_current(cfg.data_dir, v)
        return {"ok": True, "active": engine.version}

    @app.post("/rp/reload")
    async def reload():
        await run_in_threadpool(engine.activate, read_current(cfg.data_dir, cfg.model))
        return {"ok": True, "active": engine.version}

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        s = store.stats()
        sl = sleeper.status()
        esc = lambda x: html.escape(str(x if x is not None else ""))
        rows = "".join(
            f"<tr><td>{esc(a['version'])}{' ✓' if a['version'] == engine.version else ''}</td><td>{esc(a.get('items'))}</td>"
            f"<td>{esc(a.get('train_minutes'))}</td><td>{esc(a.get('train_nll_chosen_before'))} → {esc(a.get('train_nll_chosen_after'))}</td>"
            f"<td>{esc(a.get('canary_drift'))}</td><td>{'yes' if a.get('guard_passed') else 'no'}</td></tr>"
            for a in list_adapters(cfg.data_dir)[::-1])
        sigs = "".join(
            f"<tr><td>{esc(x['kind'])}</td><td>{esc(x['weight'])}</td><td class=t>{esc((x['chosen'] or '')[:160])}</td>"
            f"<td class=t>{esc((x['rejected'] or '')[:160])}</td><td>{esc(x['consumed_in'] or 'new')}</td></tr>"
            for x in store.recent_signals(15))
        tail = esc("\n".join(sl["log_tail"]))
        page = f"""<!doctype html><html lang=zh><head><meta charset=utf-8><meta http-equiv=refresh content=4>
<meta name=viewport content="width=device-width,initial-scale=1"><title>rp learning dashboard</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;background:#0f1115;color:#e6e6e6}}h1{{font-size:20px}}
table{{border-collapse:collapse;width:100%;margin:8px 0 20px}}td,th{{border-bottom:1px solid #2a2f3a;padding:6px 8px;text-align:left;vertical-align:top}}
th{{color:#9aa4b2;font-weight:600}}.t{{max-width:420px}}.pill{{display:inline-block;padding:2px 10px;border-radius:99px;background:#1f6feb33;color:#79b8ff;margin-right:8px}}
pre{{background:#161b22;padding:10px;border-radius:6px;white-space:pre-wrap;font-size:12px}}</style></head><body>
<h1>rp · role-play server that learns from chatting</h1>
<p><span class=pill>adapter {esc(engine.version)}</span><span class=pill>{s['generations']} generations</span>
<span class=pill>signals {esc(s['signals'])}</span><span class=pill>{s['new_signals_since_sleep']} new since last sleep</span>
<span class=pill>sleep: {'running…' if sl['running'] else 'idle'}</span></p>
<h3>Recent learning signals</h3><table><tr><th>kind</th><th>w</th><th>preferred (chosen)</th><th>rejected</th><th>learned in</th></tr>{sigs}</table>
<h3>Adapter versions</h3><table><tr><th>version</th><th>items</th><th>minutes</th><th>train NLL (chosen)</th><th>canary drift</th><th>guard</th></tr>{rows}</table>
<h3>Sleep log</h3><pre>{tail}</pre></body></html>"""
        return page

    return app


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--auth-token", default=os.environ.get("RP_AUTH_TOKEN", ""))
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--model", default=os.environ.get("RP_MODEL", DEFAULT_MODEL))
    ap.add_argument("--model-id", default="qwen2.5-0.5b-rp", help="name shown in SillyTavern's model list")
    ap.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--sleep-threads", type=int, default=None)
    ap.add_argument("--max-context", type=int, default=3072)
    ap.add_argument("--max-tokens", type=int, default=300, help="default reply length")
    ap.add_argument("--max-tokens-cap", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.08)
    ap.add_argument("--auto-sleep-signals", type=int, default=10,
                    help="auto-sleep once this many new signals exist and the chat has been idle (0 = off)")
    ap.add_argument("--auto-sleep-idle", type=float, default=300, help="seconds of chat inactivity before auto-sleep")
    return ap


def main():
    cfg = parser().parse_args()
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
