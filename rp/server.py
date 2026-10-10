"""OpenAI-compatible role-play server that learns from chatting.

    python -m rp.server [--port 8619] [--data-dir rp_data] [--model Qwen/Qwen2.5-0.5B-Instruct]

SillyTavern: API = Chat Completion, source = Custom (OpenAI-compatible), endpoint = http://127.0.0.1:8619/v1
"""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from . import DEFAULT_DATA_DIR, DEFAULT_MODEL, DEFAULT_PORT
from .engine import Engine, list_adapters, read_current, write_current
from .learn import ACK_PREFIX, Learner
from .store import Store


class SleepManager:
    """Runs `python -m rp.sleep` in a subprocess (isolated memory, killable) and hot-reloads on success."""

    def __init__(self, cfg, engine: Engine):
        self.cfg, self.engine = cfg, engine
        self.proc = None
        self.started = self.finished = None
        self.last_result = None
        self.log_path = os.path.join(cfg.data_dir, "sleep.log")
        self._lock = threading.Lock()

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, extra: dict | None = None, reason="manual"):
        with self._lock:
            if self.running:
                return self.status()
            cmd = [sys.executable, "-m", "rp.sleep", "--data-dir", self.cfg.data_dir, "--model", self.cfg.model]
            if self.cfg.sleep_threads:
                cmd += ["--threads", str(self.cfg.sleep_threads)]
            for k, v in (extra or {}).items():
                flag = "--" + k.replace("_", "-")
                if v is True:
                    cmd.append(flag)
                elif v not in (None, False):
                    cmd += [flag, str(v)]
            logf = open(self.log_path, "a")
            logf.write(f"\n=== sleep started ({reason}) {time.ctime()} :: {' '.join(cmd[3:])}\n")
            logf.flush()
            self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.started, self.finished, self.last_result = time.time(), None, None
            threading.Thread(target=self._wait, args=(self.proc, logf), daemon=True).start()
            return self.status()

    def _wait(self, proc, logf):
        code = proc.wait()
        logf.close()
        result = {"status": "failed", "exit_code": code}
        for line in reversed(self._tail(200)):
            if line.startswith("RESULT "):
                result = json.loads(line[7:])
                break
        if result.get("activated"):
            try:
                self.engine.activate(read_current(self.cfg.data_dir))
                result["reloaded"] = True
            except Exception as e:
                result["reload_error"] = str(e)
        self.last_result, self.finished = result, time.time()

    def _tail(self, n=30):
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, errors="replace") as f:
            return [l.rstrip("\n") for l in f.readlines()[-n:]]

    def wait(self, timeout=None):
        t0 = time.time()
        while self.running or (self.started and not self.finished):
            if timeout and time.time() - t0 > timeout:
                break
            time.sleep(0.5)
        return self.status()

    def status(self):
        return {"running": self.running, "started": self.started, "finished": self.finished,
                "last_result": self.last_result, "log_tail": self._tail(12)}


def create_app(cfg) -> FastAPI:
    store = Store(cfg.data_dir)
    engine = Engine(cfg.model, cfg.data_dir, dtype=cfg.dtype, threads=cfg.threads, max_context=cfg.max_context)
    learner = Learner(store)
    sleeper = SleepManager(cfg, engine)
    state = {"last_request": time.time()}
    app = FastAPI(title="rp: role-play server that learns from chatting")
    model_id = cfg.model_id

    def model_label():
        return f"{model_id}@{engine.version}"

    # ---------- auto sleep ----------
    def auto_sleep_loop():
        while True:
            time.sleep(20)
            if cfg.auto_sleep_signals <= 0 or sleeper.running:
                continue
            new = store.stats()["new_signals_since_sleep"]
            idle = time.time() - state["last_request"]
            if new >= cfg.auto_sleep_signals and idle >= cfg.auto_sleep_idle:
                sleeper.start(reason=f"auto: {new} new signals, idle {idle:.0f}s")

    threading.Thread(target=auto_sleep_loop, daemon=True).start()

    # ---------- OpenAI API ----------
    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "rp",
             "adapter": engine.version, "base_model": cfg.model}]}

    def ack_text(a):
        if a.command in ("good", "bad", "fix"):
            msg = learner.apply_command(a)
        elif a.command == "sleep":
            st = sleeper.start(reason="chat command")
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
        body = await request.json()
        state["last_request"] = time.time()
        msgs = body.get("messages") or []
        if not msgs:
            raise HTTPException(400, "messages is required")
        a = await run_in_threadpool(learner.analyze, msgs)
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
            rep = cfg.repetition_penalty + 0.25 * fp
        stop = body.get("stop") or []
        params = dict(max_tokens=min(int(max_tokens), cfg.max_tokens_cap), temperature=temperature,
                      top_p=body.get("top_p", cfg.top_p), top_k=body.get("top_k", 0) or 0,
                      repetition_penalty=rep, seed=body.get("seed"),
                      stop=[stop] if isinstance(stop, str) else list(stop))
        cancel = threading.Event()
        info: dict = {}

        def record(text):
            gid = store.add_generation(
                a.turn_key, a.messages, text, info.get("adapter", engine.version), params, prefill=a.prefill,
                latency_s=info.get("latency_s"), first_token_s=info.get("first_token_s"),
                prompt_tokens=info.get("prompt_tokens"), completion_tokens=info.get("completion_tokens"))
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
            text, _ = await run_in_threadpool(lambda: engine.generate(a.messages, a.prefill, info=info, **params))
            gid = await run_in_threadpool(record, text)
            pt, ct = info.get("prompt_tokens", 0), info.get("completion_tokens", 0)
            return completion(cid, created, text, finish_reason(),
                              {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}, gid)

        async def events():
            produced = []
            try:
                yield chunk(cid, created, {"role": "assistant", "content": ""})
                async for delta in iterate_in_threadpool(engine.stream(a.messages, a.prefill, cancel=cancel, info=info, **params)):
                    produced.append(delta)
                    yield chunk(cid, created, {"content": delta})
                    if await request.is_disconnected():
                        cancel.set()
                        break
                yield chunk(cid, created, {}, finish_reason())
                yield "data: [DONE]\n\n"
            finally:
                cancel.set()
                record("".join(produced))

        return StreamingResponse(events(), media_type="text/event-stream")

    # ---------- learning admin ----------
    @app.get("/rp/stats")
    def stats():
        return {"adapter": engine.version, "model": cfg.model, **store.stats(),
                "sleep": sleeper.status(), "auto_sleep": {"signals": cfg.auto_sleep_signals, "idle_s": cfg.auto_sleep_idle}}

    @app.get("/rp/signals")
    def signals(n: int = 20):
        return store.recent_signals(n)

    @app.post("/rp/feedback")
    async def feedback(request: Request):
        b = await request.json()
        g = store.get_generation(b["gen_id"]) if b.get("gen_id") else store.last_generation()
        if not g:
            raise HTTPException(404, "no generation to give feedback on")
        full = (g["prefill"] or "") + g["output"]
        rating, corr = b.get("rating"), b.get("correction")
        if corr:
            sid = store.add_signal("fix", g["turn_key"], g["id"], g["messages"], chosen=corr, rejected=full)
        elif rating == "good":
            sid = store.add_signal("good", g["turn_key"], g["id"], g["messages"], chosen=full)
        elif rating == "bad":
            sid = store.add_signal("bad", g["turn_key"], g["id"], g["messages"], rejected=full)
        else:
            raise HTTPException(400, "give rating=good|bad or a correction")
        return {"ok": True, "signal_id": sid, "gen_id": g["id"]}

    @app.post("/rp/sleep")
    async def sleep(request: Request):
        try:
            b = await request.json()
        except Exception:
            b = {}
        st = sleeper.start(b.get("args"), reason="api")
        if b.get("wait"):
            st = await run_in_threadpool(sleeper.wait, b.get("timeout"))
        return st

    @app.get("/rp/sleep")
    def sleep_status():
        return sleeper.status()

    @app.get("/rp/adapters")
    def adapters():
        return {"active": engine.version, "current_file": read_current(cfg.data_dir),
                "versions": [{k: v for k, v in a.items() if k != "signal_ids"} for a in list_adapters(cfg.data_dir)]}

    @app.post("/rp/activate")
    async def activate(request: Request):
        v = (await request.json()).get("version", "base")
        try:
            await run_in_threadpool(engine.activate, v)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        write_current(cfg.data_dir, v)
        return {"ok": True, "active": engine.version}

    @app.post("/rp/reload")
    async def reload():
        await run_in_threadpool(engine.activate, read_current(cfg.data_dir))
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
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--model", default=os.environ.get("RP_MODEL", DEFAULT_MODEL))
    ap.add_argument("--model-id", default="qwen2.5-0.5b-rp", help="name shown in SillyTavern's model list")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
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
