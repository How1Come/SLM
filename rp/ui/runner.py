"""Background jobs and rp.server process for the role-play trainer UI."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ..engine import list_adapters, narration_adapter_dir, read_current
from ..sleep import build_examples
from ..store import Store
from .config import UiConfig

ROOT = Path(__file__).resolve().parents[2]
EPOCH_RE = re.compile(r"epoch (\d+)/(\d+)")
RESULT_RE = re.compile(r"^RESULT (.+)$")


@dataclass
class JobSnapshot:
    id: str
    kind: str
    state: str
    progress: float
    log_tail: list[str]
    result: dict | None
    error: str | None
    started_at: float | None
    finished_at: float | None


@dataclass
class _Job:
    id: str
    kind: str
    state: str = "queued"
    progress: float = 0.0
    log: deque = field(default_factory=lambda: deque(maxlen=4000))
    proc: subprocess.Popen | None = None
    result: dict | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    listeners: list[Callable[[dict], None]] = field(default_factory=list)

    def emit(self, event: dict) -> None:
        for fn in list(self.listeners):
            try:
                fn(event)
            except Exception:
                pass

    def snapshot(self) -> JobSnapshot:
        return JobSnapshot(
            id=self.id,
            kind=self.kind,
            state=self.state,
            progress=self.progress,
            log_tail=list(self.log)[-100:],
            result=self.result,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


class RpServerProcess:
    def __init__(self, cfg: UiConfig) -> None:
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def status(self) -> dict:
        with self._lock:
            running = self.proc is not None and self.proc.poll() is None
            return {
                "running": running,
                "pid": self.proc.pid if running and self.proc else None,
                "endpoint": self.cfg.st_endpoint,
                "model_id": self.cfg.model_id,
                "base_url": self.cfg.rp_base_url,
            }

    def start(self) -> dict:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return self.status()
            cmd = [
                sys.executable,
                "-m",
                "rp.server",
                "--host",
                self.cfg.rp_host,
                "--port",
                str(self.cfg.rp_port),
                "--data-dir",
                self.cfg.data_dir,
                "--model",
                self.cfg.model,
                "--model-id",
                self.cfg.model_id,
            ]
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return self.status()

    def stop(self) -> bool:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                self.proc = None
                return False
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
            return True


class JobManager:
    def __init__(self, cfg: UiConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._current: _Job | None = None
        self._last: dict[str, JobSnapshot] = {}

    def current(self) -> JobSnapshot | None:
        with self._lock:
            if self._current:
                return self._current.snapshot()
            return None

    def last(self, kind: str) -> JobSnapshot | None:
        return self._last.get(kind)

    def subscribe(self, cb: Callable[[dict], None]) -> None:
        with self._lock:
            job = self._current
        if job:
            job.listeners.append(cb)

    def unsubscribe(self, cb: Callable[[dict], None]) -> None:
        with self._lock:
            job = self._current
        if job:
            try:
                job.listeners.remove(cb)
            except ValueError:
                pass

    def stop(self) -> bool:
        with self._lock:
            job = self._current
            if not job or job.state not in ("running", "queued"):
                return False
            if job.proc and job.proc.poll() is None:
                job.proc.terminate()
            job.state = "stopped"
            job.finished_at = time.time()
            job.emit({"type": "state", "state": "stopped"})
            return True

    def start_narration_train(self) -> JobSnapshot:
        return self._start(
            "narration_train",
            [sys.executable, "-m", "rp.train_narration"],
        )

    def start_sleep(self) -> JobSnapshot:
        cmd = [
            sys.executable,
            "-m",
            "rp.sleep",
            "--data-dir",
            self.cfg.data_dir,
            "--model",
            self.cfg.model,
            "--server",
            self.cfg.rp_base_url,
        ]
        return self._start("sleep", cmd)

    def start_narration_eval(self) -> JobSnapshot:
        out = ROOT / "runs" / "rp-narration" / "ui-eval"
        cmd = [
            sys.executable,
            "-m",
            "rp.narration_eval",
            "--versions",
            "base",
            "narration",
            "--data-dir",
            self.cfg.data_dir,
            "--model",
            self.cfg.model,
            "--out",
            str(out),
        ]
        return self._start("narration_eval", cmd, out_base=out)

    def _start(self, kind: str, cmd: list[str], out_base: Path | None = None) -> JobSnapshot:
        with self._lock:
            if self._current and self._current.state in ("running", "queued"):
                raise RuntimeError("a job is already in progress")
            job = _Job(id=uuid.uuid4().hex[:10], kind=kind)
            self._current = job
        threading.Thread(target=self._run, args=(job, cmd, out_base), daemon=True).start()
        return job.snapshot()

    def _run(self, job: _Job, cmd: list[str], out_base: Path | None) -> None:
        job.started_at = time.time()
        job.state = "running"
        job.emit({"type": "state", "state": "running", "kind": job.kind, "cmd": " ".join(cmd)})
        self._log(job, f"[ui] {' '.join(cmd)}\n")
        code = self._stream(job, cmd)
        if job.state == "stopped":
            job.finished_at = time.time()
            return
        if code != 0 and job.result is None:
            job.state = "error"
            job.error = f"exit {code}"
        elif job.result and job.result.get("status") == "failed":
            job.state = "error"
            job.error = job.result.get("reason", "failed")
        else:
            job.state = "done"
            if out_base and (out_base.with_suffix(".md")).exists():
                job.result = job.result or {}
                job.result["eval_md"] = (out_base.with_suffix(".md")).read_text(encoding="utf-8")
        job.progress = 1.0
        job.finished_at = time.time()
        snap = job.snapshot()
        with self._lock:
            self._last[job.kind] = snap
            self._current = None
        job.emit({"type": "state", "state": job.state, "snapshot": asdict(snap)})

    def _stream(self, job: _Job, cmd: list[str]) -> int:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job.proc = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            self._log(job, line if line.endswith("\n") else line + "\n")
            if job.state == "stopped":
                break
        return proc.wait()

    def _log(self, job: _Job, line: str) -> None:
        stripped = line.rstrip("\n")
        job.log.append(stripped)
        m = EPOCH_RE.search(line)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            job.progress = min(cur / max(total, 1), 0.99)
        rm = RESULT_RE.match(stripped)
        if rm:
            try:
                job.result = json.loads(rm.group(1))
            except json.JSONDecodeError:
                pass
        job.emit({"type": "log", "line": stripped, "progress": job.progress, "result": job.result})


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", str(min(4, os.cpu_count() or 4)))
    return env


def collect_overview(cfg: UiConfig) -> dict:
    data_dir = cfg.data_dir
    store = Store(data_dir)
    stats = store.stats()
    signals = store.active_signals()
    examples = build_examples(signals, 400)
    n_preferred = sum(1 for e in examples if e["chosen"] is not None)
    current = read_current(data_dir)
    adapters = list_adapters(data_dir)
    narr_path = narration_adapter_dir(data_dir)

    train_path = cfg.narration_data_path(str(ROOT))
    train_count = 0
    if os.path.exists(train_path):
        with open(train_path, encoding="utf-8") as f:
            train_count = sum(1 for line in f if line.strip())

    narr_meta = None
    if narr_path and os.path.exists(os.path.join(narr_path, "meta.json")):
        with open(os.path.join(narr_path, "meta.json"), encoding="utf-8") as f:
            narr_meta = json.load(f)

    adapter_label = current
    if current == "narration":
        adapter_label = "narration (bundled scene LoRA)"
    elif current.startswith("v"):
        adapter_label = f"{current} (chat sleep LoRA)"

    sleep_ready = n_preferred >= 1
    sleep_reason = (
        f"{n_preferred} preferred reply(ies) ready for sleep"
        if sleep_ready
        else "nothing to learn yet — chat in SillyTavern and use edits, !good, or !fix"
    )

    return {
        "adapter": current,
        "adapter_label": adapter_label,
        "adapters": adapters,
        "data_dir": os.path.abspath(data_dir),
        "narration_train_examples": train_count,
        "narration_meta": narr_meta,
        "narration_adapter_path": narr_path,
        "store": stats,
        "sleep": {
            "ready": sleep_ready,
            "preferred_replies": n_preferred,
            "training_items": len(examples),
            "message": sleep_reason,
        },
        "sillytavern": {
            "endpoint": cfg.st_endpoint,
            "model_id": cfg.model_id,
            "hint": "API = Chat Completion, source = Custom (OpenAI-compatible)",
        },
        "cradle_ui_url": cfg.cradle_ui_url,
    }
