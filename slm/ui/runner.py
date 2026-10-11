"""Subprocess training jobs for the Cradle trainer UI."""

from __future__ import annotations

import json
import math
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

from .presets import DEFAULT_BUDGET, DEFAULT_SEED, PRESETS

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
LAST_JOB_FILE = RUNS / ".ui_last_job.json"
PROGRESS_RE = re.compile(r"\[\s*(\d+(?:\.\d+)?)%\]")
PPL_RE = re.compile(r"ppl\s+(\d+(?:\.\d+)?)")
DONE_RE = re.compile(r"\[done\] saved (.+?) \|")


@dataclass
class JobSnapshot:
    id: str
    state: str
    preset_id: str
    name: str
    progress: float
    stories_ppl: float | None
    childes_ppl: float | None
    log_tail: list[str]
    run_dir: str | None
    checkpoint: str | None
    sample: str | None
    error: str | None
    started_at: float | None
    finished_at: float | None


@dataclass
class _Job:
    id: str
    preset_id: str
    name: str
    state: str = "queued"
    progress: float = 0.0
    stories_ppl: float | None = None
    childes_ppl: float | None = None
    log: deque = field(default_factory=lambda: deque(maxlen=4000))
    proc: subprocess.Popen | None = None
    prep_proc: subprocess.Popen | None = None
    run_dir: Path | None = None
    checkpoint: Path | None = None
    sample: str | None = None
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
            state=self.state,
            preset_id=self.preset_id,
            name=self.name,
            progress=self.progress,
            stories_ppl=self.stories_ppl,
            childes_ppl=self.childes_ppl,
            log_tail=list(self.log)[-80:],
            run_dir=str(self.run_dir) if self.run_dir else None,
            checkpoint=str(self.checkpoint) if self.checkpoint else None,
            sample=self.sample,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: _Job | None = None
        self._last: JobSnapshot | None = self._read_last()

    def current(self) -> JobSnapshot | None:
        with self._lock:
            if self._current:
                return self._current.snapshot()
            return self._last

    def _read_last(self) -> JobSnapshot | None:
        if not LAST_JOB_FILE.exists():
            return None
        try:
            data = json.loads(LAST_JOB_FILE.read_text())
            return JobSnapshot(**data)
        except (json.JSONDecodeError, TypeError):
            return None

    def _persist_last(self, snap: JobSnapshot) -> None:
        RUNS.mkdir(parents=True, exist_ok=True)
        LAST_JOB_FILE.write_text(json.dumps(asdict(snap), indent=2))

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
            if not job or job.state not in ("preparing", "running"):
                return False
            for proc in (job.proc, job.prep_proc):
                if proc and proc.poll() is None:
                    proc.terminate()
            job.state = "stopped"
            job.finished_at = time.time()
            job.emit({"type": "state", "state": "stopped"})
            return True

    def start(self, preset_id: str, budget: float | None = None, seed: int | None = None) -> JobSnapshot:
        if preset_id not in PRESETS:
            raise ValueError(f"unknown preset {preset_id}")
        with self._lock:
            if self._current and self._current.state in ("preparing", "running", "queued"):
                raise RuntimeError("a run is already in progress")
            job = _Job(
                id=uuid.uuid4().hex[:10],
                preset_id=preset_id,
                name=f"ui-{PRESETS[preset_id].default_name}-{job_id_suffix()}",
            )
            self._current = job
        threading.Thread(target=self._run_job, args=(job, budget, seed), daemon=True).start()
        return job.snapshot()

    def _run_job(self, job: _Job, budget: float | None, seed: int | None) -> None:
        py = sys.executable
        preset = PRESETS[job.preset_id]
        budget_val = budget if budget is not None else DEFAULT_BUDGET
        seed_val = seed if seed is not None else DEFAULT_SEED
        job.started_at = time.time()
        job.state = "preparing"
        job.emit({"type": "state", "state": "preparing"})

        prepared = ROOT / "data" / "prepared.pt"
        if not prepared.exists():
            self._log(job, "[ui] downloading data and building tokenizer (first run only)…\n")
            code = self._stream_process(job, [py, "-m", "slm.data"], env=self._env(), attr="prep_proc")
            if code != 0 and job.state != "stopped":
                job.state = "error"
                job.error = f"data prep failed (exit {code})"
                job.finished_at = time.time()
                job.emit({"type": "state", "state": "error", "error": job.error})
                return

        if job.state == "stopped":
            job.finished_at = time.time()
            return

        cmd = [
            py,
            "-m",
            "slm.train",
            "--recipe",
            preset.recipe,
            "--name",
            job.name,
            "--seed",
            str(int(seed_val)),
            *preset.train_args,
        ]
        if job.preset_id != "smoke":
            cmd.extend(["--budget-tokens", str(budget_val)])

        job.state = "running"
        job.emit({"type": "state", "state": "running", "cmd": " ".join(cmd)})
        self._log(job, f"[ui] starting: {' '.join(cmd)}\n")

        code = self._stream_process(job, cmd, env=self._env(), attr="proc")
        if job.state == "stopped":
            job.finished_at = time.time()
            return
        if code != 0:
            job.state = "error"
            job.error = f"training failed (exit {code})"
            job.finished_at = time.time()
            job.emit({"type": "state", "state": "error", "error": job.error})
            return

        job.run_dir = RUNS / job.name
        job.checkpoint = job.run_dir / "model.pt"
        self._ingest_log_json(job)
        job.state = "done"
        job.progress = 1.0
        job.finished_at = time.time()
        snap = job.snapshot()
        with self._lock:
            self._last = snap
        self._persist_last(snap)
        job.emit({"type": "state", "state": "done", "snapshot": asdict(snap)})

    def _env(self) -> dict[str, str]:
        import os

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("OMP_NUM_THREADS", str(min(4, os.cpu_count() or 4)))
        return env

    def _stream_process(self, job: _Job, cmd: list[str], env: dict[str, str], attr: str) -> int:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        setattr(job, attr, proc)
        assert proc.stdout is not None
        for line in proc.stdout:
            self._log(job, line if line.endswith("\n") else line + "\n")
            if job.state == "stopped":
                break
        return proc.wait()

    def _log(self, job: _Job, line: str) -> None:
        job.log.append(line.rstrip("\n"))
        m = PROGRESS_RE.search(line)
        if m:
            job.progress = min(float(m.group(1)) / 100.0, 1.0)
        pm = PPL_RE.search(line)
        if pm:
            job.stories_ppl = float(pm.group(1))
        dm = DONE_RE.search(line)
        if dm:
            job.run_dir = Path(dm.group(1).strip())
            job.checkpoint = job.run_dir / "model.pt"
        job.emit({"type": "log", "line": line.rstrip("\n"), "progress": job.progress})

    def _ingest_log_json(self, job: _Job) -> None:
        log_path = (job.run_dir or RUNS / job.name) / "log.json"
        if not log_path.exists():
            return
        try:
            data = json.loads(log_path.read_text())
        except json.JSONDecodeError:
            return
        curve = data.get("curve") or []
        if curve:
            last = curve[-1]
            sv = last.get("stories_val")
            cv = last.get("childes_val")
            if sv is not None:
                job.stories_ppl = round(math.exp(sv), 2)
            if cv is not None:
                job.childes_ppl = round(math.exp(cv), 2)
        diary = data.get("diary") or []
        if diary:
            samples = diary[-1].get("samples") or []
            if samples:
                job.sample = samples[0]
        if not job.sample and job.checkpoint and job.checkpoint.exists():
            job.sample = _quick_sample(job.checkpoint)


def job_id_suffix() -> str:
    return time.strftime("%H%M%S") + uuid.uuid4().hex[:4]


def _quick_sample(checkpoint: Path) -> str | None:
    try:
        import torch

        from slm.chat import complete
        from slm.data import load_tokenizer
        from slm.evaluate import load_model

        model = load_model(str(checkpoint))
        tok = load_tokenizer()
        prompt = "once upon a time"
        return complete(model, tok, prompt, max_new=60, temperature=0.8, top_k=40, n=1)[0]
    except Exception:
        return None
