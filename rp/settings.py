from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

DEFAULTS = {
    "enabled": True,
    "auto_sleep": True,
    "mode": "idle",
    "min_signals": 10,
    "idle_seconds": 300,
    "max_minutes": 5.0,
    "epochs": 2,
    "threads": 1,
    "guard_max_drift": 0.3,
}
BOUNDS = {
    "min_signals": (1, 10000, int),
    "idle_seconds": (10, 86400, int),
    "max_minutes": (0.1, 120, float),
    "epochs": (1, 20, int),
    "threads": (1, 64, int),
    "guard_max_drift": (0, 10, float),
}
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_LOCK = threading.Lock()


def validate_settings(values: dict) -> dict:
    if not isinstance(values, dict):
        raise ValueError("Learning settings must be a JSON object")
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown learning setting: " + ", ".join(sorted(unknown)))
    result = dict(values)
    for key in ("enabled", "auto_sleep"):
        if key in values and not isinstance(values[key], bool):
            raise ValueError(f"{key} must be true or false")
    if "mode" in values and values["mode"] not in {"collect_only", "idle", "background"}:
        raise ValueError("mode must be collect_only, idle, or background")
    for key, (minimum, maximum, value_type) in BOUNDS.items():
        if key not in values:
            continue
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}")
        if value_type is int and int(value) != value:
            raise ValueError(f"{key} must be a whole number")
        result[key] = value_type(value)
    return result


class LearningSettings:
    def __init__(self, data_dir: str):
        self.path = Path(data_dir) / "learning.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCKS_LOCK:
            self._lock = _LOCKS.setdefault(str(self.path.resolve()), threading.RLock())

    @contextmanager
    def _locked(self):
        with self._lock:
            with self.path.with_suffix(".lock").open("a+b") as lock_file:
                try:
                    import fcntl
                except ImportError:
                    fcntl = None
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read(self):
        if not self.path.exists():
            return dict(DEFAULTS)
        try:
            values = json.loads(self.path.read_text(encoding="utf-8"))
            return {**DEFAULTS, **validate_settings(values)}
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("Learning settings could not be read; restore or remove learning.json") from error

    def get(self):
        with self._locked():
            return self._read()

    def update(self, values: dict):
        validated = validate_settings(values)
        with self._locked():
            updated = {**self._read(), **validated}
            descriptor, temporary = tempfile.mkstemp(prefix=".learning-", suffix=".json", dir=self.path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                    json.dump(updated, output, ensure_ascii=False, indent=2)
                    output.write("\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return updated


def get_learning_settings(data_dir: str) -> dict:
    return LearningSettings(data_dir).get()


def update_learning_settings(data_dir: str, values: dict) -> dict:
    return LearningSettings(data_dir).update(values)
