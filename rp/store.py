"""SQLite store for generations and learning signals.

Tables
  generations  every reply the server produced: the exact prompt messages, output, turn key, adapter version
  signals      learning signals derived from chatting:
                 edit   user edited a reply in SillyTavern  -> chosen = edited text, rejected = model text
                 fix    explicit "!fix <text>" correction    -> chosen = given text,  rejected = model text
                 good   explicit thumbs-up                   -> chosen = model text
                 bad    explicit thumbs-down                 -> rejected = model text
                 swipe  user regenerated / swiped a reply    -> rejected = model text (weak); once the user
                        keeps one of the swipes, chosen = the kept swipe
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
  id TEXT PRIMARY KEY,
  ts REAL NOT NULL,
  turn_key TEXT NOT NULL,
  messages TEXT NOT NULL,
  output TEXT NOT NULL,
  prefill TEXT NOT NULL DEFAULT '',
  adapter TEXT NOT NULL,
  params TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'new',
  latency_s REAL,
  first_token_s REAL,
  prompt_tokens INTEGER,
  completion_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS gen_key ON generations(turn_key, ts);
CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  kind TEXT NOT NULL,
  turn_key TEXT NOT NULL,
  gen_id TEXT,
  messages TEXT NOT NULL,
  chosen TEXT,
  rejected TEXT,
  weight REAL NOT NULL,
  active INTEGER NOT NULL DEFAULT 1,
  consumed_in TEXT
);
CREATE INDEX IF NOT EXISTS sig_key ON signals(turn_key, kind);
"""

WEIGHTS = {"edit": 1.0, "fix": 1.0, "good": 0.5, "bad": 1.0, "swipe": 0.3}


class Store:
    def __init__(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "rp.db")
        self._lock = threading.Lock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ---------- generations ----------
    def add_generation(self, turn_key, messages, output, adapter, params, prefill="", latency_s=None,
                       first_token_s=None, prompt_tokens=None, completion_tokens=None) -> str:
        gid = "gen-" + uuid.uuid4().hex[:16]
        with self._lock:
            self.db.execute(
                "INSERT INTO generations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (gid, time.time(), turn_key, json.dumps(messages, ensure_ascii=False), output, prefill, adapter,
                 json.dumps(params), "new", latency_s, first_token_s, prompt_tokens, completion_tokens))
            self.db.commit()
        return gid

    def generations_for_key(self, turn_key):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM generations WHERE turn_key=? ORDER BY ts", (turn_key,))]

    def get_generation(self, gid):
        r = self.db.execute("SELECT * FROM generations WHERE id=?", (gid,)).fetchone()
        return dict(r) if r else None

    def last_generation(self):
        r = self.db.execute("SELECT * FROM generations ORDER BY ts DESC LIMIT 1").fetchone()
        return dict(r) if r else None

    def set_status(self, gid, status):
        with self._lock:
            self.db.execute("UPDATE generations SET status=? WHERE id=?", (status, gid))
            self.db.commit()

    # ---------- signals ----------
    def find_signals(self, turn_key, kind=None, gen_id=None, active_only=True):
        q, a = "SELECT * FROM signals WHERE turn_key=?", [turn_key]
        if kind:
            q += " AND kind=?"; a.append(kind)
        if gen_id:
            q += " AND gen_id=?"; a.append(gen_id)
        if active_only:
            q += " AND active=1"
        return [dict(r) for r in self.db.execute(q, a)]

    def add_signal(self, kind, turn_key, gen_id, messages, chosen=None, rejected=None, weight=None) -> int:
        w = WEIGHTS[kind] if weight is None else weight
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO signals (ts, kind, turn_key, gen_id, messages, chosen, rejected, weight) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), kind, turn_key, gen_id,
                 messages if isinstance(messages, str) else json.dumps(messages, ensure_ascii=False),
                 chosen, rejected, w))
            self.db.commit()
            return cur.lastrowid

    def update_signal(self, sid, **fields):
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self.db.execute(f"UPDATE signals SET {cols} WHERE id=?", (*fields.values(), sid))
            self.db.commit()

    def active_signals(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM signals WHERE active=1 ORDER BY ts")]

    def recent_signals(self, n=20):
        return [dict(r) for r in self.db.execute("SELECT * FROM signals ORDER BY ts DESC LIMIT ?", (n,))]

    def mark_consumed(self, ids, version):
        with self._lock:
            self.db.executemany("UPDATE signals SET consumed_in=? WHERE id=?", [(version, i) for i in ids])
            self.db.commit()

    def stats(self):
        g = self.db.execute("SELECT COUNT(*), AVG(latency_s), AVG(first_token_s), "
                            "SUM(completion_tokens), SUM(latency_s) FROM generations").fetchone()
        by_kind = {r[0]: r[1] for r in self.db.execute(
            "SELECT kind, COUNT(*) FROM signals WHERE active=1 GROUP BY kind")}
        unconsumed = self.db.execute(
            "SELECT COUNT(*) FROM signals WHERE active=1 AND consumed_in IS NULL "
            "AND (chosen IS NOT NULL OR kind IN ('bad'))").fetchone()[0]
        status = {r[0]: r[1] for r in self.db.execute("SELECT status, COUNT(*) FROM generations GROUP BY status")}
        return {
            "generations": g[0], "avg_latency_s": g[1], "avg_first_token_s": g[2],
            "avg_tokens_per_s": (g[3] / g[4]) if g[4] else None,
            "generation_status": status, "signals": by_kind, "new_signals_since_sleep": unconsumed,
        }
