from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

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
  completion_tokens INTEGER,
  session_id TEXT,
  scope_id TEXT NOT NULL DEFAULT 'legacy'
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
  consumed_in TEXT,
  session_id TEXT,
  scope_id TEXT NOT NULL DEFAULT 'legacy',
  updated_ts REAL NOT NULL DEFAULT 0,
  record_version INTEGER NOT NULL DEFAULT 1,
  review_status TEXT NOT NULL DEFAULT 'accepted'
);
CREATE INDEX IF NOT EXISTS sig_key ON signals(turn_key, kind);
"""

WEIGHTS = {"edit": 1.0, "fix": 1.0, "good": 0.5, "bad": 1.0, "swipe": 0.3}
TRUSTED_SQL = "active=1 AND (NULLIF(TRIM(chosen), '') IS NOT NULL OR kind='bad')"


class Store:
    def __init__(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "rp.db")
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            db.execute("BEGIN IMMEDIATE")
            self._migrate(db)
            db.commit()

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.close()
        return False

    @contextmanager
    def _connection(self, write=False):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if write:
                db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _migrate(self, db):
        additions = {
            "generations": {"session_id": "TEXT", "scope_id": "TEXT NOT NULL DEFAULT 'legacy'"},
            "signals": {
                "session_id": "TEXT", "scope_id": "TEXT NOT NULL DEFAULT 'legacy'",
                "updated_ts": "REAL NOT NULL DEFAULT 0", "record_version": "INTEGER NOT NULL DEFAULT 1",
                "review_status": "TEXT NOT NULL DEFAULT 'accepted'",
            },
        }
        legacy_signals = "scope_id" not in {r[1] for r in db.execute("PRAGMA table_info(signals)")}
        for table, columns in additions.items():
            existing = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
            for name, declaration in columns.items():
                if name not in existing:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
        if legacy_signals:
            db.execute("UPDATE signals SET active=0, review_status='needs_review', updated_ts=ts")
        db.execute("CREATE INDEX IF NOT EXISTS sig_pending ON signals(active, consumed_in, ts)")
        db.execute("PRAGMA user_version=2")

    def add_generation(self, turn_key, messages, output, adapter, params, prefill="", latency_s=None,
                       first_token_s=None, prompt_tokens=None, completion_tokens=None,
                       session_id=None, scope_id=None) -> str:
        gid = "gen-" + uuid.uuid4().hex[:16]
        scope_id = scope_id or (turn_key.split(":", 1)[0] if ":" in turn_key else "legacy")
        with self._connection(write=True) as db:
            db.execute(
                "INSERT INTO generations (id,ts,turn_key,messages,output,prefill,adapter,params,status,"
                "latency_s,first_token_s,prompt_tokens,completion_tokens,session_id,scope_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (gid, time.time(), turn_key, json.dumps(messages, ensure_ascii=False), output, prefill, adapter,
                 json.dumps(params), "new", latency_s, first_token_s, prompt_tokens, completion_tokens,
                 session_id, scope_id))
        return gid

    def _rows(self, query, args=()):
        with self._connection() as db:
            return [dict(r) for r in db.execute(query, args)]

    def generations_for_key(self, turn_key):
        return self._rows("SELECT * FROM generations WHERE turn_key=? ORDER BY ts,id", (turn_key,))

    def get_generation(self, gid):
        rows = self._rows("SELECT * FROM generations WHERE id=?", (gid,))
        return rows[0] if rows else None

    def last_generation(self):
        rows = self._rows("SELECT * FROM generations ORDER BY ts DESC,id DESC LIMIT 1")
        return rows[0] if rows else None

    def set_status(self, gid, status):
        if status not in {"new", "accepted", "swiped", "edited", "other"}:
            raise ValueError("Unsupported generation status")
        with self._connection(write=True) as db:
            db.execute("UPDATE generations SET status=? WHERE id=?", (status, gid))

    def find_signals(self, turn_key, kind=None, gen_id=None, active_only=True):
        query, args = "SELECT * FROM signals WHERE turn_key=?", [turn_key]
        if kind:
            query += " AND kind=?"
            args.append(kind)
        if gen_id is not None:
            query += " AND gen_id=?"
            args.append(gen_id)
        if active_only:
            query += " AND active=1"
        return self._rows(query + " ORDER BY ts,id", args)

    def add_signal(self, kind, turn_key, gen_id, messages, chosen=None, rejected=None, weight=None,
                   session_id=None, scope_id=None) -> int:
        if kind not in WEIGHTS:
            raise ValueError("Unsupported learning signal")
        now = time.time()
        with self._connection(write=True) as db:
            generation = db.execute("SELECT session_id,scope_id FROM generations WHERE id=?", (gen_id,)).fetchone()
            if generation:
                session_id = session_id if session_id is not None else generation["session_id"]
                scope_id = scope_id or generation["scope_id"]
            scope_id = scope_id or (turn_key.split(":", 1)[0] if ":" in turn_key else "legacy")
            cur = db.execute(
                "INSERT INTO signals (ts,kind,turn_key,gen_id,messages,chosen,rejected,weight,session_id,scope_id,updated_ts) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (now, kind, turn_key, gen_id,
                 messages if isinstance(messages, str) else json.dumps(messages, ensure_ascii=False),
                 chosen, rejected, WEIGHTS[kind] if weight is None else weight, session_id, scope_id, now))
            return cur.lastrowid

    def get_signal(self, sid):
        rows = self._rows("SELECT * FROM signals WHERE id=?", (sid,))
        return rows[0] if rows else None

    def update_signal(self, sid, **fields):
        allowed = {"chosen", "rejected", "weight", "active", "consumed_in", "review_status"}
        if not fields:
            return
        if set(fields) - allowed:
            raise ValueError("Unsupported signal fields")
        if "active" in fields and fields["active"] not in (0, 1, False, True):
            raise ValueError("Active must be a boolean")
        if "review_status" in fields and fields["review_status"] not in {"accepted", "excluded", "needs_review"}:
            raise ValueError("Unsupported review status")
        with self._connection(write=True) as db:
            row = db.execute("SELECT * FROM signals WHERE id=?", (sid,)).fetchone()
            if row is None:
                raise KeyError(sid)
            changed = {key: value for key, value in fields.items() if row[key] != value}
            if not changed:
                return
            if set(changed) - {"consumed_in"}:
                changed["consumed_in"] = None
            changed["updated_ts"] = time.time()
            columns = ", ".join(f"{key}=?" for key in changed)
            db.execute(f"UPDATE signals SET {columns}, record_version=record_version+1 WHERE id=?",
                       (*changed.values(), sid))

    def review_signal(self, sid, active):
        if not isinstance(active, bool):
            raise ValueError("Active must be a boolean")
        self.update_signal(sid, active=int(active), review_status="accepted" if active else "excluded")
        return self.get_signal(sid)

    def active_signals(self):
        return self._rows(f"SELECT * FROM signals WHERE {TRUSTED_SQL} ORDER BY ts,id")

    def fresh_signals(self):
        return self._rows(f"SELECT * FROM signals WHERE {TRUSTED_SQL} AND consumed_in IS NULL ORDER BY ts,id")

    def recent_signals(self, n=20, offset=0, active=None, kind=None):
        if not 1 <= n <= 500 or offset < 0:
            raise ValueError("Use a page size from 1 to 500 and a non-negative offset")
        query, args = "SELECT * FROM signals WHERE 1=1", []
        if active is not None:
            query += " AND active=?"
            args.append(int(active))
        if kind is not None:
            if kind not in WEIGHTS:
                raise ValueError("Unsupported learning signal")
            query += " AND kind=?"
            args.append(kind)
        return self._rows(query + " ORDER BY ts DESC,id DESC LIMIT ? OFFSET ?", [*args, n, offset])

    def mark_consumed(self, ids, version, expected_versions=None):
        with self._connection(write=True) as db:
            for sid in ids:
                if expected_versions is not None:
                    expected = expected_versions.get(sid)
                    if expected is None:
                        continue
                    db.execute("UPDATE signals SET consumed_in=? WHERE id=? AND active=1 AND record_version=?",
                               (version, sid, expected))
                else:
                    db.execute("UPDATE signals SET consumed_in=? WHERE id=? AND active=1", (version, sid))

    def export_feedback(self, active_only=True):
        records = self.active_signals() if active_only else self._rows("SELECT * FROM signals ORDER BY ts,id")
        exported = []
        for signal in records:
            try:
                messages = json.loads(signal["messages"])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(messages, list) or any(
                not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str) for message in messages
            ):
                continue
            context = [{"role": message["role"], "content": message["content"]} for message in messages]
            metadata = {key: signal[key] for key in (
                "ts", "kind", "weight", "active", "consumed_in", "session_id", "scope_id", "record_version", "review_status"
            )}
            metadata["source"] = "human_feedback"
            metadata["status"] = "approved" if signal["active"] else "rejected"
            record = {"id": f"feedback-{signal['id']}", "messages": context, "metadata": metadata}
            if signal["chosen"]:
                record["messages"] = [*context, {"role": "assistant", "content": signal["chosen"]}]
            if signal["chosen"] and signal["rejected"]:
                record["preference"] = {"prompt": context, "chosen": signal["chosen"], "rejected": signal["rejected"]}
            elif signal["rejected"]:
                record["rejected"] = signal["rejected"]
            exported.append(record)
        return exported

    def stats(self):
        with self._connection() as db:
            generation = db.execute("SELECT COUNT(*),AVG(latency_s),AVG(first_token_s),SUM(completion_tokens),"
                                    "SUM(latency_s) FROM generations").fetchone()
            by_kind = {row[0]: row[1] for row in db.execute(
                f"SELECT kind,COUNT(*) FROM signals WHERE {TRUSTED_SQL} GROUP BY kind")}
            unconsumed = db.execute(
                f"SELECT COUNT(*) FROM signals WHERE {TRUSTED_SQL} AND consumed_in IS NULL").fetchone()[0]
            status = {row[0]: row[1] for row in db.execute("SELECT status,COUNT(*) FROM generations GROUP BY status")}
            review = db.execute("SELECT COUNT(*) FROM signals WHERE review_status='needs_review'").fetchone()[0]
            excluded = db.execute("SELECT COUNT(*) FROM signals WHERE active=0").fetchone()[0]
            total = db.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        return {
            "generations": generation[0], "avg_latency_s": generation[1], "avg_first_token_s": generation[2],
            "avg_tokens_per_s": generation[3] / generation[4] if generation[4] else None,
            "generation_status": status, "signals": by_kind, "new_signals_since_sleep": unconsumed,
            "signals_needing_review": review, "excluded_signals": excluded,
            "total_signals": total,
        }
