"""Turn SillyTavern chat requests into learning signals.

SillyTavern resends the whole chat history with every request. Each assistant message in that history is
matched back to the generation(s) the server logged for the same turn:

* history text == a logged output        -> the user kept that reply ("accepted"); any other swipes of the
                                            same turn become weak preference pairs (kept > swiped-away)
* history text differs from every output  -> the user edited the reply in SillyTavern: a recast.
                                            chosen = edited text (preferred), rejected = model text
* a new request for a turn that already   -> regenerate / swipe: weak rejection of the earlier reply
  has a logged reply

A turn is identified by the last non-system message before the reply plus the message before it (normalised,
whitespace-free), so it survives changes to the system prompt, world info, and context trimming.

Explicit hooks typed as a chat message (SillyTavern reserves "/" for its own commands, so these use "!"):
  !good / !好          the previous reply was good
  !bad / !差           the previous reply was bad
  !fix <text> / !改    replace the previous reply with <text> (a correction)
  !sleep / !睡觉       consolidate now (train + hot-reload the LoRA adapter)
  !status / !状态      show learning stats
The command and the server's acknowledgement are stripped from later prompts.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field

from .store import Store

ACK_PREFIX = "〔学习〕"
COMMANDS = {
    "good": "good", "好": "good", "赞": "good",
    "bad": "bad", "差": "bad", "踩": "bad",
    "fix": "fix", "改": "fix",
    "sleep": "sleep", "睡觉": "sleep",
    "status": "status", "状态": "status",
}
_CMD_RE = re.compile(r"^\s*[!！]\s*(" + "|".join(sorted(map(re.escape, COMMANDS), key=len, reverse=True)) + r")(?=\s|$|[:：])[:：]?\s*(.*)$", re.S | re.I)


def parse_command(text: str):
    m = _CMD_RE.match(text or "")
    if not m:
        return None
    return COMMANDS[m.group(1).lower()], m.group(2).strip()


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type", "text") == "text")
    return "" if content is None else str(content)


def clean_messages(raw) -> list[dict]:
    """Plain {role, content} list; feedback commands and acks removed; leading system messages merged."""
    out = []
    for m in raw:
        role = m.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user" if role != "tool" else "system"
        text = content_text(m.get("content"))
        if role == "user" and parse_command(text):
            continue
        if role == "assistant" and text.lstrip().startswith(ACK_PREFIX):
            continue
        out.append({"role": role, "content": text})
    merged, i = [], 0
    while i < len(out) and out[i]["role"] == "system":
        i += 1
    if i > 1:
        merged.append({"role": "system", "content": "\n\n".join(m["content"] for m in out[:i] if m["content"].strip())})
        out = merged + out[i:]
    return out


def turn_key(prefix: list[dict]) -> str:
    nonsys = [m for m in prefix if m["role"] != "system"]
    if not nonsys:
        tail = "∅"
    else:
        last = nonsys[-1]
        prev = nonsys[-2]["content"] if len(nonsys) > 1 else ""
        tail = last["role"] + "\x00" + norm(last["content"])[-600:] + "\x00" + norm(prev)[-300:]
    return hashlib.sha1(tail.encode()).hexdigest()[:20]


def _full(g) -> str:
    return (g.get("prefill") or "") + g["output"]


def _matches(content: str, output: str) -> bool:
    a, b = norm(content), norm(output)
    if a == b:
        return True
    # SillyTavern's "trim incomplete sentences" / stop-button can shorten a reply without the user editing it
    return len(a) >= 8 and b.startswith(a) and len(a) >= 0.5 * len(b)


@dataclass
class Analysis:
    kind: str                      # reply | continue | other | command
    messages: list                 # prompt messages for the model (cleaned, without prefill)
    prefill: str = ""
    turn_key: str = ""
    command: str | None = None
    command_arg: str = ""
    target: dict | None = None     # for commands: {"turn_key", "gen", "content"} of the previous reply
    events: list = field(default_factory=list)


class Learner:
    def __init__(self, store: Store):
        self.store = store

    def _resolve(self, prefix, content):
        """Find the logged generation behind an assistant message. Returns (key, gens, matched_gen)."""
        key = turn_key(prefix)
        gens = self.store.generations_for_key(key)
        matched = next((g for g in reversed(gens) if _matches(content, _full(g))), None)
        return key, gens, matched

    def _scan_history(self, msgs, upto, events):
        for i in range(upto):
            if msgs[i]["role"] != "assistant":
                continue
            content = msgs[i]["content"]
            key, gens, matched = self._resolve(msgs[:i], content)
            if not gens:
                continue  # greeting or a message the server never generated
            if matched:
                if matched["status"] in ("new", "swiped"):
                    self.store.set_status(matched["id"], "accepted")
                    for s in self.store.find_signals(key, kind="swipe", gen_id=matched["id"]):
                        self.store.update_signal(s["id"], active=0)
                for g in gens:
                    if g["id"] == matched["id"] or norm(_full(matched)).startswith(norm(_full(g))):
                        continue  # same text, or the part that was later continued
                    self._swipe_pair(key, g, chosen=_full(matched), events=events)
                continue
            base = max(gens, key=lambda g: difflib.SequenceMatcher(None, norm(_full(g)), norm(content)).ratio())
            existing = self.store.find_signals(key, kind="edit")
            if existing:
                if existing[-1]["chosen"] != content:
                    self.store.update_signal(existing[-1]["id"], chosen=content, consumed_in=None)
                    events.append(("edit-updated", key))
            else:
                self.store.add_signal("edit", key, base["id"], base["messages"], chosen=content, rejected=_full(base))
                self.store.set_status(base["id"], "edited")
                events.append(("edit", key))
            for g in gens:
                if g["id"] != base["id"]:
                    self._swipe_pair(key, g, chosen=content, events=events)

    def _swipe_pair(self, key, g, chosen, events):
        sig = self.store.find_signals(key, kind="swipe", gen_id=g["id"])
        if sig:
            if sig[0]["chosen"] != chosen:
                self.store.update_signal(sig[0]["id"], chosen=chosen, consumed_in=None)
                events.append(("swipe-pair", key))
        else:
            self.store.add_signal("swipe", key, g["id"], g["messages"], chosen=chosen, rejected=_full(g))
            self.store.set_status(g["id"], "swiped")
            events.append(("swipe-pair", key))

    def analyze(self, raw_messages) -> Analysis:
        raw = [{"role": m.get("role", "user"), "content": content_text(m.get("content"))} for m in raw_messages]
        msgs = clean_messages(raw)
        events: list = []
        nonsys_raw = [m for m in raw if m["role"] != "system"]
        last_raw = nonsys_raw[-1] if nonsys_raw else None

        if last_raw and last_raw["role"] == "user" and parse_command(last_raw["content"]):
            cmd, arg = parse_command(last_raw["content"])
            self._scan_history(msgs, len(msgs), events)
            target = None
            for i in range(len(msgs) - 1, -1, -1):
                if msgs[i]["role"] == "assistant":
                    key, gens, matched = self._resolve(msgs[:i], msgs[i]["content"])
                    gen = matched or (gens[-1] if gens else None)
                    target = {"turn_key": key, "gen": gen, "content": msgs[i]["content"],
                              "messages": gen["messages"] if gen else json.dumps(msgs[:i], ensure_ascii=False)}
                    break
            return Analysis("command", msgs, command=cmd, command_arg=arg, target=target, events=events)

        if msgs and msgs[-1]["role"] == "assistant":
            self._scan_history(msgs, len(msgs) - 1, events)
            prefix = msgs[:-1]
            return Analysis("continue", prefix, prefill=msgs[-1]["content"], turn_key=turn_key(prefix), events=events)

        self._scan_history(msgs, len(msgs), events)
        key = turn_key(msgs)
        nonsys = [m for m in msgs if m["role"] != "system"]
        if not nonsys or nonsys[-1]["role"] != "user":
            return Analysis("other", msgs, turn_key=key, events=events)
        for g in self.store.generations_for_key(key):
            if g["status"] in ("new", "accepted"):
                if not self.store.find_signals(key, kind="swipe", gen_id=g["id"]):
                    self.store.add_signal("swipe", key, g["id"], g["messages"], rejected=_full(g))
                    events.append(("swipe", key))
                self.store.set_status(g["id"], "swiped")
        return Analysis("reply", msgs, turn_key=key, events=events)

    def apply_command(self, a: Analysis) -> str:
        """Record an explicit feedback command. Returns a short human-readable result."""
        t = a.target
        if a.command in ("good", "bad", "fix"):
            if not t:
                return "没有找到上一条回复。"
            gid = t["gen"]["id"] if t["gen"] else None
            if a.command == "good":
                self.store.add_signal("good", t["turn_key"], gid, t["messages"], chosen=t["content"])
                return "已记录：这条回复很好 (good)。"
            if a.command == "bad":
                self.store.add_signal("bad", t["turn_key"], gid, t["messages"], rejected=t["content"])
                return "已记录：这条回复不好 (bad)。可以用 !fix <正确的回复> 给出示范。"
            if not a.command_arg:
                return "用法：!fix <你希望角色说的话>"
            self.store.add_signal("fix", t["turn_key"], gid, t["messages"], chosen=a.command_arg, rejected=t["content"])
            return "已记录修正 (fix)。下次睡眠时会学习。"
        return ""
