from __future__ import annotations

import difflib
import hashlib
import json
import re
import threading
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
    match = _CMD_RE.match(text or "")
    if not match:
        return None
    return COMMANDS[match.group(1).lower()], match.group(2).strip()


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type", "text") == "text")
    return "" if content is None else str(content)


def clean_messages(raw) -> list[dict]:
    out = []
    for message in raw:
        role = message.get("role", "user")
        if role not in ("system", "user", "assistant"):
            role = "user" if role != "tool" else "system"
        text = content_text(message.get("content"))
        if role == "user" and parse_command(text):
            continue
        if role == "assistant" and text.lstrip().startswith(ACK_PREFIX):
            continue
        out.append({"role": role, "content": text})
    index = 0
    while index < len(out) and out[index]["role"] == "system":
        index += 1
    if index > 1:
        system = "\n\n".join(message["content"] for message in out[:index] if message["content"].strip())
        out = [{"role": "system", "content": system}, *out[index:]]
    return out


def conversation_scope(messages: list[dict], session_id: str | None = None) -> str:
    if session_id is not None and (not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 256):
        raise ValueError("Session ID must contain 1 to 256 characters")
    if session_id is not None:
        identity = "session\x00" + session_id.strip()
    else:
        system = "\n".join(message["content"] for message in messages if message["role"] == "system")
        identity = "system\x00" + norm(system)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def turn_key(prefix: list[dict], scope_id: str | None = None) -> str:
    scope = scope_id or conversation_scope(prefix)
    nonsys = [message for message in prefix if message["role"] != "system"]
    if not nonsys:
        tail = "∅"
    else:
        last = nonsys[-1]
        previous = nonsys[-2]["content"] if len(nonsys) > 1 else ""
        tail = last["role"] + "\x00" + norm(last["content"]) + "\x00" + norm(previous)
    return scope + ":" + hashlib.sha256(tail.encode("utf-8")).hexdigest()[:24]


def _full(generation) -> str:
    return (generation.get("prefill") or "") + generation["output"]


def _matches(content: str, output: str) -> bool:
    first, second = norm(content), norm(output)
    if first == second:
        return True
    return len(first) >= 8 and second.startswith(first) and len(first) >= 0.5 * len(second)


@dataclass
class Analysis:
    kind: str
    messages: list
    prefill: str = ""
    turn_key: str = ""
    command: str | None = None
    command_arg: str = ""
    target: dict | None = None
    events: list = field(default_factory=list)
    scope_id: str = ""
    session_id: str | None = None


class Learner:
    def __init__(self, store: Store):
        self.store = store
        self._lock = threading.RLock()

    def _resolve(self, prefix, content, scope_id):
        key = turn_key(prefix, scope_id)
        generations = self.store.generations_for_key(key)
        matched = next((generation for generation in reversed(generations) if _matches(content, _full(generation))), None)
        return key, generations, matched

    def _scan_history(self, messages, upto, events, scope_id):
        for index in range(upto):
            if messages[index]["role"] != "assistant":
                continue
            content = messages[index]["content"]
            key, generations, matched = self._resolve(messages[:index], content, scope_id)
            if not generations:
                continue
            if matched:
                self.store.set_status(matched["id"], "accepted")
                for signal in self.store.find_signals(key, kind="edit"):
                    self.store.update_signal(signal["id"], active=0)
                for signal in self.store.find_signals(key, kind="swipe", gen_id=matched["id"]):
                    self.store.update_signal(signal["id"], active=0)
                for generation in generations:
                    if generation["id"] == matched["id"] or norm(_full(matched)).startswith(norm(_full(generation))):
                        continue
                    self._swipe_pair(key, generation, chosen=_full(matched), events=events)
                continue
            base = max(generations, key=lambda generation: difflib.SequenceMatcher(None, norm(_full(generation)), norm(content)).ratio())
            existing = self.store.find_signals(key, kind="edit", active_only=False)
            if existing:
                signal = existing[-1]
                updates = {"chosen": content}
                if signal["review_status"] == "accepted":
                    updates["active"] = 1
                if signal["chosen"] != content or signal["active"] != updates.get("active", signal["active"]):
                    self.store.update_signal(signal["id"], **updates)
                    events.append(("edit-updated", key))
            else:
                self.store.add_signal("edit", key, base["id"], base["messages"], chosen=content, rejected=_full(base))
                events.append(("edit", key))
            self.store.set_status(base["id"], "edited")
            for generation in generations:
                if generation["id"] != base["id"]:
                    self._swipe_pair(key, generation, chosen=content, events=events)

    def _swipe_pair(self, key, generation, chosen, events):
        existing = self.store.find_signals(key, kind="swipe", gen_id=generation["id"], active_only=False)
        if existing:
            signal = existing[0]
            updates = {"chosen": chosen}
            if signal["review_status"] == "accepted":
                updates["active"] = int(norm(chosen) != norm(_full(generation)))
            if signal["chosen"] != chosen or signal["active"] != updates.get("active", signal["active"]):
                self.store.update_signal(signal["id"], **updates)
                events.append(("swipe-pair", key))
        else:
            self.store.add_signal("swipe", key, generation["id"], generation["messages"], chosen=chosen, rejected=_full(generation))
            events.append(("swipe-pair", key))
        self.store.set_status(generation["id"], "swiped")

    def analyze(self, raw_messages, session_id: str | None = None) -> Analysis:
        with self._lock:
            return self._analyze(raw_messages, session_id)

    def _analyze(self, raw_messages, session_id) -> Analysis:
        raw = [{"role": message.get("role", "user"), "content": content_text(message.get("content"))} for message in raw_messages]
        messages = clean_messages(raw)
        scope_id = conversation_scope(messages, session_id)
        common = {"scope_id": scope_id, "session_id": session_id}
        events: list = []
        nonsys_raw = [message for message in raw if message["role"] != "system"]
        last_raw = nonsys_raw[-1] if nonsys_raw else None
        if last_raw and last_raw["role"] == "user" and parse_command(last_raw["content"]):
            command, argument = parse_command(last_raw["content"])
            self._scan_history(messages, len(messages), events, scope_id)
            target = None
            for index in range(len(messages) - 1, -1, -1):
                if messages[index]["role"] == "assistant":
                    key, generations, matched = self._resolve(messages[:index], messages[index]["content"], scope_id)
                    generation = matched or (generations[-1] if generations else None)
                    target = {"turn_key": key, "gen": generation, "content": messages[index]["content"],
                              "messages": generation["messages"] if generation else json.dumps(messages[:index], ensure_ascii=False)}
                    break
            return Analysis("command", messages, command=command, command_arg=argument, target=target, events=events, **common)
        if messages and messages[-1]["role"] == "assistant":
            self._scan_history(messages, len(messages) - 1, events, scope_id)
            prefix = messages[:-1]
            return Analysis("continue", prefix, prefill=messages[-1]["content"], turn_key=turn_key(prefix, scope_id), events=events, **common)
        self._scan_history(messages, len(messages), events, scope_id)
        key = turn_key(messages, scope_id)
        nonsys = [message for message in messages if message["role"] != "system"]
        if not nonsys or nonsys[-1]["role"] != "user":
            return Analysis("other", messages, turn_key=key, events=events, **common)
        generations = self.store.generations_for_key(key)
        if generations:
            for signal in self.store.find_signals(key, kind="swipe", active_only=False):
                if signal["chosen"] is not None:
                    self.store.update_signal(signal["id"], chosen=None)
        for generation in generations:
            if generation["status"] in ("new", "accepted"):
                if not self.store.find_signals(key, kind="swipe", gen_id=generation["id"], active_only=False):
                    self.store.add_signal("swipe", key, generation["id"], generation["messages"], rejected=_full(generation))
                    events.append(("swipe", key))
                self.store.set_status(generation["id"], "swiped")
        return Analysis("reply", messages, turn_key=key, events=events, **common)

    def apply_command(self, analysis: Analysis) -> str:
        with self._lock:
            target = analysis.target
            if analysis.command not in ("good", "bad", "fix"):
                return ""
            if not target:
                return "没有找到上一条回复。"
            generation_id = target["gen"]["id"] if target["gen"] else None
            if analysis.command == "fix" and not analysis.command_arg:
                return "用法：!fix <你希望角色说的话>"
            chosen = analysis.command_arg if analysis.command == "fix" else target["content"] if analysis.command == "good" else None
            rejected = target["content"] if analysis.command in ("bad", "fix") else None
            for opposite in ({"good": ("bad",), "bad": ("good",), "fix": ("good", "bad")}[analysis.command]):
                for signal in self.store.find_signals(target["turn_key"], kind=opposite, gen_id=generation_id):
                    self.store.update_signal(signal["id"], active=0)
            existing = self.store.find_signals(target["turn_key"], kind=analysis.command, gen_id=generation_id, active_only=False)
            if existing:
                self.store.update_signal(existing[-1]["id"], chosen=chosen, rejected=rejected, active=1, review_status="accepted")
            else:
                self.store.add_signal(analysis.command, target["turn_key"], generation_id, target["messages"],
                                      chosen=chosen, rejected=rejected, session_id=analysis.session_id, scope_id=analysis.scope_id)
            responses = {
                "good": "已记录：这条回复很好 (good)。",
                "bad": "已记录：这条回复不好 (bad)。可以用 !fix <正确的回复> 给出示范。",
                "fix": "已记录修正 (fix)。下次睡眠时会学习。",
            }
            return responses[analysis.command]
