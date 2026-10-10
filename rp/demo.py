"""Scripted "muzi edits the reply" sessions, sent to the server exactly like SillyTavern sends them.

    python -m rp.demo teach --chats 0 1     # run the scripted chats; every model reply is replaced by muzi's edit

Each turn: SillyTavern-style streaming request -> model reply (logged by the server) -> the reply is replaced by
the edited in-character text in the history -> the next request carries the edit, and the server records it as
a correction (edited = preferred, original = rejected). Nothing here talks to the learner directly.
"""
import argparse
import json
import os

import httpx

from . import DEFAULT_DATA_DIR, DEFAULT_PORT
from .stprompt import build, default_system, greeting, system_from_db

HERE = os.path.dirname(__file__)


def chat(client, messages, max_tokens=250, temperature=0.7, seed=None):
    body = {"model": "qwen2.5-0.5b-rp", "messages": messages, "stream": True, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.9}
    if seed is not None:
        body["seed"] = seed
    out = []
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                d = json.loads(line[6:])["choices"][0]["delta"]
                out.append(d.get("content") or "")
    return "".join(out)


def load(card_path, demo_path, data_dir, user):
    with open(card_path, encoding="utf-8") as f:
        card = json.load(f)
    with open(demo_path, encoding="utf-8") as f:
        demo = json.load(f)
    system = system_from_db(os.path.join(data_dir, "rp.db"), card["data"]["name"])
    src = "captured from SillyTavern"
    if not system:
        system, src = default_system(card, user), "default SillyTavern layout (no captured prompt yet)"
    return card, demo, system, src


def teach(a):
    card, demo, system, src = load(a.card, a.demo, a.data_dir, a.user)
    print(f"system prompt: {src}")
    greet = greeting(card, a.user)
    c = httpx.Client(base_url=a.server, timeout=600)
    for ci in a.chats:
        turns = []
        print(f"\n=== chat {ci} ===")
        for step in demo["train_chats"][ci]:
            reply = chat(c, build(system, greet, turns + [(step["user"], None)]), seed=a.seed)
            print(f"\nmuzi: {step['user']}\nmodel: {reply}\nedited: {step['edit']}")
            turns.append((step["user"], step["edit"]))
        reply = chat(c, build(system, greet, turns + [(a.closing, None)]), seed=a.seed)
        print(f"\nmuzi: {a.closing}\nmodel (kept): {reply}")
    print(json.dumps(c.get("/rp/stats").json()["signals"], ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["teach"])
    ap.add_argument("--server", default=f"http://127.0.0.1:{DEFAULT_PORT}")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--card", default=os.path.join(HERE, "characters", "su_xiaotang.json"))
    ap.add_argument("--demo", default=os.path.join(HERE, "characters", "su_xiaotang_demo.json"))
    ap.add_argument("--user", default="muzi")
    ap.add_argument("--chats", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--closing", default="嗯。")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    teach(a)


if __name__ == "__main__":
    main()
