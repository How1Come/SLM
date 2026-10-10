"""Build chat requests the way SillyTavern's default Chat Completion preset does, for scripted demos and evals.

Prefer `system_from_db`, which reuses the exact system prompt SillyTavern sent for the character (as logged by
the server), so scripted traffic and held-out evals see the same prompt as real SillyTavern chats.
"""
import json
import sqlite3


def macros(text, char, user):
    return (text or "").replace("{{char}}", char).replace("{{user}}", user).replace("<USER>", user).replace("<BOT>", char)


def default_system(card, user):
    d = card["data"]
    c = d["name"]
    parts = [f"Write {c}'s next reply in a fictional chat between {c} and {user}.", macros(d["description"], c, user)]
    if d.get("personality"):
        parts.append(f"{c}'s personality: {macros(d['personality'], c, user)}")
    if d.get("scenario"):
        parts.append(f"Scenario: {macros(d['scenario'], c, user)}")
    parts.append("[Start a new Chat]")
    return "\n\n".join(parts)


def system_from_db(db_path, char_name):
    """Most recent merged system prompt that mentions the character and came from SillyTavern."""
    db = sqlite3.connect(db_path)
    for (msgs,) in db.execute("SELECT messages FROM generations ORDER BY ts DESC"):
        m = json.loads(msgs)
        if m and m[0]["role"] == "system" and char_name in m[0]["content"] and "[Start a new Chat]" in m[0]["content"]:
            return m[0]["content"]
    return None


def greeting(card, user):
    d = card["data"]
    return macros(d["first_mes"], d["name"], user)


def build(system, greet, turns):
    """turns: list of (user, assistant) with assistant=None for the turn to be generated."""
    msgs = [{"role": "system", "content": system}, {"role": "assistant", "content": greet}]
    for u, a in turns:
        msgs.append({"role": "user", "content": u})
        if a is not None:
            msgs.append({"role": "assistant", "content": a})
    return msgs
