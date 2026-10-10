"""Point a local SillyTavern install at the rp server (Chat Completion -> Custom (OpenAI-compatible)).

    python -m rp.st_setup --st ~/SillyTavern [--server http://127.0.0.1:8619/v1] [--user muzi]

Edits data/default-user/settings.json (close SillyTavern browser tabs first, the UI saves its own copy) and
installs the sample character card. Everything here can also be done by hand in the SillyTavern UI.
"""
import argparse
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", default="~/SillyTavern")
    ap.add_argument("--server", default="http://127.0.0.1:8619/v1")
    ap.add_argument("--model-id", default="qwen2.5-0.5b-rp")
    ap.add_argument("--user", default="muzi")
    ap.add_argument("--card", default=os.path.join(os.path.dirname(__file__), "characters", "su_xiaotang.json"))
    a = ap.parse_args()
    root = os.path.expanduser(a.st)
    path = os.path.join(root, "data", "default-user", "settings.json")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: start SillyTavern once (node server.js) so it creates its data folder")
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    s["firstRun"] = False
    s["username"] = a.user
    s["main_api"] = "openai"
    o = s.setdefault("oai_settings", {})
    o.update({
        "chat_completion_source": "custom",
        "custom_url": a.server,
        "custom_model": a.model_id,
        "stream_openai": True,
        "openai_max_context": 4096,
        "openai_max_tokens": 250,
        "temp_openai": 0.7,
        "top_p_openai": 0.9,
        "freq_pen_openai": 0,
        "pres_pen_openai": 0,
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=4)
    print("patched", path)
    subprocess.run([sys.executable, "-m", "rp.make_card", a.card, "--install",
                    os.path.join(root, "data", "default-user", "characters")], check=True)


if __name__ == "__main__":
    main()
