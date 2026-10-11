"""CLI for a running rp server.

    python -m rp.admin stats
    python -m rp.admin sleep [--wait]          # train + hot-reload now
    python -m rp.admin adapters
    python -m rp.admin rollback [VERSION]      # default: the version that was active before the current one
    python -m rp.admin good | bad              # rate the last reply
    python -m rp.admin fix "更符合角色的回复"   # correct the last reply
"""
import argparse
import json
import os

import httpx

from . import DEFAULT_PORT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.environ.get("RP_SERVER", f"http://127.0.0.1:{DEFAULT_PORT}"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    s = sub.add_parser("sleep")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--epochs", type=int)
    s.add_argument("--lr", type=float)
    sub.add_parser("adapters")
    r = sub.add_parser("rollback")
    r.add_argument("version", nargs="?")
    sub.add_parser("good")
    sub.add_parser("bad")
    f = sub.add_parser("fix")
    f.add_argument("text")
    a = ap.parse_args()
    token = os.environ.get("RP_AUTH_TOKEN", "")
    headers = {"Authorization":"Bearer " + token} if token else {}
    c = httpx.Client(base_url=a.server, timeout=3600, headers=headers, trust_env=False)

    if a.cmd == "stats":
        out = c.get("/rp/stats").json()
    elif a.cmd == "sleep":
        extra = {k: v for k, v in {"epochs": a.epochs, "lr": a.lr}.items() if v is not None}
        out = c.post("/rp/sleep", json={"wait": a.wait, "args": extra}).json()
    elif a.cmd == "adapters":
        out = c.get("/rp/adapters").json()
    elif a.cmd == "rollback":
        v = a.version
        if not v:
            info = c.get("/rp/adapters").json()
            cur = next((x for x in info["versions"] if x["version"] == info["active"]), None)
            v = (cur or {}).get("previous", "base")
        out = c.post("/rp/activate", json={"version": v}).json()
    elif a.cmd in ("good", "bad"):
        out = c.post("/rp/feedback", json={"rating": a.cmd}).json()
    else:
        out = c.post("/rp/feedback", json={"correction": a.text}).json()
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
