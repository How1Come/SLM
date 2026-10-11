"""Run the Cradle trainer web UI.

    python -m slm.ui [--port 8765] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse

from .app import create_app


def main() -> None:
    ap = argparse.ArgumentParser(description="Cradle trainer web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    app = create_app()
    print(f"[slm.ui] http://{args.host}:{args.port}/  (CPU training, no login)")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
