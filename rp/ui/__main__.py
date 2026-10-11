"""Run the role-play trainer web UI.

    python -m rp.ui [--port 8847] [--rp-port 8627] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse

from .app import create_app
from .config import DEFAULT_RP_PORT, DEFAULT_UI_PORT, UiConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Role-play trainer web UI (CPU)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_UI_PORT, help="UI port (not the rp.server port)")
    ap.add_argument("--rp-port", type=int, default=DEFAULT_RP_PORT, help="port for python -m rp.server when started from UI")
    ap.add_argument("--rp-host", default="127.0.0.1")
    ap.add_argument("--data-dir", default="rp_data")
    ap.add_argument("--model", default=None)
    ap.add_argument("--model-id", default="qwen2.5-0.5b-rp")
    ap.add_argument("--cradle-ui", default="http://127.0.0.1:8765/", help="link to python -m slm.ui")
    args = ap.parse_args()

    from .. import DEFAULT_MODEL

    cfg = UiConfig(
        host=args.host,
        ui_port=args.port,
        rp_host=args.rp_host,
        rp_port=args.rp_port,
        data_dir=args.data_dir,
        model=args.model or DEFAULT_MODEL,
        model_id=args.model_id,
        cradle_ui_url=args.cradle_ui,
    )
    app = create_app(cfg)
    print(f"[rp.ui] http://{args.host}:{args.port}/  (CPU, narration + sleep + server)")
    print(f"[rp.ui] SillyTavern endpoint when server is running: {cfg.st_endpoint}")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
