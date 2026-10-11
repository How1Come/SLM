"""Flask app for the role-play trainer UI."""

from __future__ import annotations

import json
import queue
from dataclasses import asdict

from flask import Flask, Response, jsonify, render_template, request

from .config import UiConfig
from .runner import JobManager, RpServerProcess, collect_overview


def create_app(cfg: UiConfig) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    jobs = JobManager(cfg)
    server = RpServerProcess(cfg)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            cradle_ui_url=cfg.cradle_ui_url,
            ui_port=cfg.ui_port,
            rp_port=cfg.rp_port,
        )

    @app.get("/api/overview")
    def overview():
        data = collect_overview(cfg)
        data["server"] = server.status()
        snap = jobs.current()
        data["job"] = asdict(snap) if snap else None
        return jsonify(data)

    @app.post("/api/narration/train")
    def narration_train():
        try:
            return jsonify(asdict(jobs.start_narration_train()))
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 409

    @app.post("/api/narration/eval")
    def narration_eval():
        try:
            return jsonify(asdict(jobs.start_narration_eval()))
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 409

    @app.post("/api/sleep")
    def sleep():
        ov = collect_overview(cfg)
        if not ov["sleep"]["ready"]:
            return jsonify({"error": ov["sleep"]["message"]}), 400
        try:
            return jsonify(asdict(jobs.start_sleep()))
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 409

    @app.post("/api/job/stop")
    def job_stop():
        return jsonify({"stopped": jobs.stop()})

    @app.get("/api/job")
    def job_status():
        snap = jobs.current()
        return jsonify(asdict(snap) if snap else {"state": "idle"})

    @app.post("/api/server/start")
    def server_start():
        return jsonify(server.start())

    @app.post("/api/server/stop")
    def server_stop():
        return jsonify({"stopped": server.stop()})

    @app.get("/api/events")
    def events():
        q: queue.Queue = queue.Queue(maxsize=512)

        def on_event(ev: dict) -> None:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass

        jobs.subscribe(on_event)
        snap = jobs.current()
        if snap:
            q.put({"type": "snapshot", "data": asdict(snap)})

        def stream():
            try:
                while True:
                    try:
                        ev = q.get(timeout=15)
                        yield f"data: {json.dumps(ev)}\n\n"
                    except queue.Empty:
                        yield f"data: {json.dumps({'type': 'ping'})}\n\n"
            finally:
                jobs.unsubscribe(on_event)

        return Response(
            stream(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
