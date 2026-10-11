"""Flask app for the Cradle trainer UI."""

from __future__ import annotations

import json
import queue
import time
from dataclasses import asdict

from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from .presets import DEFAULT_BUDGET, DEFAULT_SEED, PRESETS
from .runner import JobManager, ROOT

manager = JobManager()


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index():
        presets = [
            {
                "id": p.id,
                "title": p.title,
                "blurb": p.blurb,
            }
            for p in PRESETS.values()
        ]
        return render_template(
            "index.html",
            presets=presets,
            default_budget=DEFAULT_BUDGET,
            default_seed=DEFAULT_SEED,
        )

    @app.get("/api/status")
    def status():
        snap = manager.current()
        return jsonify(asdict(snap) if snap else {"state": "idle"})

    @app.post("/api/train")
    def train():
        body = request.get_json(force=True, silent=True) or {}
        preset_id = body.get("preset")
        if not preset_id:
            return jsonify({"error": "preset required"}), 400
        budget = body.get("budget")
        seed = body.get("seed")
        try:
            if budget is not None:
                budget = float(budget)
            if seed is not None:
                seed = int(seed)
            snap = manager.start(preset_id, budget=budget, seed=seed)
            return jsonify(asdict(snap))
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 409
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.post("/api/stop")
    def stop():
        ok = manager.stop()
        return jsonify({"stopped": ok})

    @app.get("/api/events")
    def events():
        q: queue.Queue = queue.Queue(maxsize=512)

        def on_event(ev: dict) -> None:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass

        manager.subscribe(on_event)
        snap = manager.current()
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
                manager.unsubscribe(on_event)

        return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/complete")
    def complete_text():
        body = request.get_json(force=True, silent=True) or {}
        checkpoint = body.get("checkpoint")
        prompt = (body.get("prompt") or "").strip()
        if not checkpoint:
            snap = manager.current()
            checkpoint = snap.checkpoint if snap else None
        if not checkpoint:
            return jsonify({"error": "no checkpoint; train a model first"}), 400
        if not prompt:
            return jsonify({"error": "prompt required"}), 400
        path = Path(checkpoint)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            return jsonify({"error": f"checkpoint not found: {checkpoint}"}), 404
        try:
            import torch

            from slm.chat import complete as chat_complete
            from slm.data import load_tokenizer
            from slm.evaluate import load_model

            max_new = int(body.get("max_new", 80))
            temperature = float(body.get("temperature", 0.8))
            top_k = int(body.get("top_k", 40))
            model = load_model(str(path))
            tok = load_tokenizer()
            seed = body.get("seed")
            if seed is not None:
                torch.manual_seed(int(seed))
            continuation = chat_complete(model, tok, prompt, max_new=max_new, temperature=temperature, top_k=top_k, n=1)[0]
            return jsonify({"prompt": prompt.lower(), "continuation": continuation, "full": f"{prompt.lower()}{continuation}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app
