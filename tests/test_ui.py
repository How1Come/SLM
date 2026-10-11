import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import psutil

from rp.engine import read_current
from rp.store import Store
from rp.ui.app import create_app
from rp.ui.config import UiConfig
from rp.ui.journal import JobJournal
from rp.ui.runner import JobManager, JobSnapshot, RpServerProcess, _Job


class StubServer:
    def __init__(self, directory):
        self.ready = False
        self.log_path = Path(directory) / "server.log"
        self.started = 0

    def status(self):
        return {"ready": self.ready, "running": self.ready, "starting": False, "external": False, "error": None}

    def start(self):
        self.started += 1
        self.ready = True
        return self.status()

    def stop(self):
        stopped = self.ready
        self.ready = False
        return stopped


class StubJobs:
    def __init__(self, directory):
        self.journal = JobJournal(directory)
        self.snapshot = None
        self.listeners = set()

    def current(self):
        return self.snapshot

    def history(self):
        return self.journal.history()

    def start_narration_train(self, values):
        self.snapshot = JobSnapshot("abcdef1234", "narration_train", "queued", 0, [], None, None, time.time(), None, config=values)
        return self.snapshot

    def start_narration_eval(self):
        return self.start_narration_train({})

    def start_sleep(self, values):
        return self.start_narration_train(values)

    def stop(self):
        return False

    def resume(self, identifier):
        raise ValueError("No complete checkpoint is available to resume.")

    def subscribe(self, listener):
        self.listeners.add(listener)

    def unsubscribe(self, listener):
        self.listeners.discard(listener)


class UiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        environment = patch.dict(os.environ, {"RP_AUTH_TOKEN": ""})
        environment.start()
        self.addCleanup(environment.stop)
        self.config = UiConfig(data_dir=self.temporary.name, model="test-model")
        self.jobs = StubJobs(self.temporary.name)
        self.server = StubServer(self.temporary.name)
        self.app = create_app(self.config, self.jobs, self.server)
        self.client = self.app.test_client()
        self.messages = [{"role": "system", "content": "Be kind"}, {"role": "user", "content": "hello"}]

    def response(self, code=200, payload=None):
        return httpx.Response(code, json=payload or {}, request=httpx.Request("GET", "http://localhost"))

    def test_ui_renders_real_navigation_without_secret(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.content_type)
        self.assertTrue((Path(self.temporary.name) / ".ui-secret").is_file())
        self.assertNotIn((Path(self.temporary.name) / ".ui-secret").read_text(), response.text)

    def test_mutating_requests_require_json_and_same_origin(self):
        response = self.client.patch("/api/settings", data='{"enabled":false}', content_type="text/plain")
        self.assertEqual(response.status_code, 415)
        response = self.client.patch("/api/settings", json={"enabled": False}, headers={"Origin": "https://evil.example"})
        self.assertEqual(response.status_code, 403)
        response = self.client.patch("/api/settings", json={"enabled": False}, headers={"Origin": "http://localhost"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json["enabled"])

    def test_settings_validate_and_survive_application_restart(self):
        response = self.client.patch("/api/settings", json={"mode": "background", "threads": 1, "min_signals": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.patch("/api/settings", json={"mode": "random"}).status_code, 422)
        self.assertEqual(self.client.patch("/api/settings", json={"threads": 0}).status_code, 422)
        reloaded = create_app(self.config, self.jobs, self.server).test_client()
        self.assertEqual(reloaded.get("/api/settings").json["mode"], "background")
        self.assertEqual(reloaded.get("/api/settings").json["min_signals"], 2)

    def test_signals_can_be_reviewed_paginated_and_exported(self):
        with Store(self.temporary.name) as store:
            signal_id = store.add_signal("fix", "turn", None, self.messages, chosen="better response", rejected="bad response")
        response = self.client.get("/api/signals?page=1&page_size=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["total"], 1)
        self.assertEqual(response.json["items"][0]["id"], signal_id)
        self.assertEqual(self.client.get("/api/signals?page=0").status_code, 422)
        self.assertEqual(self.client.get("/api/signals?page_size=101").status_code, 422)
        self.assertEqual(self.client.patch(f"/api/signals/{signal_id}", json={"active": False}).status_code, 200)
        self.assertEqual(self.client.patch(f"/api/signals/{signal_id}", json={"active": True}).status_code, 200)
        exported = self.client.get("/api/signals/export")
        self.assertEqual(exported.status_code, 200)
        self.assertIn("attachment", exported.headers["Content-Disposition"])
        record = json.loads(exported.text.strip())
        self.assertEqual(record["preference"]["chosen"], "better response")
        self.assertEqual(record["messages"][-1], {"role": "assistant", "content": "better response"})

    def test_no_preferred_feedback_disables_chat_training(self):
        response = self.client.post("/api/sleep", json={})
        self.assertEqual(response.status_code, 409)
        self.assertIn("preferred reply", response.json["error"])
        self.assertIsNone(self.jobs.snapshot)

    def test_offline_base_selection_persists_and_missing_adapter_returns_404(self):
        response = self.client.post("/api/adapters/activate", json={"version": "base"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(read_current(self.temporary.name), "base")
        self.assertEqual(self.client.post("/api/adapters/activate", json={"version": "v999"}).status_code, 404)
        self.assertEqual(self.client.post("/api/adapters/activate", json={"version": "../../secrets"}).status_code, 422)

    def test_adapter_export_includes_frozen_foundation_and_excludes_symlink(self):
        output = Path(self.temporary.name) / "adapters" / "v1"
        foundation = output / "foundation" / "narration"
        foundation.mkdir(parents=True)
        (output / "adapter_config.json").write_text("{}")
        (output / "adapter_model.safetensors").write_bytes(b"weights")
        (output / "meta.json").write_text(json.dumps({"foundation": {"path": "foundation/narration"}}))
        (foundation / "adapter_config.json").write_text("{}")
        secret = Path(self.temporary.name) / "secret"
        secret.write_text("DO NOT EXPORT")
        (output / "leaked-secret").symlink_to(secret)
        response = self.client.get("/api/adapters/v1/export")
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
            self.assertIn("v1/foundation/narration/adapter_config.json", archive.namelist())
            self.assertIn("v1/meta.json", archive.namelist())
            self.assertNotIn("v1/leaked-secret", archive.namelist())

    def test_proxy_hides_connection_and_timeout_details(self):
        with patch("rp.ui.app.httpx.request", side_effect=httpx.ConnectError("sensitive host details")):
            response = self.client.post("/api/chat", json={"messages": self.messages})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("sensitive", response.text)
        with patch("rp.ui.app.httpx.request", side_effect=httpx.ReadTimeout("sensitive timeout")):
            response = self.client.post("/api/chat", json={"messages": self.messages})
        self.assertEqual(response.status_code, 504)
        self.assertNotIn("sensitive", response.text)

    def test_proxy_passes_feedback_generation_id_and_model_settings(self):
        with patch("rp.ui.app.httpx.request", return_value=self.response(payload={"ok": True})) as request:
            response = self.client.post("/api/chat", json={"messages": self.messages, "stream": True})
            self.assertEqual(response.status_code, 200)
            parameters = request.call_args.kwargs["json"]
            self.assertFalse(parameters["stream"])
            self.assertEqual(parameters["model"], self.config.model_id)
            self.assertEqual(self.client.post("/api/feedback", json={"rating": "good"}).status_code, 422)
            self.assertEqual(self.client.post("/api/feedback", json={"gen_id": "generation", "rating": "good"}).status_code, 200)

    def test_job_logs_can_be_downloaded(self):
        snapshot = JobSnapshot("abcdef1234", "sleep", "paused", .5, [], None, None, 1, 2)
        self.jobs.journal.save(asdict(snapshot))
        self.jobs.journal.log_path(snapshot.id).write_text("METRIC {}\nRESULT {}\n")
        response = self.client.get(f"/api/jobs/{snapshot.id}/logs")
        self.assertEqual(response.status_code, 200)
        self.assertIn("METRIC", response.text)
        self.assertIn("attachment", response.headers["Content-Disposition"])
        response.close()
        self.assertEqual(self.client.get("/api/jobs/missing/logs").status_code, 404)

    def test_authentication_protects_api_and_openai_even_after_browser_login(self):
        self.config.auth_token = "a" * 32
        app = create_app(self.config, self.jobs, self.server)
        client = app.test_client()
        self.assertEqual(client.get("/api/settings").status_code, 401)

        self.assertEqual(client.post("/api/auth/login", json={"token": "wrong"}).status_code, 401)
        self.assertEqual(client.post("/api/auth/login", json={"token": self.config.auth_token}).status_code, 200)
        self.assertEqual(client.get("/api/settings").status_code, 200)
        self.assertEqual(client.get("/v1/models").status_code, 401)
        with patch("rp.ui.app.httpx.request", return_value=self.response(payload={"object": "list"})):
            response = client.get("/v1/models", headers={"Authorization": "Bearer " + self.config.auth_token})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.post("/api/auth/logout", json={}).status_code, 200)
        self.assertEqual(client.get("/api/settings").status_code, 401)

    def test_unicode_login_token_is_rejected_without_server_error(self):
        self.config.auth_token = "a" * 32
        client = create_app(self.config, self.jobs, self.server).test_client()
        response = client.post("/api/auth/login", json={"token": "你好"})
        self.assertEqual(response.status_code, 401)

    def test_public_ui_binding_requires_access_token(self):
        with self.assertRaisesRegex(ValueError, "RP_AUTH_TOKEN"):
            UiConfig(host="0.0.0.0", data_dir=self.temporary.name, auth_token="short")


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = UiConfig(data_dir=self.temporary.name, model="test-model", auth_token="")

    def managed_worker(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def cleanup():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)

        self.addCleanup(cleanup)
        record = {"pid": process.pid, "process_created": psutil.Process(process.pid).create_time(),
                  "model": self.config.model, "endpoint": self.config.rp_base_url}
        path = Path(self.temporary.name) / "server-process.json"
        path.write_text(json.dumps(record))
        return process, record, path

    def test_restarted_ui_adopts_live_model_worker_and_reaps_it_on_stop(self):
        process, record, path = self.managed_worker()
        server = RpServerProcess(self.config)
        with patch("rp.ui.runner.httpx.get", side_effect=httpx.ConnectError("offline")), patch("rp.ui.runner.subprocess.Popen") as launch:
            status = server.start()
            launch.assert_not_called()
        self.assertTrue(status["managed"])
        self.assertTrue(status["running"])
        self.assertEqual(status["pid"], record["pid"])
        self.assertTrue(server.stop())
        self.assertIsNotNone(process.poll())
        self.assertFalse(psutil.pid_exists(process.pid))
        self.assertFalse(path.exists())

    def test_restart_refuses_worker_with_different_endpoint_or_model(self):
        process, record, path = self.managed_worker()
        for field, value in (("endpoint", "http://127.0.0.1:9999"), ("model", "another-model")):
            with self.subTest(field=field):
                path.write_text(json.dumps({**record, field: value}))
                server = RpServerProcess(self.config)
                self.assertIsNone(server.managed_process())
                self.assertFalse(server.stop())
                self.assertIsNone(process.poll())

    def test_restart_refuses_worker_when_pid_creation_time_does_not_match(self):
        process, record, path = self.managed_worker()
        path.write_text(json.dumps({**record, "process_created": record["process_created"] - 10}))
        server = RpServerProcess(self.config)
        self.assertIsNone(server.managed_process())
        self.assertFalse(server.stop())
        self.assertIsNone(process.poll())

    def test_restart_refuses_corrupt_process_record_without_error(self):
        process, record, path = self.managed_worker()
        for corrupt in ({**record, "process_created": "invalid"}, [record], {**record, "pid": "invalid"}):
            with self.subTest(record=corrupt):
                path.write_text(json.dumps(corrupt))
                server = RpServerProcess(self.config)
                self.assertIsNone(server.managed_process())
                self.assertFalse(server.stop())
                self.assertIsNone(process.poll())

    def test_training_presets_are_distinct_and_inputs_bounded(self):
        manager = JobManager(self.config)
        choices = [manager.options({"preset": name}) for name in ("quick", "balanced", "thorough")]
        self.assertEqual([choice["epochs"] for choice in choices], [1, 2, 4])
        self.assertEqual([choice["max_minutes"] for choice in choices], [5, 15, 45])
        for body in ({"preset": "other"}, {"epochs": 0}, {"threads": True}, {"epochs": 1.5}, {"lr": 99}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                manager.options(body)

    def test_restart_marks_unfinished_job_interrupted_and_can_resume_complete_checkpoint(self):
        checkpoint = Path(self.temporary.name) / "checkpoints" / "step-one"
        checkpoint.mkdir(parents=True)
        snapshot = JobSnapshot("abcdef1234", "sleep", "running", .4, [], None, None, 1, None,
                               checkpoint=str(checkpoint), config={"epochs": 2})
        journal = JobJournal(self.temporary.name)
        journal.save({**asdict(snapshot), "pid": None, "process_created": None})
        manager = JobManager(self.config)
        self.assertEqual(manager.current().state, "interrupted")
        with patch.object(manager, "_start", return_value=snapshot) as start:
            manager.resume(snapshot.id)
        self.assertEqual(start.call_args.args[1]["resume"], str(checkpoint.resolve()))
        self.assertEqual(start.call_args.args[1]["epochs"], 2)

    def test_real_metric_and_checkpoint_records_preserve_progress(self):
        manager = JobManager(self.config)
        job = _Job("abcdef1234", "sleep", state="running")
        manager._log(job, 'METRIC {"step": 2, "total_steps": 4, "progress": 0.5, "loss": 2.3}')
        manager._log(job, 'METRIC {"event": "checkpoint", "checkpoint": "/safe/checkpoint", "step": 2}')
        self.assertEqual(job.progress, .5)
        self.assertEqual(job.checkpoint, "/safe/checkpoint")
        self.assertEqual(job.metrics[0]["loss"], 2.3)
        manager._log(job, 'RESULT {"status": "paused", "checkpoint": "/safe/checkpoint"}')
        self.assertEqual(manager.journal.get(job.id)["result"]["status"], "paused")
        manager._log(job, "METRIC not-json")
        self.assertEqual(job.progress, .5)

    def test_running_job_prevents_another_job_start(self):
        manager = JobManager(self.config)
        manager._current = _Job("abcdef1234", "sleep", state="running")
        with self.assertRaisesRegex(RuntimeError, "already running"):
            manager._start("sleep", {})

    def test_server_start_and_status_do_not_reenter_deadlock(self):
        server = RpServerProcess(self.config)
        process = SimpleNamespace(pid=1234, returncode=None, poll=lambda: None, terminate=lambda: None,
                                  wait=lambda timeout: None, kill=lambda: None)
        with patch("rp.ui.runner.httpx.get", side_effect=httpx.ConnectError("offline")), patch("rp.ui.runner.subprocess.Popen", return_value=process):
            started = time.monotonic()
            result = server.start()
            self.assertLess(time.monotonic() - started, 2)
            self.assertTrue(result["starting"])
            self.assertTrue(server.stop())

    def test_paused_worker_result_persists_job_state(self):
        manager = JobManager(self.config)
        job = _Job("abcdef1234", "sleep", config={"epochs": 2})
        output_lines = ['METRIC {"step": 1, "progress": 0.25, "loss": 2.0}',
                        'METRIC {"event": "checkpoint", "checkpoint": "/safe/checkpoint", "step": 1}',
                        'RESULT {"status": "paused", "checkpoint": "/safe/checkpoint"}']

        def launch(*args, **kwargs):
            kwargs["stdout"].write("\n".join(output_lines) + "\n")
            kwargs["stdout"].flush()
            return SimpleNamespace(pid=1234, returncode=0, poll=lambda: 0, wait=lambda: 0)

        with patch("rp.ui.runner.subprocess.Popen", side_effect=launch), patch("rp.ui.runner.psutil.Process", return_value=SimpleNamespace(create_time=lambda: 1)):
            manager._run(job)
        self.assertEqual(job.state, "paused")
        self.assertEqual(job.progress, .25)
        self.assertEqual(manager.journal.get(job.id)["state"], "paused")

    def test_training_exit_without_result_is_failed(self):
        manager = JobManager(self.config)
        job = _Job("abcdef1234", "sleep", config={"epochs": 2})

        def launch(*args, **kwargs):
            kwargs["stdout"].write("No completion record\n")
            kwargs["stdout"].flush()
            return SimpleNamespace(pid=1234, returncode=0, poll=lambda: 0, wait=lambda: 0)

        with patch("rp.ui.runner.subprocess.Popen", side_effect=launch), patch("rp.ui.runner.psutil.Process", return_value=SimpleNamespace(create_time=lambda: 1)):
            manager._run(job)
        self.assertEqual(job.state, "error")
        self.assertIsNotNone(job.error)


if __name__ == "__main__":
    unittest.main()
