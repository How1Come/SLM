import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
import psutil

from rp.server import create_app, parser
from rp import engine as engine_module
from tests.test_adapters import tiny_model


class MockEngine:
    def __init__(self):
        self.version = "base"
        self.calls = []
        self.output = "hello friend"
        self.failure = None
        self.fail_after_delta = False

    def activate(self, version):
        if version == "v999":
            raise FileNotFoundError("Missing adapter")
        self.version = version

    def generate(self, messages, prefill="", info=None, **parameters):
        if self.failure:
            raise self.failure
        self.calls.append((messages, parameters))
        info.update(adapter=self.version, prompt_tokens=5, completion_tokens=2, latency_s=.02, first_token_s=.01)
        return self.output, info

    def stream(self, messages, prefill="", info=None, cancel=None, **parameters):
        if self.failure:
            raise self.failure
        self.calls.append((messages, parameters))
        info.update(adapter=self.version, prompt_tokens=5, completion_tokens=2, latency_s=.02, first_token_s=.01)
        yield "hello "
        if self.fail_after_delta:
            raise ValueError("Sensitive internal stream failure")
        yield "friend"


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        environment = patch.dict(os.environ, {"RP_AUTH_TOKEN": ""})
        environment.start()
        self.addCleanup(environment.stop)
        self.config = parser().parse_args(["--data-dir", self.temporary.name, "--model", "test-model", "--auto-sleep-signals", "0"])
        self.engine = MockEngine()
        self.app = create_app(self.config, engine=self.engine, start_scheduler=False)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.messages = [{"role": "system", "content": "Play a kind character"}, {"role": "user", "content": "hello"}]

    def chat(self, **values):
        return self.client.post("/v1/chat/completions", json={"messages": self.messages, "session_id": "scene-one", **values})

    def test_model_list_and_non_streaming_generation_persist_exact_record(self):
        self.assertEqual(self.client.get("/v1/models").status_code, 200)
        response = self.chat(max_tokens=10, seed=42)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["choices"][0]["message"]["content"], "hello friend")
        self.assertEqual(result["usage"]["total_tokens"], 7)
        record = self.app.state.store.get_generation(result["rp_generation_id"])
        self.assertEqual(record["output"], "hello friend")
        self.assertEqual(record["session_id"], "scene-one")
        self.assertEqual(self.engine.calls[-1][1]["seed"], 42)

    def test_streaming_returns_openai_chunks_and_done(self):
        response = self.chat(stream=True, max_tokens=10)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn('"object": "chat.completion.chunk"', response.text)
        self.assertIn("data: [DONE]", response.text)
        self.assertEqual(self.app.state.store.stats()["generations"], 1)
        self.assertEqual(self.app.state.learning_state["busy"], 0)

    def test_invalid_generation_inputs_are_rejected_before_inference(self):
        invalid = [dict(messages=[]), dict(messages=[{"role": "tool", "content": "hello"}]), dict(max_tokens=-1),
                   dict(top_p=0), dict(temperature=3), dict(seed=-2), dict(stop=["x"] * 9),
                   dict(session_id=20), dict(top_k=-1)]
        for values in invalid:
            with self.subTest(values=values):
                response = self.chat(**values)
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.engine.calls, [])

    def test_empty_message_is_rejected_before_inference(self):
        response = self.chat(messages=[{"role": "user", "content": "   "}])
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.engine.calls, [])

    def test_feedback_requires_existing_generation_and_valid_correction(self):
        self.assertEqual(self.client.post("/rp/feedback", json={"gen_id": "missing", "rating": "good"}).status_code, 404)
        generated = self.chat().json()["rp_generation_id"]
        self.assertEqual(self.client.post("/rp/feedback", json={"gen_id": generated, "correction": "   "}).status_code, 422)
        response = self.client.post("/rp/feedback", json={"gen_id": generated, "correction": "A better roleplay response"})
        self.assertEqual(response.status_code, 200)
        record = self.app.state.store.get_signal(response.json()["signal_id"])
        self.assertEqual(record["chosen"], "A better roleplay response")
        self.assertEqual(record["rejected"], "hello friend")

    def test_session_identity_isolates_same_turn_feedback(self):
        first = self.chat(session_id="first").json()["rp_generation_id"]
        second = self.chat(session_id="second").json()["rp_generation_id"]
        self.assertEqual(self.app.state.store.active_signals(), [])
        edited = self.messages + [{"role": "assistant", "content": "edited second reply"}, {"role": "user", "content": "continue"}]
        self.chat(messages=edited, session_id="second")
        signals = self.app.state.store.active_signals()
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["gen_id"], second)
        self.assertNotEqual(signals[0]["gen_id"], first)

    def test_learning_disabled_does_not_log_or_accept_feedback(self):
        settings = self.client.patch("/rp/settings", json={"enabled": False})
        self.assertEqual(settings.status_code, 200)
        response = self.chat()
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["rp_generation_id"])
        self.assertEqual(self.app.state.store.stats()["generations"], 0)
        self.assertEqual(self.client.post("/rp/feedback", json={"rating": "good"}).status_code, 409)
        self.assertFalse(self.app.state.auto_sleep_check())

    def test_generation_failure_has_friendly_error_without_traceback(self):
        self.engine.failure = ValueError("Sensitive internal trace")
        response = self.chat()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("Sensitive", response.text)
        self.assertNotIn("Traceback", response.text)
        self.assertEqual(self.app.state.learning_state["busy"], 0)

    def test_streaming_failure_has_friendly_error_and_preserves_dataset(self):
        self.engine.failure = ValueError("Sensitive internal trace")
        response = self.chat(stream=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("error", response.text)
        self.assertIn("data: [DONE]", response.text)
        self.assertNotIn("Sensitive", response.text)
        self.assertEqual(self.app.state.store.stats()["generations"], 0)
        self.assertEqual(self.app.state.learning_state["busy"], 0)

    def test_partial_stream_failure_is_not_training_data(self):
        self.engine.fail_after_delta = True
        response = self.chat(stream=True)
        self.assertIn("hello", response.text)
        self.assertIn("generation_error", response.text)
        self.assertNotIn("Sensitive", response.text)
        self.assertEqual(self.app.state.store.stats()["generations"], 0)

    def test_real_tiny_qwen_stream_is_persisted_after_completion(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name)
        model = tiny_model(str(path / "model"))
        config = parser().parse_args(["--data-dir", str(path / "data"), "--model", model, "--max-context", "128",
                                      "--max-tokens", "4", "--auto-sleep-signals", "0"])
        with patch.object(engine_module, "BUNDLED_NARRATION", str(path / "missing")):
            actual_engine = engine_module.Engine(model, config.data_dir, threads=1, max_context=128)
            app = create_app(config, engine=actual_engine, start_scheduler=False)
            with TestClient(app) as client:
                response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello world"}],
                                              "stream": True, "max_tokens": 4, "temperature": 0, "session_id": "real-stream"})
                self.assertEqual(response.status_code, 200)
                self.assertIn("data: [DONE]", response.text)
                self.assertNotIn("generation_error", response.text)
                self.assertEqual(app.state.store.stats()["generations"], 1)

    def test_authentication_blocks_health_and_chat_without_token(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = parser().parse_args(["--data-dir", temporary.name, "--auth-token", "a" * 32])
        app = create_app(config, engine=MockEngine(), start_scheduler=False)
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 401)
            self.assertEqual(client.post("/v1/chat/completions", json={"messages": self.messages}).status_code, 401)
            self.assertEqual(client.get("/health", headers={"Authorization": "Bearer " + "a" * 32}).status_code, 200)
            self.assertEqual(client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code, 401)

    def test_public_binding_requires_long_access_token(self):
        config = parser().parse_args(["--data-dir", self.temporary.name, "--host", "0.0.0.0", "--auth-token", "short"])
        with self.assertRaisesRegex(ValueError, "RP_AUTH_TOKEN"):
            create_app(config, engine=MockEngine(), start_scheduler=False)

    def test_idle_learning_waits_for_idle_time_and_no_active_chat(self):
        generated = self.chat().json()["rp_generation_id"]
        self.client.post("/rp/feedback", json={"gen_id": generated, "rating": "good"})
        self.app.state.learning.update({"auto_sleep": True, "min_signals": 1, "idle_seconds": 10, "mode": "idle"})
        with patch.object(self.app.state.sleeper, "start", return_value={"running": True}) as start:
            self.assertFalse(self.app.state.auto_sleep_check())
            self.app.state.learning_state["last_request"] = time.time() - 20
            self.app.state.learning_state["busy"] = 1
            self.assertFalse(self.app.state.auto_sleep_check())
            self.app.state.learning_state["busy"] = 0
            self.assertTrue(self.app.state.auto_sleep_check())
            start.assert_called_once()

    def test_background_learning_can_train_during_chat_and_collect_only_cannot(self):
        generated = self.chat().json()["rp_generation_id"]
        self.client.post("/rp/feedback", json={"gen_id": generated, "rating": "good"})
        self.app.state.learning.update({"auto_sleep": True, "min_signals": 1, "mode": "collect_only"})
        with patch.object(self.app.state.sleeper, "start", return_value={"running": True}) as start:
            self.assertFalse(self.app.state.auto_sleep_check())
            self.app.state.learning.update({"mode": "background"})
            self.app.state.learning_state["busy"] = 1
            self.assertTrue(self.app.state.auto_sleep_check())
            start.assert_called_once()

    def test_bad_only_feedback_does_not_start_auto_training(self):
        generated = self.chat().json()["rp_generation_id"]
        self.client.post("/rp/feedback", json={"gen_id": generated, "rating": "bad"})
        self.app.state.learning.update({"auto_sleep": True, "min_signals": 1, "mode": "background"})
        with patch.object(self.app.state.sleeper, "start") as start:
            self.assertFalse(self.app.state.auto_sleep_check())
            start.assert_not_called()

    def test_auto_training_resumes_paused_run_instead_of_restarting(self):
        generated = self.chat().json()["rp_generation_id"]
        self.client.post("/rp/feedback", json={"gen_id": generated, "rating": "good"})
        self.app.state.learning.update({"auto_sleep": True, "min_signals": 1, "mode": "background"})
        self.app.state.sleeper.last_result = {"status": "paused"}
        self.app.state.sleeper.checkpoint = "/safe/paused-checkpoint"
        with patch.object(self.app.state.sleeper, "start", return_value={"running": True}) as start:
            self.assertTrue(self.app.state.auto_sleep_check())
        self.assertEqual(start.call_args.kwargs["resume"], "/safe/paused-checkpoint")

    def test_invalid_adapter_selection_cannot_change_current(self):
        self.assertEqual(self.client.post("/rp/activate", json={"version": "../../secrets"}).status_code, 422)
        self.assertEqual(self.client.post("/rp/activate", json={"version": "v999"}).status_code, 404)
        self.assertEqual(self.engine.version, "base")

    def test_invalid_sleep_settings_are_rejected_without_process(self):
        with patch("rp.server.subprocess.Popen") as start:
            response = self.client.post("/rp/sleep", json={"args": {"epochs": 0}})
            self.assertEqual(response.status_code, 422)
            self.assertFalse(self.app.state.sleeper.running)
            start.assert_not_called()

    def test_invalid_wait_timeout_does_not_start_training(self):
        with patch.object(self.app.state.sleeper, "start", return_value={"running": True}) as start:
            response = self.client.post("/rp/sleep", json={"wait": True, "timeout": 0})
            self.assertEqual(response.status_code, 422)
            start.assert_not_called()

    def test_previous_training_finalization_cannot_race_new_process_start(self):
        sleeper = self.app.state.sleeper
        sleeper.proc = SimpleNamespace(poll=lambda: 0)
        sleeper.started = time.time()
        sleeper.finished = None
        sleeper.progress = .75
        with patch("rp.server.subprocess.Popen") as start:
            state = sleeper.start({"epochs": 2})
            start.assert_not_called()
        self.assertFalse(state["running"])
        self.assertEqual(state["started"], sleeper.started)
        self.assertIsNone(state["finished"])
        self.assertEqual(state["progress"], .75)

    def test_live_training_job_from_other_runner_blocks_new_sleep_process(self):
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
        sleeper = self.app.state.sleeper
        sleeper.journal.save({"id": "abcdef1234", "kind": "narration_train", "state": "running",
                              "pid": process.pid, "process_created": psutil.Process(process.pid).create_time(),
                              "started_at": time.time()})
        with patch("rp.server.subprocess.Popen") as start:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                sleeper.start({"epochs": 2})
            start.assert_not_called()
        self.assertIsNone(process.poll())

    def test_queued_training_job_from_other_runner_blocks_new_sleep_process(self):
        sleeper = self.app.state.sleeper
        sleeper.journal.save({"id": "abcdef1234", "kind": "narration_train", "state": "queued",
                              "pid": None, "started_at": time.time()})
        with patch("rp.server.subprocess.Popen") as start:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                sleeper.start({"epochs": 2})
            start.assert_not_called()

    def test_auto_training_checkpoint_metrics_preserve_progress(self):
        lines = ['METRIC {"step":1,"progress":0.25,"loss":2.0}',
                 'METRIC {"event":"checkpoint","checkpoint":"/safe/checkpoint","step":1}',
                 'RESULT {"status":"paused","checkpoint":"/safe/checkpoint"}']

        def launch(*args, **kwargs):
            kwargs["stdout"].write("\n".join(lines) + "\n")
            kwargs["stdout"].flush()
            return SimpleNamespace(pid=1234, poll=lambda: 0, wait=lambda: 0)

        with patch("rp.server.subprocess.Popen", side_effect=launch), patch("rp.server.psutil.Process", return_value=SimpleNamespace(create_time=lambda: 1)):
            self.app.state.sleeper.start({"epochs": 2})
            state = self.app.state.sleeper.wait(timeout=2)
        self.assertEqual(state["last_result"]["status"], "paused")
        self.assertEqual(state["progress"], .25)


if __name__ == "__main__":
    unittest.main()
