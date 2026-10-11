import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from rp import engine, sleep, train_narration
from rp.store import Store
from tests.test_adapters import narration_adapter, tiny_model


class SleepTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_path = tiny_model(str(self.root / "tiny"))
        self.patch = patch.object(engine, "BUNDLED_NARRATION", str(self.root / "missing"))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.temporary.cleanup)
        self.messages = [{"role": "system", "content": "be kind"}, {"role": "user", "content": "hello"}]
        self.canary = [{"messages": self.messages, "response": "good world"}]

    def data(self, name="data"):
        directory = str(self.root / name)
        store = Store(directory)
        store.add_signal("fix", "turn-one", None, self.messages, chosen="good world", rejected="bad world")
        store.add_signal("fix", "turn-two", None,
                         self.messages[:-1] + [{"role": "user", "content": "story"}], chosen="kind friend", rejected="bad friend")
        return directory, store

    def args(self, data):
        return sleep.parser().parse_args(["--data-dir", data, "--model", self.model_path, "--epochs", "2", "--rank", "2",
                                          "--accum", "1", "--max-prompt-tokens", "48", "--max-response-tokens", "12",
                                          "--threads", "1", "--checkpoint-every", "1", "--no-activate", "--max-minutes", "10"])

    def run_sleep(self, args, event=None):
        with patch.object(sleep, "canary_set", return_value=self.canary):
            return sleep.run(args, event)

    def test_real_optimizer_saves_root_adapter_and_reload_matches(self):
        directory, _ = self.data()
        result = self.run_sleep(self.args(directory))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["optimizer_steps"], 4)
        adapter = Path(directory) / "adapters" / result["version"]
        self.assertTrue((adapter / "adapter_config.json").is_file())
        self.assertFalse((adapter / result["version"]).exists())
        self.assertTrue((Path(result["checkpoint"]) / "state.pt").is_file())
        service = engine.Engine(self.model_path, directory, dtype="float32")
        service.activate(result["version"])
        self.assertEqual(service.model.active_adapters, ["default"])
        weights = load_file(str(adapter / "adapter_model.safetensors"))
        self.assertTrue(any(bool(tensor.abs().sum()) for name, tensor in weights.items() if "lora_B" in name))

    def test_pause_resume_matches_uninterrupted_training_with_real_optimizer(self):
        uninterrupted, _ = self.data("full")
        expected = self.run_sleep(self.args(uninterrupted))
        directory, store = self.data("paused")
        stopped = threading.Event()
        original_metric = sleep.emit_metric

        def interrupt(**values):
            original_metric(**values)
            if values.get("step") == 1 and values.get("event") != "checkpoint":
                stopped.set()

        with patch.object(sleep, "emit_metric", side_effect=interrupt):
            paused = self.run_sleep(self.args(directory), stopped)
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["step"], 1)
        self.assertEqual(engine.read_current(directory), "base")
        self.assertTrue(all(item["consumed_in"] is None for item in store.active_signals()))
        store.add_signal("fix", "later-turn", None, self.messages, chosen="one two", rejected="bad")
        args = self.args(directory)
        args.resume = paused["checkpoint"]
        args.no_activate = False
        resumed = self.run_sleep(args)
        self.assertEqual(resumed["version"], paused["version"])
        self.assertEqual(resumed["items"], 2)
        self.assertFalse(resumed["activated"])
        expected_weights = load_file(str(Path(uninterrupted) / "adapters" / expected["version"] / "adapter_model.safetensors"))
        actual_weights = load_file(str(Path(directory) / "adapters" / resumed["version"] / "adapter_model.safetensors"))
        for name, tensor in expected_weights.items():
            self.assertTrue(torch.equal(tensor, actual_weights[name]), name)

    def test_sleep_training_and_serving_share_narration_foundation(self):
        directory, _ = self.data()
        narration_adapter(self.model_path, str(Path(directory) / "adapters" / "narration"))
        result = self.run_sleep(self.args(directory))
        self.assertEqual(result["foundation"]["kind"], "narration")
        output = Path(directory) / "adapters" / result["version"]
        foundation = engine.load_foundation(self.model_path, torch.float32, result["foundation"], str(output))
        expected = engine.load_adapter(foundation, str(output))
        service = engine.Engine(self.model_path, directory, dtype="float32")
        service.activate(result["version"])
        inputs = torch.tensor([[6, 7, 8]])
        with torch.no_grad():
            self.assertTrue(torch.equal(expected(inputs).logits, service.model(inputs).logits))
        self.assertTrue((output / "foundation" / "narration" / "adapter_model.safetensors").exists())

    def test_guard_failure_preserves_active_version_and_unconsumed_signals(self):
        directory, store = self.data()
        args = self.args(directory)
        args.no_activate = False
        with patch.object(sleep, "canary_nll", side_effect=[1.0, 10.0]):
            result = self.run_sleep(args)
        self.assertFalse(result["activated"])
        self.assertFalse(result["guard_passed"])
        self.assertEqual(engine.read_current(directory), "base")
        self.assertTrue(all(item["consumed_in"] is None for item in store.active_signals()))

    def test_successful_guard_activates_and_consumes_snapshot(self):
        directory, store = self.data()
        args = self.args(directory)
        args.no_activate = False
        args.guard_max_drift = 100
        result = self.run_sleep(args)
        self.assertTrue(result["activated"])
        self.assertEqual(engine.read_current(directory), result["version"])
        self.assertTrue(all(item["consumed_in"] == result["version"] for item in store.active_signals()))

    def test_withdrawn_feedback_blocks_automatic_activation(self):
        directory, store = self.data()
        args = self.args(directory)
        args.no_activate = False
        args.guard_max_drift = 100
        original_metric = sleep.emit_metric
        first = store.active_signals()[0]["id"]

        def withdraw(**values):
            original_metric(**values)
            if values.get("step") == 1 and values.get("event") != "checkpoint":
                store.review_signal(first, False)

        with patch.object(sleep, "emit_metric", side_effect=withdraw):
            result = self.run_sleep(args)
        self.assertTrue(result["dataset_changed_since_start"])
        self.assertFalse(result["activated"])
        self.assertEqual(engine.read_current(directory), "base")
        self.assertTrue(all(item["consumed_in"] is None for item in store.active_signals()))

    def test_checkpoint_snapshot_tamper_is_rejected(self):
        directory, _ = self.data()
        event = threading.Event()
        event.set()
        result = self.run_sleep(self.args(directory), event)
        path = Path(result["checkpoint"])
        metadata = json.loads((path / "checkpoint.json").read_text())
        metadata["dataset_sha256"] = "incorrect"
        (path / "checkpoint.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "modified"):
            sleep.read_checkpoint(str(path), "sleep")

    def test_numbering_with_narration_does_not_crash(self):
        directory, _ = self.data()
        (Path(directory) / "adapters" / "narration").mkdir(parents=True)
        self.assertEqual(sleep.next_version(directory), "v1")

    def test_paused_base_only_run_reserves_its_adapter_version(self):
        directory, _ = self.data()
        event = threading.Event()
        event.set()
        paused = self.run_sleep(self.args(directory), event)
        self.assertEqual(paused["version"], "v1")
        self.assertTrue((Path(directory) / "adapters" / "v1").is_dir())
        fresh = self.run_sleep(self.args(directory))
        self.assertEqual(fresh["version"], "v2")
        wrong_directory = str(self.root / "wrong-resume-directory")
        args = self.args(wrong_directory)
        args.resume = paused["checkpoint"]
        resumed = self.run_sleep(args)
        self.assertEqual(resumed["version"], "v1")
        self.assertEqual(args.data_dir, directory)
        self.assertFalse(Path(wrong_directory).exists())
        self.assertEqual([item["version"] for item in engine.list_adapters(directory)], ["v1", "v2"])

    def test_duplicate_contexts_with_different_systems_keep_sft_terms(self):
        rows = []
        for index, system in enumerate(("character one", "character two"), 1):
            rows.append({"id": index, "messages": json.dumps([{"role": "system", "content": system},
                          {"role": "user", "content": "hello"}]), "chosen": "hello friend", "rejected": None,
                         "weight": 1.0, "kind": "fix", "turn_key": str(index)})
        self.assertTrue(all(item["sft"] for item in sleep.build_examples(rows, 10)))

    def test_unresolved_swipes_never_become_negative_training_data(self):
        rows = [{"id": 1, "messages": json.dumps(self.messages), "chosen": None, "rejected": "bad world",
                 "weight": 0.3, "kind": "swipe", "turn_key": "turn"}]
        self.assertEqual(sleep.build_examples(rows, 10), [])

    def test_real_narration_resume_exits_time_budget_then_finishes(self):
        data = self.root / "narration.jsonl"
        data.write_text("\n".join(json.dumps({"messages": self.messages, "response": response})
                                  for response in ("good world", "kind friend")), encoding="utf-8")
        args = train_narration.parser().parse_args(["--data", str(data), "--out", str(self.root / "trained-narration"),
                    "--model", self.model_path, "--epochs", "2", "--min-examples", "1", "--rank", "2",
                    "--max-prompt-tokens", "48", "--max-response-tokens", "12", "--threads", "1"])
        stopped = threading.Event()
        stopped.set()
        paused = train_narration.run(args, stopped)
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["step"], 0)
        args.resume = paused["checkpoint"]
        wrong_output = str(self.root / "wrong-narration-output")
        args.out = wrong_output
        result = train_narration.run(args)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["optimizer_steps"], 4)
        self.assertTrue((Path(args.out) / "adapter_config.json").exists())
        self.assertFalse(Path(wrong_output + ".training.lock").exists())

    def test_training_argument_validation(self):
        args = self.args(str(self.root / "invalid"))
        args.accum = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            sleep.validate_args(args)
        args.accum = 1
        args.max_minutes = float("nan")
        with self.assertRaisesRegex(ValueError, "time budget"):
            sleep.validate_args(args)

    def test_narration_parser_rejects_malformed_samples(self):
        path = self.root / "invalid.jsonl"
        path.write_text('{"messages": []}')
        with self.assertRaisesRegex(ValueError, "messages list"):
            train_narration.load_jsonl(str(path))
        path.write_text("{invalid json}")
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            train_narration.load_jsonl(str(path))


if __name__ == "__main__":
    unittest.main()
