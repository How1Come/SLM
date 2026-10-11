import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from rp import engine
from rp.sleep import next_version


def tiny_model(directory: str) -> str:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    vocabulary = {value: index for index, value in enumerate([
        "[PAD]", "[UNK]", "[EOS]", "system", "user", "assistant", "hello", "world", "good", "bad",
        "be", "kind", "answer", "story", "friend", "scene", "continue", "learn", "one", "two"])}
    raw = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' [EOS] ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    tokenizer.save_pretrained(directory)
    config = Qwen2Config(vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                         eos_token_id=2, pad_token_id=0, bos_token_id=None, tie_word_embeddings=True)
    Qwen2ForCausalLM(config).save_pretrained(directory)
    return directory


def narration_adapter(model_path: str, output: str):
    base = AutoModelForCausalLM.from_pretrained(model_path)
    adapter = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                                            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.25)
    adapter.save_pretrained(output)
    return adapter


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model_path = tiny_model(str(self.root / "tiny"))
        self.data = str(self.root / "data")
        self.patch = patch.object(engine, "BUNDLED_NARRATION", str(self.root / "missing"))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.addCleanup(self.temporary.cleanup)

    def logits(self, model):
        with torch.no_grad():
            return model(input_ids=torch.tensor([[6, 7, 6]])).logits.clone()

    def test_explicit_base_is_preserved_with_narration_available(self):
        narration_adapter(self.model_path, str(self.root / "data" / "adapters" / "narration"))
        self.assertEqual(engine.read_current(self.data), "narration")
        engine.write_current(self.data, "base")
        self.assertEqual(engine.read_current(self.data), "base")

    def test_narration_and_base_switch_reload_pristine_weights(self):
        narration_adapter(self.model_path, str(self.root / "data" / "adapters" / "narration"))
        engine.write_current(self.data, "base")
        service = engine.Engine(self.model_path, self.data, dtype="float32", threads=1)
        baseline = self.logits(service.model)
        service.activate("narration")
        self.assertFalse(torch.equal(baseline, self.logits(service.model)))
        service.activate("narration")
        repeated = self.logits(service.model)
        service.activate("base")
        self.assertTrue(torch.equal(baseline, self.logits(service.model)))
        service.activate("narration")
        self.assertTrue(torch.equal(repeated, self.logits(service.model)))
        service.activate("base")
        self.assertTrue(torch.equal(baseline, self.logits(service.model)))
        self.assertFalse(any("lora_" in name for name, _ in service.model.named_parameters()))

    def test_chat_adapter_uses_immutable_narration_snapshot(self):
        narration_path = self.root / "data" / "adapters" / "narration"
        narration_adapter(self.model_path, str(narration_path))
        output = self.root / "data" / "adapters" / "v1"
        foundation = engine.snapshot_foundation(self.data, str(output), self.model_path)
        base = engine.load_foundation(self.model_path, torch.float32, foundation, str(output))
        adapter = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"))
        adapter.save_pretrained(str(output))
        (output / "meta.json").write_text(json.dumps({"foundation": foundation}), encoding="utf-8")
        service = engine.Engine(self.model_path, self.data, dtype="float32")
        service.activate("v1")
        before = self.logits(service.model)
        narration_adapter(self.model_path, str(narration_path))
        weights = narration_path / "adapter_model.safetensors"
        weights.write_bytes(weights.read_bytes() + b"changed")
        service.activate("v1")
        self.assertTrue(torch.equal(before, self.logits(service.model)))
        self.assertEqual(service.model.active_adapters, ["default"])
        service.activate("base")
        self.assertFalse(torch.equal(before, self.logits(service.model)))

    def test_tampered_foundation_is_rejected(self):
        narration_adapter(self.model_path, str(self.root / "data" / "adapters" / "narration"))
        output = self.root / "data" / "adapters" / "v1"
        foundation = engine.snapshot_foundation(self.data, str(output), self.model_path)
        config = output / "foundation" / "narration" / "adapter_config.json"
        config.write_text(config.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "modified"):
            engine.load_foundation(self.model_path, torch.float32, foundation, str(output))

    def test_numbering_ignores_narration_and_reserves_paused_versions(self):
        path = self.root / "data" / "adapters"
        for name in ("narration", "v2", "v10", "versions", "vbad", "v0"):
            (path / name).mkdir(parents=True, exist_ok=True)
        self.assertEqual(next_version(self.data), "v11")

    def test_adapter_config_compatibility_rejects_active_unknown_features(self):
        output = self.root / "adapter"
        narration_adapter(self.model_path, str(output))
        config_file = output / "adapter_config.json"
        config = json.loads(config_file.read_text())
        config.update(unknown_future_feature=None, peft_version="99.0.0")
        config_file.write_text(json.dumps(config))
        base = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.assertIsNotNone(engine.load_adapter(base, str(output)))
        config["unknown_future_feature"] = True
        config_file.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "newer PEFT"):
            engine.load_adapter(AutoModelForCausalLM.from_pretrained(self.model_path), str(output))

    def test_legacy_nested_named_adapter_is_loadable(self):
        root = self.root / "data" / "adapters" / "v1"
        base = AutoModelForCausalLM.from_pretrained(self.model_path)
        adapter = get_peft_model(base, LoraConfig(r=2, target_modules=["q_proj"], task_type="CAUSAL_LM"), adapter_name="v1")
        adapter.save_pretrained(str(root), selected_adapters=["v1"])
        (root / "meta.json").write_text(json.dumps({"base_model": self.model_path}))
        self.assertEqual(engine.adapter_path(self.data, "v1"), str(root / "v1"))
        service = engine.Engine(self.model_path, self.data, dtype="float32")
        service.activate("v1")
        self.assertEqual(service.model.active_adapters, ["default"])
        self.assertTrue(engine.list_adapters(self.data)[0]["legacy_foundation"])

    def test_path_traversal_version_is_rejected(self):
        with self.assertRaises(ValueError):
            engine.adapter_path(self.data, "../../secrets")
        with self.assertRaises(ValueError):
            engine.write_current(self.data, "../../secrets")

    def test_real_qwen_generation_accepts_generic_tokenizer_and_records_completion(self):
        import threading
        service = engine.Engine(self.model_path, self.data, dtype="float32", threads=1)
        cancel = threading.Event()
        info = {}
        text = "".join(service.stream([{"role": "user", "content": "hello world"}], max_tokens=4,
                                     temperature=0, cancel=cancel, info=info))
        self.assertIsInstance(text, str)
        self.assertTrue(info["completed"])
        self.assertGreater(info["completion_tokens"], 0)
        self.assertFalse(cancel.is_set())

    def test_long_chat_reserves_reply_space_and_keeps_latest_turn(self):
        service = engine.Engine(self.model_path, self.data, dtype="float32", threads=1, max_context=48)
        messages = [{"role": "system", "content": "be kind"}]
        for _ in range(8):
            messages.extend([{"role": "user", "content": "hello world one two"},
                             {"role": "assistant", "content": "good story friend scene"}])
        messages.append({"role": "user", "content": "continue"})
        text, info = service.generate(messages, prefill="good ", max_tokens=8, temperature=0)
        self.assertIsInstance(text, str)
        self.assertLessEqual(info["prompt_tokens"] + 8, 48)
        self.assertLess(info["prompt_tokens"], len(engine.prompt_ids(service.tok, messages)))
        rendered = service.render(messages, prompt_budget=38)
        self.assertIn("be kind", rendered)
        self.assertIn("continue", rendered)

    def test_context_limit_tracks_model_and_oversized_card_fails_clearly(self):
        service = engine.Engine(self.model_path, self.data, dtype="float32", max_context=3072)
        self.assertEqual(service.max_context, 256)
        with self.assertRaisesRegex(ValueError, "context limit"):
            service.generate([{"role": "system", "content": "good " * 256}, {"role": "user", "content": "hello"}], max_tokens=4)

    def test_closing_stream_cancels_worker_and_releases_model_lock(self):
        import threading
        service = engine.Engine(self.model_path, self.data, dtype="float32", threads=1)
        cancel = threading.Event()
        info = {}
        stream = service.stream([{"role": "user", "content": "hello"}], max_tokens=64,
                                temperature=0, cancel=cancel, info=info)
        next(stream)
        stream.close()
        self.assertTrue(cancel.is_set())
        self.assertFalse(info["completed"])
        self.assertTrue(service.lock.acquire(timeout=.5))
        service.lock.release()


if __name__ == "__main__":
    unittest.main()
