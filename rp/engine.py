"""CPU inference and immutable LoRA foundations."""
from __future__ import annotations

import hashlib
import dataclasses
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

BUNDLED_NARRATION = os.path.join(os.path.dirname(__file__), "bundled_adapters", "narration")
NARRATION_VERSION = "narration"
VERSION_PATTERN = re.compile(r"v[1-9][0-9]*\Z")


def adapters_dir(data_dir: str) -> str:
    return os.path.join(data_dir, "adapters")


def _adapter_ok(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json")) and any(
        os.path.isfile(os.path.join(path, name)) for name in ("adapter_model.safetensors", "adapter_model.bin"))


def adapter_path(data_dir: str, version: str) -> str:
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError("Choose base, narration, or a numbered adapter such as v1.")
    path = os.path.join(adapters_dir(data_dir), version)
    if _adapter_ok(path):
        return path
    nested = os.path.join(path, version)
    if _adapter_ok(nested):
        return nested
    raise FileNotFoundError(f"Adapter {version} has no saved weights.")


def narration_adapter_dir(data_dir: str) -> str | None:
    local = os.path.join(adapters_dir(data_dir), NARRATION_VERSION)
    if _adapter_ok(local):
        return local
    return BUNDLED_NARRATION if _adapter_ok(BUNDLED_NARRATION) else None


def ensure_narration_adapter(data_dir: str) -> str | None:
    dest = os.path.join(adapters_dir(data_dir), NARRATION_VERSION)
    if _adapter_ok(dest):
        return dest
    if not _adapter_ok(BUNDLED_NARRATION):
        return None
    os.makedirs(adapters_dir(data_dir), exist_ok=True)
    if not os.path.exists(dest):
        staging = dest + ".tmp-" + str(time.time_ns())
        shutil.copytree(BUNDLED_NARRATION, staging)
        try:
            os.rename(staging, dest)
        except FileExistsError:
            shutil.rmtree(staging)
    return dest if _adapter_ok(dest) else None


def adapter_hash(path: str) -> str:
    digest = hashlib.sha256()
    files = sorted(p for p in Path(path).rglob("*") if p.is_file() and
                   (p.name == "adapter_config.json" or p.suffix in {".safetensors", ".bin"}))
    if not files or not _adapter_ok(path):
        raise ValueError("Adapter configuration and weights are required.")
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode())
        with file.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def local_model_hash(path: str) -> str | None:
    root = Path(path)
    if not root.is_dir():
        return None
    files = {file for pattern in ("config.json", "model*.safetensors", "pytorch_model*.bin", "tokenizer*", "special_tokens_map.json")
             for file in root.glob(pattern) if file.is_file()}
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(file.name.encode())
        with file.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def narration_matches_model(path: str, model_name: str) -> bool:
    with open(os.path.join(path, "adapter_config.json"), encoding="utf-8") as source:
        identity = json.load(source).get("base_model_name_or_path")
    return not identity or identity == model_name


def snapshot_foundation(data_dir: str, output_dir: str, model_name: str, mode: str = "auto") -> dict:
    os.makedirs(output_dir, exist_ok=False)
    foundation = {"kind": "base", "base_model": model_name}
    model_digest = local_model_hash(model_name)
    if model_digest:
        foundation["base_sha256"] = model_digest
    narration = ensure_narration_adapter(data_dir) if mode != "base" else None
    if not narration:
        if mode == "narration":
            raise FileNotFoundError("No narration adapter is available for this training foundation.")
        return foundation
    if not narration_matches_model(narration, model_name):
        if mode == "narration":
            raise ValueError("The installed narration adapter belongs to a different base model.")
        return foundation
    dest = os.path.join(output_dir, "foundation", "narration")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        raise FileExistsError("This training foundation already exists; resume its checkpoint.")
    original_hash = adapter_hash(narration)
    shutil.copytree(narration, dest)
    snapshot_hash = adapter_hash(dest)
    if original_hash != snapshot_hash:
        raise ValueError("Narration changed while its foundation was being copied. Finish narration training, then start a new run.")
    return {**foundation, "kind": "narration", "path": "foundation/narration", "sha256": snapshot_hash}


def load_foundation(model_name: str, dtype: torch.dtype, foundation: dict | None = None,
                    artifact_dir: str | None = None):
    if foundation and foundation.get("base_sha256") and local_model_hash(model_name) != foundation["base_sha256"]:
        raise ValueError("The local base model or tokenizer changed. Restore the original files before resuming this adapter.")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,
                                               revision=(foundation or {}).get("revision"))
    if foundation and foundation.get("base_model", model_name) != model_name:
        raise ValueError("The adapter was trained with a different base model.")
    if foundation and foundation.get("kind") == "narration":
        if not artifact_dir:
            raise ValueError("The narration foundation location is missing.")
        root = Path(artifact_dir).resolve()
        narration = (root / foundation["path"]).resolve()
        if not narration.is_relative_to(root):
            raise ValueError("The foundation path must stay inside its adapter directory.")
        if foundation.get("sha256") and adapter_hash(str(narration)) != foundation["sha256"]:
            raise ValueError("The frozen narration foundation was modified; restore the original snapshot.")
        model = load_adapter(model, str(narration)).merge_and_unload(safe_merge=True)
        if hasattr(model, "peft_config"):
            del model.peft_config
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_adapter(model, path: str, trainable: bool = False):
    from peft import LoraConfig, PeftModel
    with open(os.path.join(path, "adapter_config.json"), encoding="utf-8") as source:
        config = json.load(source)
    if config.get("peft_type") != "LORA":
        raise ValueError("Only LoRA adapters are supported by this CPU worker.")
    known = {field.name for field in dataclasses.fields(LoraConfig)}
    unsupported = {key: value for key, value in config.items() if key not in known and key != "peft_version"
                   and value not in (None, False, {}, [], "")}
    if unsupported:
        raise ValueError("This adapter requires a newer PEFT release: " + ", ".join(sorted(unsupported)))
    compatible = LoraConfig(**{key: value for key, value in config.items() if key in known})
    return PeftModel.from_pretrained(model, path, config=compatible, is_trainable=trainable)


def read_current(data_dir: str, model_name: str | None = None) -> str:
    path = os.path.join(adapters_dir(data_dir), "current.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as source:
            version = json.load(source).get("version")
        if version in {"base", NARRATION_VERSION}:
            return version
        if isinstance(version, str) and VERSION_PATTERN.fullmatch(version):
            adapter_path(data_dir, version)
            return version
        raise ValueError("current.json contains an invalid adapter selection.")
    narration = narration_adapter_dir(data_dir)
    return NARRATION_VERSION if narration and (not model_name or narration_matches_model(narration, model_name)) else "base"


def write_current(data_dir: str, version: str) -> None:
    if version not in {"base", NARRATION_VERSION} and not VERSION_PATTERN.fullmatch(version):
        raise ValueError("Invalid adapter version.")
    os.makedirs(adapters_dir(data_dir), exist_ok=True)
    path = os.path.join(adapters_dir(data_dir), "current.json")
    staging = path + ".tmp-" + str(time.time_ns())
    with open(staging, "w", encoding="utf-8") as destination:
        json.dump({"version": version, "ts": time.time()}, destination)
    os.replace(staging, path)


def list_adapters(data_dir: str) -> list[dict]:
    entries = []
    narration = narration_adapter_dir(data_dir)
    if narration:
        entries.append((NARRATION_VERSION, narration))
    directory = adapters_dir(data_dir)
    if os.path.isdir(directory):
        for name in sorted((n for n in os.listdir(directory) if VERSION_PATTERN.fullmatch(n)), key=lambda n: int(n[1:])):
            try:
                entries.append((name, adapter_path(data_dir, name)))
            except FileNotFoundError:
                continue
    results = []
    for version, path in entries:
        meta_path = os.path.join(directory, version, "meta.json") if version != NARRATION_VERSION else os.path.join(path, "meta.json")
        meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as source:
                meta = json.load(source)
        results.append({**meta, "version": version,
                        "legacy_foundation": version != NARRATION_VERSION and "foundation" not in meta})
    return results


def prompt_ids(tok, messages: list[dict]) -> list[int]:
    output = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(output, list):
        ids = output
    elif torch.is_tensor(output):
        ids = output.tolist()
    else:
        ids = output["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def truncate_messages(tok, messages: list[dict], max_tokens: int) -> list[dict]:
    messages = list(messages)
    while True:
        length = len(prompt_ids(tok, messages))
        droppable = [i for i, message in enumerate(messages) if message["role"] != "system"]
        if length <= max_tokens or len(droppable) <= 1:
            return messages
        del messages[droppable[0]]


class _Cancel(StoppingCriteria):
    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class Engine:
    def __init__(self, model_name: str, data_dir: str, dtype: str = "float32", threads: int | None = None,
                 max_context: int = 3072, initial_version: str | None = None):
        if threads:
            torch.set_num_threads(threads)
        self.model_name = model_name
        self.data_dir = data_dir
        self.dtype = getattr(torch, dtype)
        self.configured_max_context = max_context
        self.max_context = max_context
        self.tok = None
        self.model = None
        self.version = None
        self.peft = False
        self.lock = threading.RLock()
        ensure_narration_adapter(data_dir)
        self.activate(initial_version if initial_version is not None else read_current(data_dir, model_name))

    def activate(self, version: str) -> None:
        with self.lock:
            foundation = None
            if version == "base":
                model = load_foundation(self.model_name, self.dtype)
                peft = False
            elif version == NARRATION_VERSION:
                path = ensure_narration_adapter(self.data_dir) or narration_adapter_dir(self.data_dir)
                if not path:
                    raise FileNotFoundError("No narration adapter is available. Train narration first.")
                if not narration_matches_model(path, self.model_name):
                    raise ValueError("This narration adapter belongs to a different base model.")
                foundation = {"kind": "narration", "path": os.path.basename(path), "sha256": adapter_hash(path)}
                meta_path = os.path.join(path, "meta.json")
                if os.path.isfile(meta_path):
                    with open(meta_path, encoding="utf-8") as source:
                        previous = json.load(source).get("foundation", {})
                    foundation.update({key: value for key, value in previous.items() if key in {"revision", "base_sha256", "base_model"}})
                model = load_foundation(self.model_name, self.dtype, foundation, os.path.dirname(path))
                peft = False
            else:
                path = adapter_path(self.data_dir, version)
                root = os.path.join(adapters_dir(self.data_dir), version)
                meta_path = os.path.join(root, "meta.json")
                with open(meta_path, encoding="utf-8") as source:
                    meta = json.load(source)
                if meta.get("base_model", self.model_name) != self.model_name:
                    raise ValueError("This adapter belongs to a different base model.")
                foundation = meta.get("foundation")
                model = load_foundation(self.model_name, self.dtype, foundation, root)
                model = load_adapter(model, path)
                peft = True
            model.eval()
            revision = (foundation or {}).get("revision") or getattr(model.config, "_commit_hash", None)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=revision)
            self.tok = tokenizer
            self.model, self.peft, self.version = model, peft, version
            detected = getattr(model.config, "max_position_embeddings", self.configured_max_context)
            self.max_context = min(self.configured_max_context, detected) if isinstance(detected, int) and detected > 0 else self.configured_max_context

    def render(self, messages: list[dict], prefill: str = "", prompt_budget: int | None = None) -> str:
        messages = truncate_messages(self.tok, messages, prompt_budget or self.max_context)
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text + prefill

    def stream(self, messages, prefill="", max_tokens=300, temperature=0.7, top_p=0.9, top_k=0,
               repetition_penalty=1.08, seed=None, stop=None, cancel: threading.Event | None = None, info=None):
        cancel = cancel or threading.Event()
        info = info if info is not None else {}
        stops = [value for value in (stop or []) if value]
        with self.lock:
            model, tokenizer, version, max_context = self.model, self.tok, self.version, self.max_context
        prefill_length = len(tokenizer(prefill, add_special_tokens=False, return_token_type_ids=False).input_ids) if prefill else 0
        prompt_budget = max_context - int(max_tokens) - prefill_length
        if prompt_budget < 1:
            raise ValueError("The requested reply exceeds this model's context limit. Choose a shorter reply.")
        rendered_messages = truncate_messages(tokenizer, messages, prompt_budget)
        text = tokenizer.apply_chat_template(rendered_messages, tokenize=False, add_generation_prompt=True) + prefill
        ids = tokenizer(text, return_tensors="pt", return_token_type_ids=False, add_special_tokens=False)
        if ids.input_ids.shape[1] + int(max_tokens) > max_context:
            raise ValueError("The conversation and reply exceed the context limit. Shorten the conversation or reply.")
        info["prompt_tokens"] = int(ids.input_ids.shape[1])
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=600)
        kwargs = dict(**ids, max_new_tokens=int(max_tokens), streamer=streamer,
                      stopping_criteria=StoppingCriteriaList([_Cancel(cancel)]),
                      repetition_penalty=float(repetition_penalty), pad_token_id=tokenizer.eos_token_id)
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=float(temperature), top_p=float(top_p or 1.0), top_k=int(top_k or 0))
        else:
            kwargs.update(do_sample=False)
        errors = []

        def run():
            with self.lock:
                info.update(adapter=version, t_start=time.time())
                try:
                    if cancel.is_set():
                        info["completion_tokens"] = 0
                        streamer.end()
                        return
                    if seed is not None and seed >= 0:
                        torch.manual_seed(int(seed))
                    with torch.inference_mode():
                        output = model.generate(**kwargs)
                    info["completion_tokens"] = int(output.shape[1] - ids.input_ids.shape[1])
                except Exception as error:
                    errors.append(error)
                    streamer.end()

        started = time.time()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        pending, first, completed = "", None, False
        tail_length = max((len(value) - 1 for value in stops), default=0)
        try:
            for delta in streamer:
                if not delta:
                    continue
                first = first or time.time()
                pending += delta
                hit = min((pending.find(value) for value in stops if value in pending), default=-1)
                if hit >= 0:
                    cancel.set()
                    if pending[:hit]:
                        yield pending[:hit]
                    pending = ""
                    break
                safe_length = max(0, len(pending) - tail_length)
                if safe_length:
                    yield pending[:safe_length]
                    pending = pending[safe_length:]
            if pending and not cancel.is_set():
                yield pending
            completed = True
        finally:
            if not completed:
                cancel.set()
            thread.join()
            info["latency_s"] = time.time() - started
            info["first_token_s"] = first - started if first else None
            info["queue_s"] = info.get("t_start", started) - started
            info["completed"] = completed and not errors
        if errors:
            raise errors[0]

    def generate(self, messages, prefill="", info=None, **kwargs):
        info = info if info is not None else {}
        text = "".join(self.stream(messages, prefill, info=info, **kwargs))
        return text, info

    @torch.inference_mode()
    def response_nll(self, messages, response: str) -> float:
        with self.lock:
            prompt = self.render(messages)
            prompt_tokens = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
            response_ids = self.tok(response, add_special_tokens=False).input_ids
            if self.tok.eos_token_id is not None:
                response_ids.append(self.tok.eos_token_id)
            response_tokens = torch.tensor([response_ids], dtype=torch.long)
            inputs = torch.cat([prompt_tokens, response_tokens], 1)
            logits = self.model(inputs).logits
        log_probs = torch.log_softmax(logits[0, prompt_tokens.shape[1] - 1:-1].float(), -1)
        return float(-log_probs.gather(1, response_tokens[0].unsqueeze(1)).mean())
