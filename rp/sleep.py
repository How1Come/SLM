"""Resume-safe CPU consolidation of reviewed chat feedback into LoRA adapters."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import signal
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import DEFAULT_DATA_DIR, DEFAULT_MODEL
from .engine import adapters_dir, load_adapter, load_foundation, prompt_ids, read_current, snapshot_foundation, truncate_messages, write_current
from .learn import norm
from .store import Store

CANARY_PROMPTS = ["请用两三句话介绍一下长城。", "一年有几个季节？分别是什么？", "帮我写一句祝朋友生日快乐的话。",
                  "水在多少度会沸腾？", "推荐一本适合初学者的编程书，并说明理由。", "把“我今天很开心”翻译成英文。"]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def log(*values) -> None:
    print(time.strftime("[%H:%M:%S]"), *values, flush=True)


def emit_metric(**values) -> None:
    print("METRIC " + json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def software_versions() -> dict:
    versions = {"python": sys.version.split()[0], "torch": torch.__version__}
    for package in ("transformers", "peft", "safetensors"):
        versions[package] = importlib.metadata.version(package)
    return versions


def atomic_json(path: str, value: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    staging = path + ".tmp-" + str(time.time_ns())
    with open(staging, "w", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(staging, path)


@contextlib.contextmanager
def training_lock(path: str):
    import fcntl
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A training run is already using this output. Wait or pause it first.") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def build_examples(signals: list[dict], max_examples: int) -> list[dict]:
    items = {}
    for item in signals:
        messages = json.loads(item["messages"]) if isinstance(item["messages"], str) else item["messages"]
        chosen, rejected = item["chosen"], item["rejected"]
        if item["kind"] == "swipe" and chosen is None:
            continue
        if chosen is not None and rejected is not None and norm(chosen) == norm(rejected):
            rejected = None
        if chosen is None and rejected is None:
            continue
        key = (item["turn_key"], norm(chosen) if chosen is not None else None, norm(rejected) if rejected is not None else None)
        previous = items.get(key)
        if previous is None or item["weight"] > previous["weight"]:
            items[key] = {"messages": messages, "chosen": chosen, "rejected": rejected, "weight": item["weight"],
                          "kind": item["kind"], "ids": [item["id"]] + (previous["ids"] if previous else [])}
        else:
            previous["ids"].append(item["id"])
    seen, output = set(), []
    for item in sorted(items.values(), key=lambda value: -max(value["ids"]))[:max_examples]:
        if item["chosen"] is not None:
            key = (fingerprint(item["messages"]), norm(item["chosen"]))
            item["sft"] = key not in seen
            seen.add(key)
        output.append(item)
    return output[::-1]


def encode(tok, messages: list[dict], response: str, max_prompt: int, max_response: int = 512) -> tuple[list[int], list[int]]:
    if not response or not response.strip():
        raise ValueError("Training replies cannot be empty.")
    messages = truncate_messages(tok, messages, max_prompt)
    prompt = prompt_ids(tok, messages)
    if len(prompt) > max_prompt:
        head = max_prompt // 2
        prompt = prompt[:head] + prompt[-(max_prompt - head):]
    response_tokens = tok(response.strip(), add_special_tokens=False).input_ids
    end = [tok.eos_token_id] if tok.eos_token_id is not None else []
    response_tokens = response_tokens[:max(1, max_response - len(end))] + end
    if not prompt or not response_tokens:
        raise ValueError("The tokenizer produced an empty prompt or response.")
    return prompt, response_tokens


def seq_logps(model, batch: list, pad_id: int):
    if not batch or any(not prompt or not response for prompt, response in batch):
        raise ValueError("Every training item needs prompt and response tokens.")
    length = max(len(prompt) + len(response) for prompt, response in batch)
    inputs = torch.full((len(batch), length), pad_id, dtype=torch.long)
    attention = torch.zeros_like(inputs)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    for index, (prompt, response) in enumerate(batch):
        sequence = prompt + response
        inputs[index, :len(sequence)] = torch.tensor(sequence)
        attention[index, :len(sequence)] = 1
        mask[index, len(prompt):len(sequence)] = True
    logits = model(input_ids=inputs, attention_mask=attention).logits[:, :-1].float()
    targets, selected = inputs[:, 1:], mask[:, 1:]
    log_probs = torch.log_softmax(logits, -1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    sums = log_probs.masked_fill(~selected, 0).sum(1)
    return sums, -sums / selected.sum(1).clamp_min(1)


def canary_set(model, tok, data_dir: str, foundation_id: str = "base") -> list[dict]:
    path = os.path.join(data_dir, "canaries", foundation_id + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as source:
            return json.load(source)
    log("Building generic checks for this frozen training foundation.")
    output = []
    for question in CANARY_PROMPTS:
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": question}]
        ids = torch.tensor([prompt_ids(tok, messages)], dtype=torch.long)
        with torch.inference_mode():
            response = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=64,
                                      do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(response[0, ids.shape[1]:], skip_special_tokens=True).strip()
        if text:
            output.append({"messages": messages, "response": text})
    if not output:
        raise ValueError("The foundation generated no usable check responses. Verify the base model and tokenizer.")
    atomic_json(path, output)
    return output


@torch.inference_mode()
def canary_nll(model, tok, canary: list[dict], max_prompt: int, max_response: int = 512) -> float:
    values = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for item in canary:
        tokens = encode(tok, item["messages"], item["response"], max_prompt, max_response)
        _, loss = seq_logps(model, [tokens], pad)
        values.append(float(loss[0]))
    return sum(values) / len(values)


def next_version(data_dir: str) -> str:
    directory = adapters_dir(data_dir)
    numbers = [int(name[1:]) for name in os.listdir(directory) if re.fullmatch(r"v[1-9][0-9]*", name)] if os.path.isdir(directory) else []
    return f"v{max(numbers, default=0) + 1}"


def validate_args(args) -> None:
    for key in ("epochs", "rank", "accum", "max_prompt_tokens", "max_response_tokens", "max_examples", "min_examples", "checkpoint_every"):
        value = getattr(args, key, None)
        if value is not None and value < 1:
            raise ValueError(key.replace("_", " ") + " must be positive.")
    if not math.isfinite(args.max_minutes) or args.max_minutes <= 0 or not math.isfinite(args.lr) or not 0 < args.lr <= 0.01:
        raise ValueError("Use a positive time budget and a learning rate between 0 and 0.01.")
    if args.threads is not None and args.threads < 1:
        raise ValueError("CPU threads must be positive.")
    preference_values = [getattr(args, key, default) for key, default in (("beta", .1), ("dpo_weight", .5), ("neg_weight", .5))]
    if not all(math.isfinite(value) for value in preference_values) or preference_values[0] <= 0 or any(value < 0 for value in preference_values[1:]):
        raise ValueError("Preference beta must be positive and preference loss weights cannot be negative.")
    if not math.isfinite(getattr(args, "guard_max_drift", 0.5)) or getattr(args, "guard_max_drift", 0.5) < 0:
        raise ValueError("The drift limit cannot be negative.")


def make_optimizer(model, args, item_count: int):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("This adapter has no trainable parameters.")
    for parameter in parameters:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    total = args.epochs * math.ceil(item_count / args.accum)
    warm = max(1, int(0.1 * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, total)))))
    return parameters, optimizer, scheduler, total


def read_checkpoint(path: str, kind: str) -> dict:
    root = Path(path).resolve()
    with (root / "checkpoint.json").open(encoding="utf-8") as source:
        metadata = json.load(source)
    if metadata.get("kind") != kind or metadata.get("format_version") != 1:
        raise ValueError("This checkpoint belongs to a different training workflow or unsupported version.")
    state = torch.load(root / "state.pt", map_location="cpu", weights_only=True)
    if fingerprint(state["examples"]) != metadata["dataset_sha256"]:
        raise ValueError("The checkpoint dataset snapshot was modified.")
    state["checkpoint"] = str(root)
    return state


def restore_args(args, state: dict) -> None:
    runtime = {key: getattr(args, key, None) for key in ("threads", "max_minutes", "server", "no_activate", "debug", "resume") if hasattr(args, key)}
    for key, value in state["config"].items():
        setattr(args, key, value)
    for key, value in runtime.items():
        setattr(args, key, value)
    if "no_activate" in runtime:
        args.no_activate = bool(runtime["no_activate"] or state["config"].get("no_activate", False))


class TrainingJournal:
    def __init__(self, kind: str, root: str, config: dict, examples: list[dict], state: dict | None = None):
        self.kind = kind
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.state = state or {"config": config, "examples": examples, "epoch": 0, "cursor": 0,
                               "order": [], "step": 0, "elapsed_s": 0.0, "losses": [], "epoch_means": [], "history": []}
        self.checkpoint = self.state.get("checkpoint")

    def restore(self, optimizer, scheduler) -> None:
        if "optimizer" not in self.state:
            return
        optimizer.load_state_dict(self.state["optimizer"])
        scheduler.load_state_dict(self.state["scheduler"])
        random.setstate(self.state["random_state"])
        torch.set_rng_state(self.state["torch_rng"])

    def save(self, model, optimizer, scheduler, elapsed_s: float, status: str = "running") -> str:
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters() if parameter.requires_grad):
            raise FloatingPointError("Adapter weights became non-finite. Resume the previous checkpoint with a lower learning rate.")
        name = f"step-{self.state['step']:06d}-{time.time_ns()}"
        directory = os.path.join(self.root, name)
        staging = directory + ".tmp"
        os.makedirs(staging)
        model.save_pretrained(staging, safe_serialization=True)
        self.state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          random_state=random.getstate(), torch_rng=torch.get_rng_state(), elapsed_s=elapsed_s)
        torch.save(self.state, os.path.join(staging, "state.pt"))
        metadata = {"format_version": 1, "kind": self.kind, "status": status, "created": time.time(),
                    "step": self.state["step"], "epoch": self.state["epoch"], "cursor": self.state["cursor"],
                    "dataset_sha256": fingerprint(self.state["examples"]), "config": self.state["config"],
                    "software": software_versions()}
        atomic_json(os.path.join(staging, "checkpoint.json"), metadata)
        os.rename(staging, directory)
        self.checkpoint = directory
        atomic_json(os.path.join(self.root, "latest.json"), {"checkpoint": directory, **metadata})
        emit_metric(event="checkpoint", checkpoint=directory, step=self.state["step"], status=status)
        return directory

    def train(self, model, parameters, optimizer, scheduler, total: int, args, loss_fn,
              stop_event: threading.Event) -> bool:
        started = time.monotonic()
        previous_elapsed = self.state.get("elapsed_s", 0.0)
        self.restore(optimizer, scheduler)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        count = len(self.state["examples"])
        while self.state["epoch"] < args.epochs:
            if not self.state["order"]:
                self.state["order"] = list(range(count))
                random.shuffle(self.state["order"])
            while self.state["cursor"] < count:
                if stop_event.is_set() or time.monotonic() - started >= args.max_minutes * 60:
                    self.save(model, optimizer, scheduler, previous_elapsed + time.monotonic() - started, "paused")
                    return False
                begin = self.state["cursor"]
                group = self.state["order"][begin:begin + args.accum]
                group_loss, tokens = 0.0, 0
                group_started = time.monotonic()
                for index in group:
                    loss, parts, token_count = loss_fn(self.state["examples"][index])
                    if not bool(torch.isfinite(loss)):
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError("Training loss became non-finite. Restore a checkpoint and use a lower learning rate.")
                    if loss.requires_grad:
                        (loss / len(group)).backward()
                    value = float(loss.detach())
                    group_loss += value
                    tokens += token_count
                    self.state["history"].append(parts)
                    self.state["losses"].append(value)
                norm_value = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not bool(torch.isfinite(norm_value)):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Training gradients became non-finite. Restore a checkpoint and use a lower learning rate.")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                self.state["step"] += 1
                self.state["cursor"] += len(group)
                elapsed = previous_elapsed + time.monotonic() - started
                self.state["elapsed_s"] = elapsed
                progress = (self.state["epoch"] + self.state["cursor"] / count) / args.epochs
                emit_metric(step=self.state["step"], total_steps=total, progress=progress,
                            epoch=self.state["epoch"] + self.state["cursor"] / count,
                            loss=group_loss / len(group), learning_rate=optimizer.param_groups[0]["lr"],
                            elapsed_s=elapsed, eta_s=elapsed / progress * (1 - progress) if progress else None,
                            tokens_per_s=tokens / max(0.001, time.monotonic() - group_started),
                            checkpoint=self.checkpoint)
                if self.state["step"] % args.checkpoint_every == 0:
                    self.save(model, optimizer, scheduler, elapsed)
            mean = sum(self.state["losses"]) / max(1, len(self.state["losses"]))
            self.state["epoch_means"].append(mean)
            self.state["epoch"] += 1
            self.state["cursor"], self.state["order"], self.state["losses"] = 0, [], []
            log(f"Epoch {self.state['epoch']}/{args.epochs}, loss {mean:.4f}, step {self.state['step']}/{total}")
        self.save(model, optimizer, scheduler, previous_elapsed + time.monotonic() - started, "completed")
        return True


def _run(args, stop_event: threading.Event, state: dict | None = None) -> dict:
    validate_args(args)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    store = Store(args.data_dir)
    signals = store.active_signals() if state is None else state["signals"]
    examples = build_examples(signals, args.max_examples) if state is None else state["examples"]
    positive = sum(item["chosen"] is not None for item in examples)
    if positive < args.min_examples:
        return {"status": "skipped", "reason": "Add or approve at least one preferred reply before training.", "examples": len(examples)}
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    version = state["version"] if state else next_version(args.data_dir)
    output = os.path.join(adapters_dir(args.data_dir), version)
    foundation = state["foundation"] if state else snapshot_foundation(args.data_dir, output, args.model, args.foundation)
    model = load_foundation(args.model, torch.float32, foundation, output)
    if getattr(model.config, "_commit_hash", None):
        foundation["revision"] = model.config._commit_hash
    tok = AutoTokenizer.from_pretrained(args.model, revision=foundation.get("revision"))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    kinds = {kind: sum(item["kind"] == kind for item in examples) for kind in sorted({item["kind"] for item in examples})}
    if state:
        canary, canary_base = state["canary"], state["canary_base"]
        model = load_adapter(model, args.resume, trainable=True)
    else:
        canary = canary_set(model, tok, args.data_dir, fingerprint(foundation))
        canary_base = canary_nll(model, tok, canary, args.max_prompt_tokens, args.max_response_tokens)
        for item in examples:
            item["c"] = encode(tok, item["messages"], item["chosen"], args.max_prompt_tokens, args.max_response_tokens) if item["chosen"] is not None else None
            item["r"] = encode(tok, item["messages"], item["rejected"], args.max_prompt_tokens, args.max_response_tokens) if item["rejected"] is not None else None
        with torch.inference_mode():
            for item in examples:
                sequences = [value for value in (item["c"], item["r"]) if value is not None]
                probabilities, losses = seq_logps(model, sequences, pad)
                if item["c"] is not None:
                    item["ref_c"], item["nll_c0"] = float(probabilities[0]), float(losses[0])
                if item["r"] is not None:
                    item["ref_r"] = float(probabilities[-1])
        config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
        model = get_peft_model(model, config)
    parameters, optimizer, scheduler, total = make_optimizer(model, args, len(examples))
    journal = TrainingJournal("sleep", args.checkpoint_dir or os.path.join(args.data_dir, "checkpoints", "sleep"), vars(args).copy(), examples, state)
    journal.state.update(version=version, foundation=foundation, signals=signals, canary=canary, canary_base=canary_base,
                         previous=state["previous"] if state else read_current(args.data_dir, args.model))

    def loss_fn(item):
        sequences = [value for value in (item["c"], item["r"]) if value is not None]
        probabilities, losses = seq_logps(model, sequences, pad)
        loss, parts = torch.zeros(()), {}
        if item["c"] is not None and item.get("sft", True):
            loss = loss + losses[0]
            parts["sft"] = float(losses[0].detach())
        if item["c"] is not None and item["r"] is not None:
            margin = (probabilities[0] - item["ref_c"]) - (probabilities[1] - item["ref_r"])
            preference = -F.logsigmoid(args.beta * margin)
            loss = loss + args.dpo_weight * preference
            parts["dpo"] = float(preference.detach())
        elif item["r"] is not None:
            negative = -F.logsigmoid(-args.beta * (probabilities[-1] - item["ref_r"]))
            loss = loss + args.neg_weight * negative
            parts["neg"] = float(negative.detach())
        if args.debug:
            log(f"Feedback {item['ids'][0]}, {item['kind']}, token lengths {[len(prompt) + len(response) for prompt, response in sequences]}, losses {parts}")
        return loss * item["weight"], parts, sum(len(prompt) + len(response) for prompt, response in sequences)

    completed = journal.train(model, parameters, optimizer, scheduler, total, args, loss_fn, stop_event)
    if not completed:
        return {"status": "paused", "reason": "Paused safely; resume this checkpoint to continue.", "version": version,
                "checkpoint": journal.checkpoint, "step": journal.state["step"], "total_steps": total, "activated": False}
    if stop_event.is_set():
        return {"status": "paused", "reason": "Training is checkpointed and ready to finish on resume.", "version": version,
                "checkpoint": journal.checkpoint, "step": journal.state["step"], "total_steps": total, "activated": False}
    model.eval()
    after, wins = [], []
    with torch.inference_mode():
        for item in examples:
            sequences = [value for value in (item["c"], item["r"]) if value is not None]
            probabilities, losses = seq_logps(model, sequences, pad)
            if item["c"] is not None:
                after.append((item["nll_c0"], float(losses[0])))
            if item["c"] is not None and item["r"] is not None:
                wins.append(float((probabilities[0] - item["ref_c"]) - (probabilities[1] - item["ref_r"])) > 0)
    canary_after = canary_nll(model, tok, canary, args.max_prompt_tokens, args.max_response_tokens)
    drift = canary_after - canary_base
    passed = math.isfinite(drift) and drift <= args.guard_max_drift
    if not math.isfinite(canary_after) or any(not math.isfinite(value) for pair in after for value in pair):
        raise FloatingPointError("Evaluation produced non-finite loss; this adapter was not activated.")
    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    signal_ids = sorted({identifier for item in examples for identifier in item["ids"]})
    snapshot = {item["id"]: item for item in signals}
    changed = False
    for identifier in signal_ids:
        current = store.get_signal(identifier)
        previous_signal = snapshot[identifier]
        if not current or not current["active"] or current.get("record_version", 0) != previous_signal.get("record_version", 0):
            changed = True
            break
    metadata = {"created": time.time(), "base_model": args.model, "foundation": foundation,
                "signals": len(signals), "items": len(examples), "kinds": kinds, "epochs_run": journal.state["epoch"],
                "optimizer_steps": journal.state["step"], "lr": args.lr, "rank": args.rank,
                "train_minutes": round(journal.state["elapsed_s"] / 60, 2), "config": vars(args).copy(),
                "dataset_sha256": fingerprint(examples), "software": software_versions(), "seed": args.seed,
                "train_nll_chosen_before": round(sum(value for value, _ in after) / max(1, len(after)), 4),
                "train_nll_chosen_after": round(sum(value for _, value in after) / max(1, len(after)), 4),
                "pair_pref_acc_after": round(sum(wins) / len(wins), 3) if wins else None,
                "canary_nll_base": round(canary_base, 4), "canary_nll_after": round(canary_after, 4),
                "canary_drift": round(drift, 4), "guard_max_drift": args.guard_max_drift, "guard_passed": passed,
                "previous": journal.state["previous"], "signal_ids": signal_ids, "checkpoint": journal.checkpoint,
                "dataset_changed_since_start": changed,
                "activation_blocked_reason": "Feedback changed during training. Review this saved adapter or start a fresh run." if changed else None,
                "activated": passed and not args.no_activate and not changed}
    atomic_json(os.path.join(output, "meta.json"), metadata)
    if stop_event.is_set():
        metadata["activated"] = False
        atomic_json(os.path.join(output, "meta.json"), metadata)
        return {"status": "paused", "reason": "Training is checkpointed and ready to finish on resume.", "version": version,
                "checkpoint": journal.checkpoint, "step": journal.state["step"], "total_steps": total, "activated": False}
    if metadata["activated"]:
        write_current(args.data_dir, version)
        expected = {item["id"]: item["record_version"] for item in signals if "record_version" in item}
        store.mark_consumed(signal_ids, version, expected_versions=expected or None)
        if args.server:
            import httpx
            try:
                token = os.environ.get("RP_AUTH_TOKEN")
                headers = {"Authorization": "Bearer " + token} if token else {}
                response = httpx.post(args.server.rstrip("/") + "/rp/reload", headers=headers, timeout=120, trust_env=False)
                response.raise_for_status()
            except httpx.HTTPError:
                log("Weights were saved; the server could not reload. Reload the adapter from the dashboard.")
    log(f"Saved {version}: preferred NLL {metadata['train_nll_chosen_before']} -> {metadata['train_nll_chosen_after']}, guard {'passed' if passed else 'failed'}")
    return {"status": "ok", "version": version, **{key: value for key, value in metadata.items() if key != "signal_ids"}}


def run(args, stop_event: threading.Event | None = None) -> dict:
    state = read_checkpoint(args.resume, "sleep") if args.resume else None
    if state:
        restore_args(args, state)
    with training_lock(os.path.join(args.data_dir, ".sleep.lock")):
        return _run(args, stop_event or threading.Event(), state)


def parser():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    argument_parser.add_argument("--model", default=os.environ.get("RP_MODEL", DEFAULT_MODEL))
    argument_parser.add_argument("--server", default=None)
    argument_parser.add_argument("--epochs", type=int, default=4)
    argument_parser.add_argument("--lr", type=float, default=2e-4)
    argument_parser.add_argument("--rank", type=int, default=16)
    argument_parser.add_argument("--accum", type=int, default=2)
    argument_parser.add_argument("--beta", type=float, default=0.1)
    argument_parser.add_argument("--dpo-weight", type=float, default=0.5)
    argument_parser.add_argument("--neg-weight", type=float, default=0.5)
    argument_parser.add_argument("--max-prompt-tokens", type=int, default=768)
    argument_parser.add_argument("--max-response-tokens", type=int, default=512)
    argument_parser.add_argument("--max-examples", type=int, default=400)
    argument_parser.add_argument("--min-examples", type=int, default=1)
    argument_parser.add_argument("--max-minutes", type=float, default=15)
    argument_parser.add_argument("--guard-max-drift", type=float, default=0.5)
    argument_parser.add_argument("--no-activate", action="store_true")
    argument_parser.add_argument("--threads", type=int, default=None)
    argument_parser.add_argument("--seed", type=int, default=0)
    argument_parser.add_argument("--debug", action="store_true")
    argument_parser.add_argument("--checkpoint-every", type=int, default=10)
    argument_parser.add_argument("--checkpoint-dir", default=None)
    argument_parser.add_argument("--resume", default=None)
    argument_parser.add_argument("--foundation", choices=["auto", "base", "narration"], default="auto")
    return argument_parser


def main() -> int:
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    try:
        result = run(parser().parse_args(), stopped)
    except Exception as error:
        log(f"Training stopped: {error}")
        result = {"status": "failed", "reason": str(error)}
    print("RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] in {"ok", "skipped", "paused"} else 1


if __name__ == "__main__":
    sys.exit(main())
