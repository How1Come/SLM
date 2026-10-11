"""Checkpointed CPU LoRA training for Chinese multi-character narration."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import threading
import time

import torch

from . import DEFAULT_MODEL
from .engine import load_adapter, load_foundation, local_model_hash
from .sleep import TARGET_MODULES, TrainingJournal, atomic_json, encode, fingerprint, log, make_optimizer, read_checkpoint, restore_args, seq_logps, software_versions, training_lock, validate_args


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {line_number} is not valid JSON.") from error
            messages = item.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Line {line_number} needs a messages list.")
            if "response" in item:
                prompt, response = messages, item["response"]
            elif messages[-1].get("role") == "assistant":
                prompt, response = messages[:-1], messages[-1].get("content")
            else:
                raise ValueError(f"Line {line_number} is missing the final assistant response.")
            if not prompt or not any(message.get("role") == "user" for message in prompt):
                raise ValueError(f"Line {line_number} needs a user message before its response.")
            if any(message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str)
                   or not message["content"].strip() for message in prompt):
                raise ValueError(f"Line {line_number} contains an empty message or unsupported role.")
            if not isinstance(response, str) or not response.strip():
                raise ValueError(f"Line {line_number} has an empty response.")
            rows.append({"messages": prompt, "response": response})
    return rows


def _run(args, stop_event: threading.Event, state: dict | None = None) -> dict:
    validate_args(args)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    examples = state["examples"] if state else load_jsonl(args.data)
    if len(examples) < args.min_examples:
        raise ValueError(f"At least {args.min_examples} reviewed examples are required; this file contains {len(examples)}.")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    foundation = state["foundation"] if state else {"kind": "base", "base_model": args.model}
    model_digest = local_model_hash(args.model) if not state else None
    if model_digest:
        foundation["base_sha256"] = model_digest
    model = load_foundation(args.model, torch.float32, foundation)
    if getattr(model.config, "_commit_hash", None):
        foundation["revision"] = model.config._commit_hash
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=foundation.get("revision"))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if not state:
        for item in examples:
            item["enc"] = encode(tokenizer, item["messages"], item["response"], args.max_prompt_tokens, args.max_response_tokens)
    if state:
        model = load_adapter(model, args.resume, trainable=True)
    elif args.init_adapter:
        if not os.path.isfile(os.path.join(args.init_adapter, "adapter_config.json")):
            raise FileNotFoundError("The requested initial adapter has no configuration; choose an existing adapter or remove --init-adapter.")
        model = load_adapter(model, args.init_adapter, trainable=True)
    else:
        config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                            task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
        model = get_peft_model(model, config)
    parameters, optimizer, scheduler, total = make_optimizer(model, args, len(examples))
    journal = TrainingJournal("narration", args.checkpoint_dir or args.out + "-checkpoints", vars(args).copy(), examples, state)
    journal.state["foundation"] = foundation

    def loss_fn(item):
        _, losses = seq_logps(model, [item["enc"]], pad)
        loss = losses[0]
        return loss, {"sft": float(loss.detach())}, sum(len(tokens) for tokens in item["enc"])

    if not journal.train(model, parameters, optimizer, scheduler, total, args, loss_fn, stop_event):
        return {"status": "paused", "reason": "Narration training paused safely; resume its checkpoint to continue.",
                "checkpoint": journal.checkpoint, "step": journal.state["step"], "total_steps": total, "out": args.out}
    model.eval()
    losses = []
    with torch.inference_mode():
        for item in examples[:64]:
            _, loss = seq_logps(model, [item["enc"]], pad)
            value = float(loss[0])
            if not math.isfinite(value):
                raise FloatingPointError("Narration evaluation produced non-finite loss; existing weights were preserved.")
            losses.append(value)
    if stop_event.is_set():
        return {"status": "paused", "reason": "Narration training is checkpointed and ready to finish on resume.",
                "checkpoint": journal.checkpoint, "step": journal.state["step"], "total_steps": total, "out": args.out}
    staging = args.out + ".tmp-" + str(time.time_ns())
    model.save_pretrained(staging, safe_serialization=True)
    rank = next(iter(model.peft_config.values())).r
    metadata = {"kind": "narration", "created": time.time(), "base_model": args.model, "data": args.data,
                "init_adapter": args.init_adapter, "examples": len(examples), "epochs": args.epochs, "rank": rank,
                "lr": args.lr, "train_minutes": round(journal.state["elapsed_s"] / 60, 2),
                "train_nll_mean": round(sum(losses) / len(losses), 4),
                "epoch_nll_means": [round(value, 4) for value in journal.state["epoch_means"]],
                "optimizer_steps": journal.state["step"], "checkpoint": journal.checkpoint,
                "dataset_sha256": fingerprint(examples), "foundation": foundation,
                "config": vars(args).copy(), "software": software_versions()}
    manifest = os.path.join(os.path.dirname(args.data), "manifest.json")
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as source:
            metadata["corpus_manifest"] = json.load(source)
    atomic_json(os.path.join(staging, "meta.json"), metadata)
    if os.path.exists(args.out):
        backup = args.out + ".previous-" + str(time.time_ns())
        os.rename(args.out, backup)
        try:
            os.rename(staging, args.out)
        except Exception:
            os.rename(backup, args.out)
            raise
        log(f"Previous narration weights are preserved at {backup}.")
    else:
        os.rename(staging, args.out)
    log(f"Saved narration adapter to {args.out}; mean NLL {metadata['train_nll_mean']}.")
    return {"status": "ok", "out": args.out, **metadata}


def run(args, stop_event: threading.Event | None = None) -> dict:
    state = read_checkpoint(args.resume, "narration") if args.resume else None
    if state:
        restore_args(args, state)
    with training_lock(args.out + ".training.lock"):
        return _run(args, stop_event or threading.Event(), state)


def parser():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--data", default=os.path.join(root, "data", "rp-narration", "train.jsonl"))
    arguments.add_argument("--init-adapter", default=None)
    arguments.add_argument("--out", default=os.path.join(root, "rp", "bundled_adapters", "narration"))
    arguments.add_argument("--model", default=os.environ.get("RP_MODEL", DEFAULT_MODEL))
    arguments.add_argument("--epochs", type=int, default=6)
    arguments.add_argument("--lr", type=float, default=3e-4)
    arguments.add_argument("--rank", type=int, default=8)
    arguments.add_argument("--accum", type=int, default=1)
    arguments.add_argument("--max-prompt-tokens", type=int, default=768)
    arguments.add_argument("--max-response-tokens", type=int, default=320)
    arguments.add_argument("--max-minutes", type=float, default=120)
    arguments.add_argument("--min-examples", type=int, default=8)
    arguments.add_argument("--threads", type=int, default=None)
    arguments.add_argument("--seed", type=int, default=0)
    arguments.add_argument("--checkpoint-every", type=int, default=10)
    arguments.add_argument("--checkpoint-dir", default=None)
    arguments.add_argument("--resume", default=None)
    return arguments


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
    return 0 if result["status"] in {"ok", "paused"} else 1


if __name__ == "__main__":
    sys.exit(main())
