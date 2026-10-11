"""CPU LoRA: teach Qwen2.5-0.5B Chinese multi-character narration.

    python data/rp-narration/build_train.py
    python -m rp.train_narration [--out rp/bundled_adapters/narration]

Writes adapter weights + meta.json. Does not touch chat sleep adapters (vN).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import sys
import time

import torch
from . import DEFAULT_MODEL
from .engine import prompt_ids, truncate_messages
from .sleep import encode, log, seq_logps


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(args) -> dict:
    t0 = time.time()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    examples = load_jsonl(args.data)
    if len(examples) < args.min_examples:
        log(f"need at least {args.min_examples} examples, got {len(examples)}")
        return {"status": "failed", "reason": "not enough data"}

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()

    for e in examples:
        e["enc"] = encode(tok, e["messages"], e["response"], args.max_prompt_tokens)

    cfg = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    pm = get_peft_model(model, cfg)
    params = [p for p in pm.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total = args.epochs * math.ceil(len(examples) / args.accum)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))
    )

    pm.train()
    step = 0
    for ep in range(args.epochs):
        order = list(range(len(examples)))
        random.shuffle(order)
        losses = []
        for j, idx in enumerate(order):
            e = examples[idx]
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _, nll = seq_logps(pm, [e["enc"]], pad)
                loss = nll[0]
            if not torch.isfinite(loss):
                continue
            (loss / args.accum).backward()
            losses.append(float(loss))
            if (j + 1) % args.accum == 0 or j == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
            if time.time() - t0 > args.max_minutes * 60:
                log(f"stopped at time budget {args.max_minutes} min")
                break
        log(f"epoch {ep + 1}/{args.epochs} mean_nll={sum(losses) / max(1, len(losses)):.3f} ({time.time() - t0:.0f}s)")

    pm.eval()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        after = []
        for e in examples:
            _, nll = seq_logps(pm, [e["enc"]], pad)
            after.append(float(nll[0]))

    os.makedirs(args.out, exist_ok=True)
    pm.save_pretrained(args.out)
    meta = {
        "kind": "narration",
        "created": time.time(),
        "base_model": args.model,
        "data": os.path.abspath(args.data),
        "examples": len(examples),
        "epochs": args.epochs,
        "rank": args.rank,
        "lr": args.lr,
        "train_minutes": round((time.time() - t0) / 60, 2),
        "train_nll_mean": round(sum(after) / len(after), 4),
        "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    log(f"saved narration adapter -> {args.out} ({meta['train_minutes']} min, RSS ~{meta['max_rss_gb']} GB)")
    return {"status": "ok", "out": args.out, **meta}


def parser():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(here, "data", "rp-narration", "train.jsonl"))
    ap.add_argument("--out", default=os.path.join(here, "rp", "bundled_adapters", "narration"))
    ap.add_argument("--model", default=os.environ.get("RP_MODEL", DEFAULT_MODEL))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--max-prompt-tokens", type=int, default=768)
    ap.add_argument("--max-minutes", type=float, default=120)
    ap.add_argument("--min-examples", type=int, default=8)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap


if __name__ == "__main__":
    res = run(parser().parse_args())
    print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
    sys.exit(0 if res.get("status") == "ok" else 1)
