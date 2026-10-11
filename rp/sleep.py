"""Sleep: consolidate chat feedback into a LoRA adapter on CPU, then (optionally) hot-reload it.

    python -m rp.sleep [--data-dir rp_data] [--server http://127.0.0.1:8619] [--epochs 4] [--max-minutes 10]

Every sleep trains a *fresh* LoRA on the base model from all active signals (capped to the most recent
--max-examples), so a version depends only on the signals, never on a chain of earlier adapters. Losses:

* SFT (cross-entropy) on every preferred reply: edits / !fix corrections (weight 1), !good (0.5), kept swipes (0.3)
* DPO on (preferred, rejected) pairs: edit vs. original, fix vs. original, kept swipe vs. swiped-away
* unpaired "undesirable" term (KTO-style, reference point 0) on rejected-only replies: !bad, bare swipes
The reference model for DPO/KTO is the base model (LoRA B=0 at init, so the reference is computed once).

Safety: adapters are versioned (rp_data/adapters/vN, never overwritten), `current.json` points at the active
one, and a canary check (NLL of the base model's own answers to generic prompts) blocks activation if the
adapter drifts too far from the base model's general behaviour. Roll back with `python -m rp.admin rollback`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

from . import DEFAULT_DATA_DIR, DEFAULT_MODEL
from .engine import adapters_dir, list_adapters, prompt_ids, read_current, truncate_messages, write_current
from .learn import norm
from .store import Store

CANARY_PROMPTS = [
    "请用两三句话介绍一下长城。",
    "一年有几个季节？分别是什么？",
    "帮我写一句祝朋友生日快乐的话。",
    "水在多少度会沸腾？",
    "推荐一本适合初学者的编程书，并说明理由。",
    "把“我今天很开心”翻译成英文。",
]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def build_examples(signals, max_examples):
    """One training item per (turn, preferred reply); pairs and rejected-only items are kept separately."""
    items = {}
    for s in signals:
        msgs = json.loads(s["messages"])
        chosen, rejected = s["chosen"], s["rejected"]
        if chosen is not None and rejected is not None and norm(chosen) == norm(rejected):
            rejected = None
        if chosen is None and rejected is None:
            continue
        k = (s["turn_key"], norm(chosen) if chosen is not None else None, norm(rejected) if rejected is not None else None)
        prev = items.get(k)
        if prev is None or s["weight"] > prev["weight"]:
            items[k] = {"messages": msgs, "chosen": chosen, "rejected": rejected, "weight": s["weight"],
                        "kind": s["kind"], "ids": [s["id"]] + (prev["ids"] if prev else [])}
        else:
            prev["ids"].append(s["id"])
    # a reply preferred in several pairs gets its SFT term once
    seen = set()
    out = []
    for it in sorted(items.values(), key=lambda x: -x["ids"][0])[:max_examples]:
        if it["chosen"] is not None:
            ck = (json.dumps(it["messages"][-1:], ensure_ascii=False), norm(it["chosen"]))
            it["sft"] = ck not in seen
            seen.add(ck)
        out.append(it)
    return out[::-1]


def encode(tok, messages, response, max_prompt):
    msgs = truncate_messages(tok, messages, max_prompt)
    p = prompt_ids(tok, msgs)
    if len(p) > max_prompt:  # a very long system prompt: keep its head and the end of the prompt
        p = p[: max_prompt // 2] + p[-max_prompt // 2:]
    r = tok(response.strip() + "<|im_end|>", add_special_tokens=False).input_ids
    return p, r


def seq_logps(model, batch, pad_id):
    """batch: list of (prompt_ids, resp_ids). Returns (sum logp per seq, mean token NLL per seq)."""
    L = max(len(p) + len(r) for p, r in batch)
    x = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.bool)
    for i, (p, r) in enumerate(batch):
        seq = p + r
        x[i, : len(seq)] = torch.tensor(seq)
        att[i, : len(seq)] = 1
        mask[i, len(p): len(seq)] = True
    logits = model(input_ids=x, attention_mask=att).logits[:, :-1].float()
    tgt = x[:, 1:]
    m = mask[:, 1:]
    lp = torch.log_softmax(logits, -1).gather(2, tgt.unsqueeze(-1)).squeeze(-1) * m
    s = lp.sum(1)
    return s, -s / m.sum(1)


def canary_set(model, tok, data_dir):
    path = os.path.join(data_dir, "canary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    log("building canary set (base model answers to generic prompts)")
    out = []
    for q in CANARY_PROMPTS:
        msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": q}]
        batch = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = batch if torch.is_tensor(batch) else batch["input_ids"]
        with torch.inference_mode():
            g = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
        out.append({"messages": msgs, "response": tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)})
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


@torch.inference_mode()
def canary_nll(model, tok, canary, max_prompt):
    vals = []
    for c in canary:
        p, r = encode(tok, c["messages"], c["response"], max_prompt)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, nll = seq_logps(model, [(p, r)], tok.pad_token_id or tok.eos_token_id)
        vals.append(float(nll[0]))
    return sum(vals) / len(vals)


def next_version(data_dir):
    nums = [int(a["version"][1:]) for a in list_adapters(data_dir)]
    d = adapters_dir(data_dir)
    if os.path.isdir(d):
        nums += [int(n[1:]) for n in os.listdir(d) if n.startswith("v") and n[1:].isdigit()]
    return f"v{max(nums, default=0) + 1}"


def run(args) -> dict:
    t_start = time.time()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    store = Store(args.data_dir)
    signals = store.active_signals()
    examples = build_examples(signals, args.max_examples)
    n_pos = sum(1 for e in examples if e["chosen"] is not None)
    if n_pos < args.min_examples:
        log(f"nothing to learn yet: {n_pos} preferred replies (< {args.min_examples})")
        return {"status": "skipped", "reason": "not enough signals", "examples": len(examples)}
    kinds = {}
    for e in examples:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    log(f"{len(signals)} active signals -> {len(examples)} training items {kinds}")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    canary = canary_set(model, tok, args.data_dir)
    canary_base = canary_nll(model, tok, canary, args.max_prompt_tokens)

    for e in examples:
        e["c"] = encode(tok, e["messages"], e["chosen"], args.max_prompt_tokens) if e["chosen"] is not None else None
        e["r"] = encode(tok, e["messages"], e["rejected"], args.max_prompt_tokens) if e["rejected"] is not None else None
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        for e in examples:
            seqs = [s for s in (e["c"], e["r"]) if s is not None]
            lp, nll = seq_logps(model, seqs, pad)
            i = 0
            if e["c"] is not None:
                e["ref_c"], e["nll_c0"] = float(lp[0]), float(nll[0]); i = 1
            if e["r"] is not None:
                e["ref_r"] = float(lp[i])
    log("reference log-probs done (%.0fs)" % (time.time() - t_start))

    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    pm = get_peft_model(model, cfg)
    params = [p for p in pm.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.epochs * math.ceil(len(examples) / args.accum)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))

    pm.train()
    step, hist, stopped, skipped = 0, [], False, 0
    for ep in range(args.epochs):
        order = list(range(len(examples)))
        random.shuffle(order)
        for j, idx in enumerate(order):
            e = examples[idx]
            seqs = [s for s in (e["c"], e["r"]) if s is not None]
            with torch.autocast("cpu", dtype=torch.bfloat16):
                lp, nll = seq_logps(pm, seqs, pad)
            loss = torch.zeros(())
            parts = {}
            if e["c"] is not None and e.get("sft", True):
                loss = loss + nll[0]
                parts["sft"] = float(nll[0])
            if e["c"] is not None and e["r"] is not None:
                margin = (lp[0] - e["ref_c"]) - (lp[1] - e["ref_r"])
                dpo = -F.logsigmoid(args.beta * margin)
                loss = loss + args.dpo_weight * dpo
                parts["dpo"] = float(dpo)
            elif e["r"] is not None:
                und = -F.logsigmoid(-args.beta * (lp[-1] - e["ref_r"]))
                loss = loss + args.neg_weight * und
                parts["neg"] = float(und)
            if args.debug:
                log(f"item {idx} {e['kind']} lens {[len(p) + len(r) for p, r in seqs]} {parts}")
            if not torch.isfinite(loss):
                skipped += 1
                continue
            (loss * e["weight"] / args.accum).backward()
            hist.append(parts)
            if (j + 1) % args.accum == 0 or j == len(order) - 1:
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                if torch.isfinite(gn):
                    opt.step()
                else:
                    skipped += 1
                    log(f"non-finite gradient (norm {float(gn)}), step skipped")
                sched.step(); opt.zero_grad()
                step += 1
            if time.time() - t_start > args.max_minutes * 60:
                log(f"time budget of {args.max_minutes} min reached, stopping early")
                stopped = True
                break
        recent = hist[-len(order):]
        avg = {k: round(sum(h[k] for h in recent if k in h) / max(1, sum(1 for h in recent if k in h)), 4)
               for k in ("sft", "dpo", "neg")}
        log(f"epoch {ep + 1}/{args.epochs} step {step}/{total} loss {avg} ({time.time() - t_start:.0f}s)")
        if stopped:
            break

    pm.eval()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        after, wins = [], []
        for e in examples:
            seqs = [s for s in (e["c"], e["r"]) if s is not None]
            lp, nll = seq_logps(pm, seqs, pad)
            if e["c"] is not None:
                after.append((e["nll_c0"], float(nll[0])))
            if e["c"] is not None and e["r"] is not None:
                wins.append(float((lp[0] - e["ref_c"]) - (lp[1] - e["ref_r"])) > 0)
    canary_after = canary_nll(pm, tok, canary, args.max_prompt_tokens)
    drift = canary_after - canary_base
    guard_ok = drift <= args.guard_max_drift
    version = next_version(args.data_dir)
    out_dir = os.path.join(adapters_dir(args.data_dir), version)
    pm.save_pretrained(out_dir)
    meta = {
        "created": time.time(), "base_model": args.model, "signals": len(signals), "items": len(examples),
        "kinds": kinds, "epochs_run": ep + 1, "optimizer_steps": step, "skipped_nonfinite": skipped, "lr": args.lr, "rank": args.rank,
        "train_minutes": round((time.time() - t_start) / 60, 2),
        "train_nll_chosen_before": round(sum(a for a, _ in after) / max(1, len(after)), 4),
        "train_nll_chosen_after": round(sum(b for _, b in after) / max(1, len(after)), 4),
        "pair_pref_acc_after": round(sum(wins) / len(wins), 3) if wins else None,
        "canary_nll_base": round(canary_base, 4), "canary_nll_after": round(canary_after, 4),
        "canary_drift": round(drift, 4), "guard_max_drift": args.guard_max_drift, "guard_passed": guard_ok,
        "previous": read_current(args.data_dir),
        "signal_ids": sorted({i for e in examples for i in e["ids"]}),
    }
    activate = guard_ok and not args.no_activate
    meta["activated"] = activate
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    store.mark_consumed(meta["signal_ids"], version)
    log(f"saved {out_dir}: chosen NLL {meta['train_nll_chosen_before']} -> {meta['train_nll_chosen_after']}, "
        f"pair acc {meta['pair_pref_acc_after']}, canary drift {drift:+.3f} (guard {'ok' if guard_ok else 'FAILED'})")
    if activate:
        write_current(args.data_dir, version)
        log(f"activated {version}")
        if args.server:
            import httpx
            try:
                httpx.post(args.server.rstrip("/") + "/rp/reload", timeout=120)
                log("server reloaded")
            except Exception as ex:
                log(f"could not notify server: {ex}")
    elif not guard_ok:
        log(f"NOT activated: canary drift {drift:.3f} > {args.guard_max_drift}; activate manually if you want it")
    return {"status": "ok", "version": version, **{k: v for k, v in meta.items() if k != "signal_ids"}}


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--model", default=os.environ.get("RP_MODEL", DEFAULT_MODEL))
    ap.add_argument("--server", default=None, help="server base URL to hot-reload after training")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--dpo-weight", type=float, default=0.5)
    ap.add_argument("--neg-weight", type=float, default=0.5)
    ap.add_argument("--max-prompt-tokens", type=int, default=768)
    ap.add_argument("--max-examples", type=int, default=400)
    ap.add_argument("--min-examples", type=int, default=1)
    ap.add_argument("--max-minutes", type=float, default=15)
    ap.add_argument("--guard-max-drift", type=float, default=0.5, help="max canary NLL increase (nats/token)")
    ap.add_argument("--no-activate", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true")
    return ap


if __name__ == "__main__":
    res = run(parser().parse_args())
    print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)
    sys.exit(0 if res["status"] in ("ok", "skipped") else 1)
