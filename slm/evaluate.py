"""Evaluation: held-out perplexity, Zorro grammaticality, and generation quality (optionally judged by TinyStories-33M)."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .caregiver import repetition_rate
from .data import Packer, load_prepared, load_tokenizer
from .model import ModelConfig, TinyLM


@torch.no_grad()
def eval_loss(model: TinyLM, docs, eos_id: int, seq_len: int = 256, max_windows: int | None = None, batch_size: int = 16) -> float:
    model.eval()
    p = Packer(docs, eos_id, shuffle=False)
    total = sum(len(d) + 1 for d in docs)
    n = total // (seq_len + 1)
    if max_windows:
        n = min(n, max_windows)
    losses = []
    for i in range(0, n, batch_size):
        b = p.batch(min(batch_size, n - i), seq_len)
        logits = model(b[:, :-1])
        losses.append(F.cross_entropy(logits.flatten(0, 1), b[:, 1:].flatten(), reduction="sum").item())
    model.train()
    return sum(losses) / (n * seq_len)


@torch.no_grad()
def sentence_logprobs(model: TinyLM, seqs: list[list[int]], eos_id: int, batch_size: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = [[eos_id] + s for s in seqs[i : i + batch_size]]
        L = max(map(len, chunk))
        x = torch.full((len(chunk), L), eos_id, dtype=torch.long)
        mask = torch.zeros(len(chunk), L - 1)
        for j, s in enumerate(chunk):
            x[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s) - 1] = 1
        logp = F.log_softmax(model(x[:, :-1]), -1).gather(-1, x[:, 1:, None]).squeeze(-1)
        out.append((logp * mask).sum(1).numpy())
    model.train()
    return np.concatenate(out)


def zorro_accuracy(model: TinyLM, zorro: dict, eos_id: int, per_paradigm: int = 400) -> dict:
    res = {}
    for name, pairs in zorro.items():
        pairs = pairs[:per_paradigm]
        good = sentence_logprobs(model, [g for g, _ in pairs], eos_id)
        bad = sentence_logprobs(model, [b for _, b in pairs], eos_id)
        res[name] = float((good > bad).mean())
    res["_mean"] = float(np.mean(list(res.values())))
    return res


def story_prompts(data: dict, n: int = 100, prompt_len: int = 12) -> list[list[int]]:
    return [d[:prompt_len] for d in data["stories_val"][:n]]


def generate_texts(model: TinyLM, tok, prompts, eos_id: int, max_new: int = 96, seed: int = 1234, temperature: float = 0.8) -> list[str]:
    torch.manual_seed(seed)
    outs, _ = model.generate(prompts, max_new_tokens=max_new, temperature=temperature, top_k=40, stop_ids=(eos_id,))
    return [tok.decode(p + [t for t in o if t != eos_id]) for p, o in zip(prompts, outs)], outs


def distinct_n(seqs: list[list[int]], n: int = 2) -> float:
    grams = [tuple(s[i : i + n]) for s in seqs for i in range(len(s) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def truecase(text: str) -> str:
    text = re.sub(r"(^|[.!?]\s+|\"\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return re.sub(r"\bi\b", "I", text)


class Judge:
    """Independent fluency judge: perplexity of a continuation under TinyStories-33M (needs `transformers`)."""

    def __init__(self, name: str = "roneneldan/TinyStories-33M"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name).eval()

    @torch.no_grad()
    def ppl(self, prompts: list[str], full_texts: list[str]) -> float:
        nll, count = 0.0, 0
        for p, t in zip(prompts, full_texts):
            p_ids = self.tok(truecase(p))["input_ids"]
            ids = self.tok(truecase(t))["input_ids"]
            if len(ids) <= len(p_ids):
                continue
            x = torch.tensor([ids])
            logp = F.log_softmax(self.model(x).logits[0, :-1], -1)
            tgt = x[0, 1:]
            lp = logp.gather(-1, tgt[:, None]).squeeze(-1)[len(p_ids) - 1 :]
            nll -= lp.sum().item()
            count += lp.numel()
        return math.exp(nll / max(count, 1))


def load_model(path: str | Path) -> TinyLM:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyLM(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.eval()


def evaluate(path: str | Path, judge: Judge | None = None, n_gen: int = 100) -> dict:
    data = load_prepared()
    tok = load_tokenizer()
    eos = data["eos_id"]
    model = load_model(path)
    res = {
        "stories_val_loss": eval_loss(model, data["stories_val"], eos),
        "childes_val_loss": eval_loss(model, data["childes_val"], eos),
    }
    res["stories_val_ppl"] = math.exp(res["stories_val_loss"])
    res["childes_val_ppl"] = math.exp(res["childes_val_loss"])
    z = zorro_accuracy(model, data["zorro"], eos)
    res["zorro_mean"] = z.pop("_mean")
    res["zorro"] = z
    prompts = story_prompts(data, n_gen)
    texts, outs = generate_texts(model, tok, prompts, eos)
    res["gen_distinct2"] = distinct_n(outs, 2)
    res["gen_repetition3"] = float(np.mean([repetition_rate(o) for o in outs]))
    res["gen_mean_len"] = float(np.mean([len(o) for o in outs]))
    if judge is not None:
        res["gen_judge_ppl"] = judge.ppl([tok.decode(p) for p in prompts], texts)
    res["samples"] = texts[:8]
    return res


def main():
    ap = argparse.ArgumentParser(description="Evaluate one or more checkpoints")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--judge", action="store_true", help="score generations with TinyStories-33M (needs transformers)")
    ap.add_argument("--out", default="runs/results.json")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    judge = Judge() if args.judge else None
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for ck in args.checkpoints:
        name = Path(ck).parent.name
        print(f"[eval] {name}")
        results[name] = evaluate(ck, judge)
        r = results[name]
        print(
            f"  stories ppl {r['stories_val_ppl']:.2f} | childes ppl {r['childes_val_ppl']:.2f} | zorro {r['zorro_mean']:.3f}"
            f" | distinct2 {r['gen_distinct2']:.3f} | rep3 {r['gen_repetition3']:.3f}"
            + (f" | judge ppl {r['gen_judge_ppl']:.2f}" if "gen_judge_ppl" in r else "")
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
