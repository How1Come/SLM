"""Talk to (or sample from) a trained model.

    python -m slm.chat                         # interactive, uses runs/listen, runs/cradle, or runs/cradle-listen-only-s0
    python -m slm.chat --prompt "once upon a time" -n 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import EOS, load_tokenizer
from .evaluate import load_model


def complete(model, tok, text: str, max_new: int, temperature: float, top_k: int, n: int = 1) -> list[str]:
    eos = tok.token_to_id(EOS)
    ids = tok.encode(text.lower().strip()).ids[-(model.cfg.max_len - max_new) :] or [eos]
    outs, _ = model.generate([ids] * n, max_new_tokens=max_new, temperature=temperature, top_k=top_k, stop_ids=(eos,))
    return [tok.decode([t for t in o if t != eos]) for o in outs]


def main():
    ap = argparse.ArgumentParser(description="Chat with a tiny Cradle model")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--prompt", help="one-shot prompt; omit for interactive mode")
    ap.add_argument("-n", type=int, default=1, help="number of samples per prompt")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.checkpoint is None:
        found = [p for p in ("runs/listen/model.pt", "runs/cradle/model.pt", "runs/cradle-listen-only-s0/model.pt") if Path(p).exists()]
        args.checkpoint = found[0] if found else "runs/listen/model.pt"
    if not Path(args.checkpoint).exists():
        raise SystemExit(f"{args.checkpoint} not found. Train first: python -m slm.train --recipe cradle")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tok = load_model(args.checkpoint), load_tokenizer()
    gen = lambda text: complete(model, tok, text, args.max_new, args.temperature, args.top_k, args.n)

    if args.prompt is not None:
        for s in gen(args.prompt):
            print(f"{args.prompt.lower()}{s}\n")
        return

    print("Type the start of a sentence or story and the model continues it. Ctrl-D to quit.")
    history = ""
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/reset":
            history = ""
            continue
        history = f"{history} {text}".strip()
        reply = gen(history)[0].strip()
        print(f"baby> {reply}")
        history = f"{history} {reply}"


if __name__ == "__main__":
    main()
