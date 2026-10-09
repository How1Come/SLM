"""Training: the baby-inspired "Cradle" recipe and a plain next-token baseline under the same compute budget.

Compute is accounted in token-forward units: a trained token costs 3 (forward + backward), a forward-only
token (sampling, teacher passes) costs 1. Both recipes stop at exactly the same number of units.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .caregiver import Caregiver, NGramStats, Recaster
from .data import Packer, load_prepared, load_tokenizer
from .evaluate import eval_loss
from .model import ModelConfig, TinyLM

DIARY_PROMPTS = ["look at the", "once upon a time, there was a little", "what do you want to"]


@dataclass
class Budget:
    total: float
    used: float = 0.0
    trained_tokens: int = 0
    forward_tokens: int = 0

    def train(self, n: int) -> None:
        self.used += 3 * n
        self.trained_tokens += n

    def forward(self, n: int) -> None:
        self.used += n
        self.forward_tokens += n

    @property
    def progress(self) -> float:
        return min(self.used / self.total, 1.0)


@dataclass
class Args:
    recipe: str = "cradle"
    budget_tokens: float = 12e6
    seed: int = 0
    batch_tokens: int = 8192
    lr: float = 2e-3
    warmup: float = 0.03
    d_model: int = 192
    n_layer: int = 6
    n_head: int = 6
    out: str = "runs"
    name: str = ""
    # Cradle phase fractions (listen-cds, listen-stories, babble, sleep) and switches for ablations.
    phases: tuple = (0.15, 0.45, 0.25, 0.15)
    prior_alpha: float = 0.3
    use_prior: bool = True
    use_curriculum: bool = True
    use_babble: bool = True
    use_sleep: bool = True
    babble_rounds: int = 6
    babble_per_round: int = 1024
    feedback_frac: float = 0.2
    sleep_beta: float = 0.5
    ema_decay: float = 0.998
    eval_every: float = 0.05
    extra: dict = field(default_factory=dict)


class Trainer:
    def __init__(self, args: Args):
        self.args = args
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        self.data = load_prepared()
        self.tok = load_tokenizer()
        self.eos = self.data["eos_id"]
        self.cfg = ModelConfig(
            vocab_size=self.data["vocab_size"], d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head, max_len=256
        )
        self.model = TinyLM(self.cfg)
        decay = [p for n, p in self.model.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in self.model.named_parameters() if p.dim() < 2]
        self.opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95)
        )
        self.budget = Budget(total=3 * args.budget_tokens)
        self.ema: TinyLM | None = None
        self.ngrams: NGramStats | None = None
        self.log = {"args": {k: v for k, v in vars(args).items()}, "curve": [], "phases": [], "babble": [], "diary": []}
        self.next_eval = 0.0
        self.step_count = 0
        self.t0 = time.time()
        self.all_train = self.data["childes_train"] + self.data["stories_train"]

    # ---------------------------------------------------------------- core step
    def lr_now(self) -> float:
        p, w = self.budget.progress, self.args.warmup
        if p < w:
            return self.args.lr * p / w
        return self.args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (p - w) / (1 - w))))

    def step(self, batch: torch.Tensor, prior_alpha: float = 0.0, teacher: TinyLM | None = None, beta: float = 0.0) -> float:
        x, y = batch[:, :-1], batch[:, 1:]
        logp = F.log_softmax(self.model(x), dim=-1)
        ce = F.nll_loss(logp.flatten(0, 1), y.flatten())
        loss = ce
        if prior_alpha > 0:
            soft = self.ngrams.soft_targets(x)
            loss = (1 - prior_alpha) * loss - prior_alpha * (soft * logp).sum(-1).mean()
        if teacher is not None and beta > 0:
            with torch.no_grad():
                t_logp = F.log_softmax(teacher(x), dim=-1)
            self.budget.forward(x.numel())
            kl = (t_logp.exp() * (t_logp - logp)).sum(-1).mean()
            loss = (1 - beta) * loss + beta * kl
        for g in self.opt.param_groups:
            g["lr"] = self.lr_now()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.budget.train(x.numel())
        self.step_count += 1
        if self.ema is not None:
            with torch.no_grad():
                for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                    pe.lerp_(pm, 1 - self.args.ema_decay)
        if self.budget.progress >= self.next_eval:
            self.record(float(ce))
            self.next_eval += self.args.eval_every
        return float(ce)

    def record(self, train_ce: float) -> None:
        sv = eval_loss(self.model, self.data["stories_val"], self.eos, max_windows=48)
        cv = eval_loss(self.model, self.data["childes_val"], self.eos, max_windows=48)
        row = {
            "progress": round(self.budget.progress, 4),
            "trained_tokens": self.budget.trained_tokens,
            "step": self.step_count,
            "train_ce": round(train_ce, 4),
            "stories_val": round(sv, 4),
            "childes_val": round(cv, 4),
            "wall_s": round(time.time() - self.t0, 1),
            "lr": self.lr_now(),
        }
        self.log["curve"].append(row)
        print(
            f"  [{row['progress']*100:5.1f}%] step {row['step']:5d} train {train_ce:.3f} | val stories {sv:.3f}"
            f" (ppl {math.exp(sv):6.2f}) childes {cv:.3f} | {row['wall_s']:.0f}s"
        )

    def run_until(self, end_progress: float, batch_fn, alpha_fn=None, **step_kw) -> None:
        while self.budget.progress < end_progress - 1e-9:
            self.step(batch_fn(), prior_alpha=alpha_fn() if alpha_fn else 0.0, **step_kw)

    def bsz(self, seq_len: int) -> int:
        return max(1, self.args.batch_tokens // seq_len)

    def diary(self, phase: str) -> None:
        torch.manual_seed(0)
        prompts = [self.tok.encode(p).ids for p in DIARY_PROMPTS]
        lines = []
        for p in prompts:
            out, n = self.model.generate([p], max_new_tokens=40, temperature=0.8, top_k=40, stop_ids=(self.eos,))
            lines.append(self.tok.decode(p + [t for t in out[0] if t != self.eos]))
        self.log["diary"].append({"phase": phase, "samples": lines})
        for l in lines:
            print(f"    > {l}")

    def phase(self, name: str) -> None:
        self.log["phases"].append({"name": name, "start_progress": self.budget.progress, "wall_s": time.time() - self.t0})
        print(f"[phase] {name} (progress {self.budget.progress*100:.1f}%)")

    # ---------------------------------------------------------------- recipes
    def replay_packer(self, seed_offset: int = 0) -> Packer:
        return Packer(self.all_train, self.eos, seed=self.args.seed + seed_offset)

    def baseline(self) -> None:
        self.phase("baseline: shuffled next-token prediction")
        p = self.replay_packer()
        self.run_until(1.0, lambda: p.batch(self.bsz(256), 256))

    def cradle(self) -> None:
        a = self.args
        f1, f2, f3, f4 = a.phases
        e1, e2, e3 = f1, f1 + f2, f1 + f2 + f3
        if a.use_prior:
            self.ngrams = NGramStats(self.all_train, self.cfg.vocab_size, self.eos)
        alpha = lambda: a.prior_alpha * max(0.0, 1 - self.budget.progress / e2) if a.use_prior else 0.0

        if a.use_curriculum:
            # Phase 1: LISTEN to child-directed speech, youngest-age transcripts first, short memory span.
            self.phase("1-listen: child-directed speech, age-ordered, context 64")
            cds = self.data["childes_train"]
            for k in range(4):
                pool = cds[: int(len(cds) * (k + 1) / 4)]
                p = Packer(pool, self.eos, seed=a.seed + k)
                self.run_until(f1 * (k + 1) / 4, lambda: p.batch(self.bsz(64), 64), alpha_fn=alpha)
            self.diary("1-listen")

            # Phase 2: LISTEN to stories, simple -> complex, growing context (working memory).
            self.phase("2-listen: stories simple->complex, context 128->256")
            order = np.argsort(self.data["stories_difficulty"])
            stories = self.data["stories_train"]
            for k, seq_len in enumerate((128, 192, 256)):
                pool = [stories[i] for i in order[: int(len(order) * (k + 1) / 3)]] + cds
                p = Packer(pool, self.eos, seed=a.seed + 10 + k)
                if k == 1 and a.use_sleep:
                    self.ema = copy.deepcopy(self.model).eval()
                self.run_until(f1 + f2 * (k + 1) / 3, lambda: p.batch(self.bsz(seq_len), seq_len), alpha_fn=alpha)
            self.diary("2-listen")
        else:
            self.phase("1+2 (ablated): shuffled next-token prediction")
            p = self.replay_packer()
            self.ema = copy.deepcopy(self.model).eval() if a.use_sleep else None
            self.run_until(e2, lambda: p.batch(self.bsz(256), 256), alpha_fn=alpha)

        if self.ema is None and a.use_sleep:
            self.ema = copy.deepcopy(self.model).eval()
        replay = self.replay_packer(seed_offset=100)
        if a.use_babble:
            self.phase("3-babble: talk, get caregiver feedback, learn from praise and recasts")
            self.babble_phase(e2, e3, replay)
            self.diary("3-babble")
        else:
            self.phase("3 (ablated): plain replay")
            self.run_until(e3, lambda: replay.batch(self.bsz(256), 256))

        if a.use_sleep:
            self.phase("4-sleep: consolidate by distilling from the EMA (memory) model")
            self.run_until(1.0, lambda: replay.batch(self.bsz(256), 256), teacher=self.ema, beta=a.sleep_beta)
        else:
            self.phase("4 (ablated): plain replay")
            self.run_until(1.0, lambda: replay.batch(self.bsz(256), 256))
        self.diary("4-sleep")

    def babble_phase(self, start: float, end: float, replay: Packer) -> None:
        a = self.args
        ngrams = self.ngrams or NGramStats(self.all_train, self.cfg.vocab_size, self.eos)
        end_ids = {self.tok.token_to_id(t) for t in (".", "!", "?", "Ġ.", "Ġ!", "Ġ?") if self.tok.token_to_id(t) is not None}
        carer = Caregiver(ngrams, self.eos, end_ids)
        segments = [d[:64] for d in self.data["childes_train"]]
        segments += [d[i : i + 64] for d in self.data["stories_train"] for i in range(0, max(len(d) - 32, 1), 64)]
        recaster = Recaster(segments, self.cfg.vocab_size)
        rng = np.random.default_rng(a.seed + 7)
        buffer: list[list[list[int]]] = []
        fb_rows = max(1, int(round(self.bsz(256) * a.feedback_frac)))
        for r in range(a.babble_rounds):
            # Talk: caregiver starts an utterance or a story; the learner continues it.
            items = []
            for pool, plen, gen_len in ((self.data["childes_train"], 6, 40), (self.data["stories_train"], 10, 64)):
                cands = [d for d in pool if len(d) >= plen + gen_len]
                pick = rng.choice(len(cands), a.babble_per_round // 2, replace=False)
                prompts = [cands[i][:plen] for i in pick]
                refs = [cands[i][plen : plen + gen_len] for i in pick]
                outs, n_fwd = self.model.generate(prompts, max_new_tokens=gen_len, temperature=1.0, top_k=50, stop_ids=(self.eos,))
                self.budget.forward(n_fwd)
                items += list(zip(prompts, outs, refs))
            babble = [[t for t in o if t != self.eos] for _, o, _ in items]
            refs = [ref for _, _, ref in items]
            # The caregiver adapts: a fresh real-vs-babble discriminator, cross-fitted so no sample scores itself.
            half = np.zeros(len(items), dtype=bool)
            half[rng.permutation(len(items))[: len(items) // 2]] = True
            scores = np.zeros(len(items))
            accs = []
            for fit_mask in (half, ~half):
                carer.disc = type(carer.disc)()
                accs.append(
                    carer.disc.fit([r_ for r_, m in zip(refs, fit_mask) if m], [b for b, m in zip(babble, fit_mask) if m])
                )
                held = np.where(~fit_mask)[0]
                s, _ = carer.score([babble[i] for i in held])
                scores[held] = s
            _, parts = carer.score(babble)
            rank = np.argsort(-scores)
            q = len(items) // 4
            praised = [items[i][0] + babble[i] for i in rank[:q]]
            corrected = recaster.nearest([items[i][0] + babble[i] for i in rank[-q:]])
            buffer.append(praised + corrected)
            buffer = buffer[-2:]
            fb = Packer([x for b in buffer for x in b], self.eos, seed=a.seed + 200 + r)
            best, worst = self.tok.decode(items[rank[0]][0] + items[rank[0]][1]), self.tok.decode(items[rank[-1]][0] + items[rank[-1]][1])
            stat = {"round": r, **parts, "disc_acc_train": float(np.mean(accs)), "best": best, "worst": worst,
                    "worst_recast": self.tok.decode(corrected[-1])}
            self.log["babble"].append(stat)
            print(
                f"  [babble {r}] fluency {parts['fluency']:.2f} rep {parts['repetition']:.3f} ended {parts['ended']:.2f}"
                f" disc-acc {stat['disc_acc_train']:.2f}\n    best : {best[:120]}\n    worst: {worst[:120]}\n    recast: {stat['worst_recast'][:120]}"
            )
            round_end = start + (end - start) * (r + 1) / a.babble_rounds

            def mixed():
                main = replay.batch(self.bsz(256) - fb_rows, 256)
                return torch.cat([main, fb.batch(fb_rows, 256)], 0)

            self.run_until(round_end, mixed)

    def save(self) -> Path:
        name = self.args.name or self.args.recipe
        out = Path(self.args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.cfg.to_dict(), "model": self.model.state_dict()}, out / "model.pt")
        self.log["summary"] = {
            "trained_tokens": self.budget.trained_tokens,
            "forward_only_tokens": self.budget.forward_tokens,
            "units": self.budget.used,
            "steps": self.step_count,
            "wall_s": time.time() - self.t0,
            "params": self.model.num_params(),
        }
        (out / "log.json").write_text(json.dumps(self.log, indent=2, default=str))
        print(f"[done] saved {out} | {json.dumps(self.log['summary'])}")
        return out


def main():
    ap = argparse.ArgumentParser(description="Train a tiny LM with the Cradle recipe or the baseline")
    ap.add_argument("--recipe", choices=["cradle", "baseline"], default="cradle")
    ap.add_argument("--budget-tokens", type=float, default=12e6, help="compute budget in trained-token equivalents")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--no-prior", action="store_true")
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--no-babble", action="store_true")
    ap.add_argument("--no-sleep", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ns = ap.parse_args()
    if ns.threads:
        torch.set_num_threads(ns.threads)
    args = Args(
        recipe=ns.recipe, budget_tokens=ns.budget_tokens, seed=ns.seed, name=ns.name, out=ns.out, d_model=ns.d_model,
        n_layer=ns.n_layer, n_head=ns.n_head, lr=ns.lr, use_prior=not ns.no_prior, use_curriculum=not ns.no_curriculum,
        use_babble=not ns.no_babble, use_sleep=not ns.no_sleep,
    )
    t = Trainer(args)
    print(f"[train] recipe={args.recipe} params={t.model.num_params()/1e6:.2f}M budget={args.budget_tokens/1e6:.1f}M tokens")
    getattr(t, args.recipe)()
    t.save()


if __name__ == "__main__":
    main()
