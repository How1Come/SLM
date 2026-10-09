"""The "caregiver": cheap, non-neural feedback signals built from the corpus itself.

* NGramStats   - transitional probabilities (bigram prior for "statistical learning"; trigram fluency).
* Discriminator - a hashed n-gram logistic regression re-fit every round to tell real speech from babble,
                  so the feedback adapts to the learner's *current* mistakes.
* Recaster     - retrieves the closest real utterance to what the learner tried to say (a "recast").
* Caregiver    - combines the above into one score per babbled utterance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NGramStats:
    def __init__(self, docs: list[list[int]], vocab_size: int, eos_id: int, prior_strength: float = 4.0):
        V = vocab_size
        self.V = V
        stream = []
        for d in docs:
            stream.append(eos_id)
            stream.extend(d)
        s = np.asarray(stream, dtype=np.int64)
        self.unigram = (np.bincount(s, minlength=V) + 1.0) / (len(s) + V)
        big = np.bincount(s[:-1] * V + s[1:], minlength=V * V).reshape(V, V).astype(np.float32)
        self.bigram_counts = big
        self.context_counts = big.sum(1)
        # Dirichlet-smoothed P(next | prev), dense V x V (64 MB at V=4096).
        prior = (big + prior_strength * self.unigram[None, :]) / (self.context_counts[:, None] + prior_strength)
        self.bigram = torch.from_numpy(prior.astype(np.float32))
        tri_keys = (s[:-2] * V + s[1:-1]) * V + s[2:]
        self.tri_keys, self.tri_counts = np.unique(tri_keys, return_counts=True)

    def soft_targets(self, inputs: torch.Tensor) -> torch.Tensor:
        """Bigram distribution over the next token for every input position: [B, T, V]."""
        return self.bigram[inputs]

    def logprob(self, seq: list[int]) -> float:
        """Mean interpolated trigram log-probability per token (higher = more caregiver-like)."""
        if len(seq) < 3:
            return float(np.log(self.unigram[seq]).mean()) if seq else -10.0
        s = np.asarray(seq, dtype=np.int64)
        a, b, c = s[:-2], s[1:-1], s[2:]
        keys = (a * self.V + b) * self.V + c
        pos = np.searchsorted(self.tri_keys, keys).clip(max=len(self.tri_keys) - 1)
        tri = np.where(self.tri_keys[pos] == keys, self.tri_counts[pos], 0)
        ctx = self.bigram_counts[a, b]
        p3 = np.where(ctx > 0, tri / np.maximum(ctx, 1), 0.0)
        p2 = self.bigram_counts[b, c] / np.maximum(self.context_counts[b], 1)
        p1 = self.unigram[c]
        p = 0.6 * p3 + 0.3 * p2 + 0.1 * p1
        return float(np.log(p).mean())


def _hashed_features(seq: list[int], dim: int) -> list[int]:
    feats = [t % dim for t in seq]
    feats += [(a * 1000003 + b) % dim for a, b in zip(seq, seq[1:])]
    feats += [((a * 1000003 + b) * 1000033 + c) % dim for a, b, c in zip(seq, seq[1:], seq[2:])]
    return feats or [0]


class Discriminator(nn.Module):
    """Bag-of-hashed-n-grams logistic regression: P(utterance came from a real caregiver)."""

    def __init__(self, dim: int = 1 << 18):
        super().__init__()
        self.dim = dim
        self.w = nn.EmbeddingBag(dim, 1, mode="mean")
        nn.init.zeros_(self.w.weight)
        self.b = nn.Parameter(torch.zeros(1))

    def _bag(self, seqs):
        feats = [_hashed_features(s, self.dim) for s in seqs]
        offsets = torch.tensor([0] + [len(f) for f in feats[:-1]]).cumsum(0)
        return torch.tensor([x for f in feats for x in f]), offsets

    def forward(self, seqs) -> torch.Tensor:
        flat, offsets = self._bag(seqs)
        return self.w(flat, offsets).squeeze(-1) + self.b

    def fit(self, real: list[list[int]], fake: list[list[int]], epochs: int = 40) -> float:
        opt = torch.optim.Adam(self.parameters(), lr=0.1)
        y = torch.cat([torch.ones(len(real)), torch.zeros(len(fake))])
        seqs = real + fake
        flat, offsets = self._bag(seqs)
        for _ in range(epochs):
            logits = self.w(flat, offsets).squeeze(-1) + self.b
            loss = F.binary_cross_entropy_with_logits(logits, y) + 1e-4 * self.w.weight.pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = ((self.w(flat, offsets).squeeze(-1) + self.b > 0).float() == y).float().mean()
        return float(acc)

    @torch.no_grad()
    def log_p_real(self, seqs) -> np.ndarray:
        return F.logsigmoid(self(seqs)).numpy()


class Recaster:
    """TF-IDF nearest-neighbour lookup over real segments (hashed bag of tokens)."""

    def __init__(self, segments: list[list[int]], vocab_size: int):
        self.segments = segments
        self.V = vocab_size
        df = np.zeros(vocab_size)
        for s in segments:
            df[np.unique(s)] += 1
        self.idf = torch.tensor(np.log((1 + len(segments)) / (1 + df)) + 1.0, dtype=torch.float32)
        self.matrix = self._embed(segments)

    def _embed(self, seqs) -> torch.Tensor:
        rows, cols, vals = [], [], []
        for i, s in enumerate(seqs):
            u, c = np.unique(np.asarray(s, dtype=np.int64), return_counts=True)
            w = torch.tensor(c, dtype=torch.float32) * self.idf[u]
            w = w / (w.norm() + 1e-8)
            rows += [i] * len(u)
            cols += u.tolist()
            vals.append(w)
        vals = torch.cat(vals) if vals else torch.zeros(0)
        return torch.sparse_coo_tensor(torch.tensor([rows, cols]), vals, (len(seqs), self.V)).coalesce()

    def nearest(self, queries: list[list[int]]) -> list[list[int]]:
        q = self._embed(queries).to_dense()
        sims = torch.sparse.mm(self.matrix, q.T)  # [N, Q]
        best = sims.argmax(0).tolist()
        return [self.segments[i] for i in best]


def repetition_rate(seq: list[int], n: int = 3) -> float:
    grams = [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


class Caregiver:
    def __init__(self, ngrams: NGramStats, eos_id: int, end_ids: set[int]):
        self.ngrams = ngrams
        self.eos = eos_id
        self.end_ids = end_ids | {eos_id}
        self.disc = Discriminator()

    def score(self, utterances: list[list[int]]) -> tuple[np.ndarray, dict]:
        flu = np.array([self.ngrams.logprob(u) for u in utterances])
        disc = self.disc.log_p_real(utterances)
        rep = np.array([repetition_rate(u) for u in utterances])
        ended = np.array([float(bool(u) and u[-1] in self.end_ids) for u in utterances])
        short = np.array([float(len(u) < 4) for u in utterances])
        z = lambda x: (x - x.mean()) / (x.std() + 1e-8)
        score = z(flu) + z(disc) - 3.0 * rep + 0.5 * ended - 2.0 * short
        parts = {"fluency": flu.mean(), "disc_log_p_real": disc.mean(), "repetition": rep.mean(), "ended": ended.mean()}
        return score, {k: float(v) for k, v in parts.items()}
