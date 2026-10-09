"""Download, clean, tokenize and curriculum-score the tiny corpus.

Sources (all public, no credentials):
  * AO-CHILDES: child-directed speech, ordered by the target child's age (via the BabyLM 10M release).
  * TinyStories (validation file): short stories written with a 3-4 year old's vocabulary.
  * Zorro: grammatical minimal pairs restricted to child-directed vocabulary (evaluation only).
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW = DATA_DIR / "raw"
PREPARED = DATA_DIR / "prepared.pt"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
EOS = "<eos>"

URLS = {
    "aochildes.txt": "https://huggingface.co/datasets/cambridge-climb/BabyLM/resolve/main/clean/10M/aochildes.txt",
    "TinyStories-valid.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
}
ZORRO_URL = "https://raw.githubusercontent.com/phueb/Zorro/master/sentences/babyberta/{}.txt"
ZORRO_PARADIGMS = [
    "agreement_determiner_noun-across_1_adjective",
    "agreement_determiner_noun-between_neighbors",
    "agreement_subject_verb-across_prepositional_phrase",
    "agreement_subject_verb-across_relative_clause",
    "agreement_subject_verb-in_question_with_aux",
    "agreement_subject_verb-in_simple_question",
    "anaphor_agreement-pronoun_gender",
    "argument_structure-dropped_argument",
    "argument_structure-swapped_arguments",
    "argument_structure-transitive",
    "binding-principle_a",
    "case-subjective_pronoun",
    "ellipsis-n_bar",
    "filler-gap-wh_question_object",
    "filler-gap-wh_question_subject",
    "irregular-verb",
    "island-effects-adjunct_island",
    "island-effects-coordinate_structure_constraint",
    "local_attractor-in_question_with_aux",
    "npi_licensing-matrix_question",
    "npi_licensing-only_npi_licensor",
    "quantifiers-existential_there",
    "quantifiers-superlative",
]


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        dest.write_bytes(r.read())


def normalize(text: str) -> str:
    text = text.lower().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def load_childes() -> list[str]:
    lines = (RAW / "aochildes.txt").read_text(encoding="utf-8").splitlines()
    return [normalize(l) for l in lines if len(l.split()) >= 2]


def load_stories() -> list[str]:
    raw = (RAW / "TinyStories-valid.txt").read_text(encoding="utf-8")
    chunks = raw.split("<|endoftext|>")[1:]  # the first chunk is a truncated story
    return [s for s in (normalize(c) for c in chunks) if len(s.split()) >= 20]


def load_zorro() -> dict[str, list[tuple[str, str]]]:
    """Returns {paradigm: [(good, bad), ...]}. Zorro files alternate bad/good lines."""
    out = {}
    for name in ZORRO_PARADIGMS:
        path = RAW / "zorro" / f"{name}.txt"
        download(ZORRO_URL.format(name), path)
        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        fix = lambda s: normalize(re.sub(r" ([.?!,])", r"\1", s))
        out[name] = [(fix(lines[i + 1]), fix(lines[i])) for i in range(0, len(lines) - 1, 2)]
    return out


def train_tokenizer(texts: list[str], vocab_size: int) -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=2, special_tokens=[EOS], show_progress=False)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tok.save_model(str(TOKENIZER_DIR))
    return tok


def load_tokenizer() -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer(str(TOKENIZER_DIR / "vocab.json"), str(TOKENIZER_DIR / "merges.txt"))
    tok.add_special_tokens([EOS])
    return tok


def difficulty_scores(docs: list[list[int]], vocab_size: int) -> np.ndarray:
    """Simple-to-complex score: mean of (length rank, mean token-rarity rank)."""
    counts = np.bincount(np.concatenate([np.asarray(d) for d in docs]), minlength=vocab_size) + 1
    surprisal = -np.log(counts / counts.sum())
    length = np.array([len(d) for d in docs], dtype=np.float64)
    rarity = np.array([surprisal[np.asarray(d)].mean() for d in docs])
    rank = lambda x: np.argsort(np.argsort(x)) / max(len(x) - 1, 1)
    return 0.5 * rank(length) + 0.5 * rank(rarity)


def prepare(n_train_stories: int = 12000, n_val_stories: int = 1000, vocab_size: int = 4096) -> dict:
    print("[prepare] fetching corpora")
    for name, url in URLS.items():
        download(url, RAW / name)
    childes = load_childes()
    stories = load_stories()
    zorro = load_zorro()

    val_mask = np.zeros(len(childes), dtype=bool)
    val_mask[::25] = True  # interleaved so validation covers every age band
    childes_train = [c for c, v in zip(childes, val_mask) if not v]
    childes_val = [c for c, v in zip(childes, val_mask) if v]
    stories_val = stories[:n_val_stories]
    stories_train = stories[n_val_stories : n_val_stories + n_train_stories]

    print(f"[prepare] training BPE tokenizer (vocab={vocab_size})")
    tok = train_tokenizer(childes_train + stories_train, vocab_size)
    eos_id = tok.token_to_id(EOS)
    enc = lambda texts: [e.ids for e in tok.encode_batch(texts)]

    data = {
        "vocab_size": tok.get_vocab_size(),
        "eos_id": eos_id,
        "childes_train": enc(childes_train),
        "childes_val": enc(childes_val),
        "stories_train": enc(stories_train),
        "stories_val": enc(stories_val),
        "zorro": {k: [(enc([g])[0], enc([b])[0]) for g, b in v] for k, v in zorro.items()},
    }
    data["stories_difficulty"] = difficulty_scores(data["stories_train"], data["vocab_size"])
    data["childes_difficulty"] = difficulty_scores(data["childes_train"], data["vocab_size"])
    torch.save(data, PREPARED)

    stats = {
        "childes_train_words": sum(len(c.split()) for c in childes_train),
        "stories_train_words": sum(len(s.split()) for s in stories_train),
        "childes_train_tokens": sum(map(len, data["childes_train"])),
        "stories_train_tokens": sum(map(len, data["stories_train"])),
        "stories_val_tokens": sum(map(len, data["stories_val"])),
        "childes_val_tokens": sum(map(len, data["childes_val"])),
        "zorro_pairs": sum(len(v) for v in zorro.values()),
        "vocab_size": data["vocab_size"],
    }
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print("[prepare] " + json.dumps(stats))
    return data


def load_prepared() -> dict:
    if not PREPARED.exists():
        raise SystemExit("No prepared data found. Run `python -m slm.data` first.")
    return torch.load(PREPARED, weights_only=False)


class Packer:
    """Packs documents (joined by EOS) into fixed-length training windows, reshuffling every epoch."""

    def __init__(self, docs: list[list[int]], eos_id: int, seed: int = 0, shuffle: bool = True):
        self.docs = docs
        self.eos = eos_id
        self.rng = np.random.default_rng(seed)
        self.shuffle = shuffle
        self.buffer: list[int] = []
        self.order: list[int] = []
        self.epochs = 0

    def _refill(self, n: int) -> None:
        while len(self.buffer) < n:
            if not self.order:
                idx = np.arange(len(self.docs))
                if self.shuffle:
                    self.rng.shuffle(idx)
                self.order = idx.tolist()[::-1]
                self.epochs += 1
            self.buffer.append(self.eos)
            self.buffer.extend(self.docs[self.order.pop()])

    def batch(self, batch_size: int, seq_len: int) -> torch.Tensor:
        n = batch_size * (seq_len + 1)
        self._refill(n)
        chunk, self.buffer = self.buffer[:n], self.buffer[n:]
        return torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len + 1)


def main():
    ap = argparse.ArgumentParser(description="Download and prepare the corpus")
    ap.add_argument("--train-stories", type=int, default=12000)
    ap.add_argument("--vocab-size", type=int, default=4096)
    args = ap.parse_args()
    prepare(n_train_stories=args.train_stories, vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()
