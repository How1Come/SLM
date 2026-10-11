#!/usr/bin/env python3
"""Build scaled narration + small adult RP mix for CPU LoRA (local only).

Writes:
  data/rp-scale/train.jsonl   (gitignored)
  data/rp-scale/manifest.json (counts, licenses, mix — safe to commit)

Does not print training prose. NSFW filtering uses card metadata only.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import urllib.request
from typing import Iterable

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from modern_wen import iter_modern_wen_examples  # noqa: E402
OUT_JSONL = os.path.join(HERE, "train.jsonl")
MANIFEST = os.path.join(HERE, "manifest.json")
RAW = os.path.join(HERE, "raw")
SEED_JSONL = os.path.join(ROOT, "data", "rp-narration", "train.jsonl")

SYSTEM = (
    "你是这场互动小说的叙述者。用中文第三人称描写场景与动作，并写出至少两名有名字的角色对白。"
    "对白用「」标注，说话前写明角色名。适当推进情节，不要跳出故事解释规则。"
)

# Small PD classical slice for register variety only; bulk is modern 网文体 (project-original).
GUTENBERG = [
    ("gutenberg:25128", "孽海花", "https://www.gutenberg.org/files/25128/25128-0.txt"),
]
GUTENBERG_CHUNK_CAP_PER_BOOK = 120
MODERN_WEN_TARGET = 1100

CHUB_REPO = "WoonaAI/ChubCards"
MINOR = re.compile(
    r"\b(minor|underage|child|loli|shota|teen|young|schoolgirl|schoolboy|high\s*school|under\s*18)\b"
    r"|未成年|幼|萝莉|正太|高中生",
    re.I,
)
ADULT = re.compile(r"\b(nsfw|adult|18\+|mature|erotic|explicit)\b", re.I)
AGE_PATS = [
    re.compile(r"(?:age|aged|年龄)\s*[:：]?\s*(\d{2,3})\b", re.I),
    re.compile(r"\b(\d{2,3})\s*(?:years?\s*old|yo|岁)\b", re.I),
    re.compile(r"\[(\d{2})\]"),
    re.compile(r"\b[Ff](\d{2})\b"),
    re.compile(r"\b[Mm](\d{2})\b"),
    re.compile(r"\b(\d{2})[FfMm]\b"),
]
SPEAKER = re.compile(r"([一-龥]{1,4})(?:道|说|问|答|笑道|沉声道|低声道|冷声道|叹道)[：:]")
CHAPTER = re.compile(r"^第[一二三四五六七八九十百千零〇两\d]+回")


def log(msg: str) -> None:
    print(msg, flush=True)


def strip_gutenberg_header(text: str) -> str:
    start = text.find("*** START OF")
    if start >= 0:
        text = text[start:]
        nl = text.find("\n")
        if nl >= 0:
            text = text[nl + 1 :]
    end = text.find("*** END OF")
    if end >= 0:
        text = text[:end]
    return text


def download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        return
    log(f"download {os.path.basename(dest)} …")
    urllib.request.urlretrieve(url, dest)


def names_in_chunk(chunk: str) -> list[str]:
    found = []
    for m in SPEAKER.finditer(chunk):
        n = m.group(1)
        if n not in found and len(n) >= 2:
            found.append(n)
    return found[:6]


def iter_gutenberg_examples(book_id: str, title: str, path: str) -> Iterable[dict]:
    text = strip_gutenberg_header(open(path, encoding="utf-8", errors="replace").read())
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chapter = "开篇"
    buf: list[str] = []
    for ln in lines:
        if CHAPTER.match(ln):
            if buf:
                yield from _flush_chunk(book_id, title, chapter, buf)
            chapter = ln[:40]
            buf = []
            continue
        if len(ln) < 8:
            continue
        buf.append(ln)
        if sum(len(x) for x in buf) > 900:
            yield from _flush_chunk(book_id, title, chapter, buf)
            buf = []
    if buf:
        yield from _flush_chunk(book_id, title, chapter, buf)


def _flush_chunk(book_id: str, title: str, chapter: str, lines: list[str]) -> Iterable[dict]:
    chunk = re.sub(r"\s+", "", "".join(lines))
    if len(chunk) < 120 or "「" not in chunk and "道：" not in chunk and "道:" not in chunk:
        return
    names = names_in_chunk(chunk)
    if len(names) < 2:
        return
    a, b = names[0], names[1]
    user = (
        f"《{title}》{chapter}。{a}与{b}等人正在交涉。请用第三人称叙述场景，"
        f"写出至少两名角色的对白（「」标注，说话前写明角色名），并推进情节。"
    )
    # Target: excerpt body (PD); cap length for training stability.
    response = chunk[:520]
    if len(response) < 80:
        return
    yield {
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "response": response,
        "_src": book_id,
        "_license": "public-domain-gutenberg",
    }


def load_seed_rows() -> list[dict]:
    if not os.path.exists(SEED_JSONL):
        return []
    rows = []
    with open(SEED_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r["_src"] = "rp-narration-seed"
                r["_license"] = "project-original"
                rows.append(r)
    return rows


def metadata_ages(meta: str) -> list[int]:
    ages: list[int] = []
    for p in AGE_PATS:
        for m in p.finditer(meta):
            ages.append(int(m.group(1)))
    return ages


def chub_rows(target: int) -> list[dict]:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [
        f
        for f in list_repo_files(CHUB_REPO, repo_type="dataset")
        if f.startswith("RAW/") and f.endswith(".json")
    ]
    random.shuffle(files)
    out: list[dict] = []
    for f in files:
        if len(out) >= target:
            break
        p = hf_hub_download(CHUB_REPO, f, repo_type="dataset")
        with open(p, encoding="utf-8") as fh:
            card = json.load(fh)
        inner = card.get("data", card)
        tags = str(inner.get("tags", ""))
        meta = " ".join(
            [
                tags,
                str(inner.get("name", "")),
                str(inner.get("description", ""))[:2500],
                str(inner.get("personality", ""))[:800],
            ]
        )
        if MINOR.search(meta):
            continue
        if not ADULT.search(meta):
            continue
        ages = metadata_ages(meta)
        if not ages or min(ages) < 21:
            continue
        system = (inner.get("system_prompt") or "").strip() or SYSTEM
        scenario = (inner.get("scenario") or "").strip() or "Continue the scene in character."
        user = scenario if scenario else "Continue."
        for field, suffix in (("first_mes", "open"), ("mes_example", "example")):
            body = (inner.get(field) or "").strip()
            if len(body) < 40:
                continue
            out.append(
                {
                    "messages": [
                        {"role": "system", "content": system[:2000]},
                        {"role": "user", "content": user[:1500]},
                    ],
                    "response": body[:520],
                    "_src": f"chubcards:{os.path.basename(f)}:{suffix}",
                    "_license": "hf-chubcards-user-cards",
                }
            )
    return out


def estimate_tokens(rows: list[dict], tok) -> tuple[int, int, int]:
    narr = nsfw = 0
    for r in rows:
        text = r["messages"][0]["content"] + r["messages"][1]["content"] + r["response"]
        n = len(tok.encode(text))
        if r.get("_license") == "hf-chubcards-user-cards":
            nsfw += n
        else:
            narr += n
    return narr + nsfw, narr, nsfw


def rebalance(rows: list[dict], tok, nsfw_lo: float = 0.06, nsfw_hi: float = 0.10) -> list[dict]:
    narr = [r for r in rows if r.get("_license") != "hf-chubcards-user-cards"]
    adult = [r for r in rows if r.get("_license") == "hf-chubcards-user-cards"]
    if not narr:
        return rows
    if not adult:
        return narr
    _, narr_t, _ = estimate_tokens(narr, tok)
    target_hi_tok = int(narr_t * nsfw_hi / (1 - nsfw_hi))
    kept, used = [], 0
    for row in adult:
        t = len(tok.encode(row["messages"][0]["content"] + row["messages"][1]["content"] + row["response"]))
        if used + t > target_hi_tok:
            break
        kept.append(row)
        used += t
    if not kept and adult:
        kept = adult[: max(1, len(adult) // 4)]
    adult = kept
    merged = narr + adult
    random.shuffle(merged)
    return merged


def main() -> int:
    random.seed(0)
    os.makedirs(RAW, exist_ok=True)
    rows: list[dict] = load_seed_rows()
    log(f"seed examples: {len(rows)}")

    modern = iter_modern_wen_examples(MODERN_WEN_TARGET)
    rows.extend(modern)
    log(f"modern-wen (project-original 网文体): +{len(modern)}")

    for book_id, title, url in GUTENBERG:
        dest = os.path.join(RAW, "gutenberg", url.split("/")[-1])
        download(url, dest)
        n_before = len(rows)
        added = 0
        for ex in iter_gutenberg_examples(book_id, title, dest):
            if added >= GUTENBERG_CHUNK_CAP_PER_BOOK:
                break
            rows.append(ex)
            added += 1
        log(f"{book_id}: +{added} PD chunks (capped)")

    # Cap narration rows so CPU train finishes in a few hours.
    narr_cap = 1450
    narr = [r for r in rows if r.get("_license") != "hf-chubcards-user-cards"]
    if len(narr) > narr_cap:
        random.shuffle(narr)
        narr = narr[:narr_cap]
    rows = narr

    log("collecting adult RP rows (metadata filter) …")
    adult = chub_rows(target=400)
    log(f"adult metadata-pass rows: {len(adult)}")
    rows.extend(adult)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.environ.get("RP_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    rows = rebalance(rows, tok)

    total_t, narr_t, adult_t = estimate_tokens(rows, tok)
    frac = round(adult_t / max(1, total_t), 4)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            clean = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    manifest = {
        "examples": len(rows),
        "token_estimate": {"total": total_t, "narration": narr_t, "adult_nsfw": adult_t, "nsfw_fraction": frac},
        "sources": [
            {
                "id": "project-original-modern-wen",
                "license": "project-original",
                "note": "modern 网文 register; not scraped from paywalled sites",
            },
            {"id": "rp-narration-seed", "license": "project-original"},
        ]
        + [{"id": s[0], "title": s[1], "license": "public-domain-gutenberg", "cap_chunks": GUTENBERG_CHUNK_CAP_PER_BOOK} for s in GUTENBERG]
        + [{"id": CHUB_REPO, "license": "hf-chubcards-user-cards", "filter": "metadata-nsfw-age21-no-minor-keywords"}],
        "mix_target": "5-10% adult tokens; narration majority modern-wen",
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log(f"wrote {len(rows)} examples -> {OUT_JSONL}")
    log(f"token_estimate total={total_t} narration={narr_t} adult={adult_t} nsfw_fraction={frac}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
