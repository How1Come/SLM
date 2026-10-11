# Scaled RP narration corpus — sources and licenses

Raw text and `train.jsonl` stay **local only** (gitignored). This file records **names and licenses**, not prose.

## Why not scraped 网文?

Paywalled or platform-hosted **网文** (起点、晋江等) is **copyrighted**; the project rule is **no scraped copyrighted web novels**. The scaled mix therefore uses **project-original modern 网文体** (都市/电竞/异能/修仙/悬疑等白话节奏) for most narration tokens, not classical-only PD fiction.

## Chinese narration (majority of tokens)

| Source | License | Use |
|--------|---------|-----|
| `data/rp-scale/modern_wen.py` → **project-original modern 网文** | Project original | ~1100 procedural scenes, simplified Chinese, web-novel pacing |
| `data/rp-narration/train.jsonl` | Project original | Seed scenes + shared system prompt |
| Project Gutenberg **孽海花** (#25128) only, capped | Public domain | Small classical register slice (~≤120 chunks), not the bulk |

## Adult NSFW slice (~5–10% of tokens)

| Source | License | Use |
|--------|---------|-----|
| Hugging Face `WoonaAI/ChubCards` | User-submitted cards (local training only) | Metadata-only filter: adult tags, age ≥ 21, no minor keywords |

## Regeneration

```bash
python data/rp-scale/build_corpus.py
python -m rp.train_narration --data data/rp-scale/train.jsonl --init-adapter rp/bundled_adapters/narration ...
```
