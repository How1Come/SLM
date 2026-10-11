# Scaled RP narration corpus — sources and licenses

Raw text and `train.jsonl` stay **local only** (gitignored). This file records **names and licenses**, not prose.

## Chinese narration (majority of tokens)

| Source | License | Use |
|--------|---------|-----|
| Project Gutenberg eBooks #23962 (西游记), #25393 (施公案), #57227 (平妖传), #25128 (孽海花), #23863 (水浒传) | Public domain (Gutenberg) | Chunked chapter prose → scene narration SFT (metadata: title + chapter heading only in prompts) |
| `data/rp-narration/train.jsonl` (project-authored) | Project original / PD-style scenes | Seeded examples + same system prompt as the 63-example narration run |

No modern paywalled web novels. No verbatim paste from fan sites.

## Adult NSFW slice (~5–10% of tokens)

| Source | License | Use |
|--------|---------|-----|
| Hugging Face dataset `WoonaAI/ChubCards` (per-file JSON under `RAW/`) | User-submitted character cards (aggregator; training use local only) | Rows kept only when **tags** show adult/NSFW, **parsed age metadata ≥ 21** in name/tags/description, and **no** minor/underage keywords in metadata |

No scraped porn sites. NSFW rows are filtered on **metadata only** before writing `train.jsonl`.

## Regeneration

```bash
python data/rp-scale/build_corpus.py
python -m rp.train_narration --data data/rp-scale/train.jsonl --init-adapter rp/bundled_adapters/narration --epochs 3 --max-minutes 210
```
