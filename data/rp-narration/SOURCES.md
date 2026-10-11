# Narration LoRA training data — sources and licenses

All training text in `train.jsonl` is **original Chinese prose** written for this project, in a classical-scene style. It is **not** scraped from copyrighted web novels or paywalled fiction sites.

## Inspiration (public domain, not copied verbatim)

Plot situations echo **public-domain** Chinese classical fiction and folklore (e.g. 聊斋志异, 水浒传, 三国演义 — authors long deceased; texts are PD in most jurisdictions). We did **not** paste chapter text from modern editions or fan sites; scenes are re-authored for the LoRA format.

## Format goal

Each example teaches the assistant to:

- narrate setting and action in third person;
- advance the scene with a small plot turn;
- include **at least two named characters** speaking in one reply, using clear name tags and quoted dialogue.

## Held-out evaluation

`heldout.json` prompts are project-authored and are **not** included in `train.jsonl`.

## Regeneration

```bash
python data/rp-narration/build_train.py   # rewrites train.jsonl from the embedded scene bank
```
