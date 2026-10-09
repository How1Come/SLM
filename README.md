# SLM: a baby-inspired small language model you can train on a laptop CPU

This repo trains a ~3.4M-parameter language model from scratch on about 2.3M words of child-directed
speech and children's stories, in roughly 13 minutes on a 4-core CPU. Training follows **Cradle**, a recipe
modeled on how babies pick up language. A plain next-token baseline gets exactly the same compute budget,
so the two can be compared directly.

| Phase | What the "baby" does | What actually happens |
|---|---|---|
| 1. Listen (CDS) | hears simple speech from caregivers, youngest-age transcripts first | next-token prediction on AO-CHILDES in age order, with a 64-token context ("short memory"), plus soft targets from a bigram model (infants learn transitional probabilities first) |
| 2. Listen (stories) | hears longer and harder language | stories sorted simple to complex, with the context growing 128 → 192 → 256; the bigram prior fades out |
| 3. Babble + feedback | talks, and the caregiver reacts | the model continues caregiver prompts; a caregiver critic (trigram fluency, a real-vs-babble discriminator re-fit every round, a repetition penalty, an ending bonus) ranks the outputs; the model trains on its **praised** utterances plus **recasts** (the closest real utterance to its worst attempts), mixed with replayed real data |
| 4. Sleep | consolidates what it learned | self-distillation from an exponential moving average (EMA) "memory" copy of its own weights |

The design rationale, the literature behind it, and measured results are in the design doc
(`docs/slm-design.md` in the Project context). A short version of the results is at the bottom of this file.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy tokenizers            # + transformers, only if you want the --judge metric

python -m slm.data                       # downloads ~22 MB (AO-CHILDES, TinyStories-valid, Zorro), trains a BPE tokenizer
python -m slm.train --recipe cradle      # ~13 min on 4 CPU cores -> runs/cradle/model.pt
python -m slm.train --recipe baseline    # same compute, plain shuffled next-token training
python -m slm.evaluate runs/cradle/model.pt runs/baseline/model.pt [--judge]
python -m slm.chat --checkpoint runs/cradle/model.pt
```

For a quick smoke test, add `--budget-tokens 1e6` to the train commands (about 1 minute each).

To reproduce the whole comparison (baseline, Cradle, and four ablations, then a Markdown table in `runs/report.md`):

```bash
scripts/run_experiments.sh 12e6 0        # budget (trained-token equivalents), seed
```

Ablation flags for `slm.train --recipe cradle`: `--no-curriculum`, `--no-prior`, `--no-babble`, `--no-sleep`.

## Data (all public, no credentials)

- **AO-CHILDES** (from the BabyLM 10M release): about 345k words of caregiver speech to children, ordered by
  the child's age.
- **TinyStories**: 12,000 stories (about 2.0M words) from the official validation file, plus 1,000 held-out
  stories for evaluation.
- **Zorro**: 23 grammatical minimal-pair tests that use child-directed vocabulary (evaluation only).

All text is lowercased, and the tokenizer is byte-level BPE with 4,096 tokens.

## Fair-compute accounting

Compute is counted in token-forward units: a trained token costs 3 units (forward + backward) and a
forward-only token costs 1 (sampling during babbling, teacher passes during sleep). Each run stops at the
same number of units. The bigram, trigram, and discriminator "caregiver" models cost seconds and are not
counted. Cradle's short early contexts make it slightly cheaper per token than the baseline, and that
saving is also not credited to Cradle.

## Layout

```
slm/data.py       corpus download, cleaning, tokenizer, curriculum difficulty scores, packing
slm/model.py      tiny GPT: RoPE, RMSNorm, tied embeddings, KV-cache sampling
slm/caregiver.py  n-gram statistics, adaptive discriminator, recaster, caregiver score
slm/train.py      Cradle phases + baseline, compute budget, EMA teacher
slm/evaluate.py   perplexity, Zorro, generation metrics, optional TinyStories-33M judge
slm/chat.py       interactive / one-shot sampling
slm/report.py     results table
```

## Results

See below (filled in from `runs/report.md` after the experiment run).
