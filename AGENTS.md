# SLM project brief

## Goal

Build a **Chinese role-play small model** used through **SillyTavern**, **learning from chat** (edits, swipes, feedback), on **CPU only**, with the working copy and runtime kept **in the cloud**. Public home (empty today, not pushed yet): [https://github.com/How1Come/SLM](https://github.com/How1Come/SLM).

## Cradle experiment (`slm/`)

A from-scratch **~3.4M-parameter** language model trained on child-directed speech (AO-CHILDES) and TinyStories using the **Cradle** recipe: listen in age/difficulty order, optional babble-with-caregiver-feedback, optional sleep (EMA self-distillation). Same compute budget as a plain shuffled next-token **baseline**.

**Recommended run:** listen-only Cradle — babble and sleep did not help in this setup; the **listening phases beat the baseline** on held-out perplexity. Rationale, phases, and numbers are in [`docs/slm-design.md`](docs/slm-design.md) and the results table in [`README.md`](README.md).

## Role-play server (`rp/`)

**Qwen2.5** on CPU: an **OpenAI-compatible** HTTP API for SillyTavern. The server logs generations, turns **edited replies**, **swipes**, and chat commands into training signals, and runs **LoRA “sleep”** to consolidate learning into adapters. Sample character card: [`rp/characters/su_xiaotang.json`](rp/characters/su_xiaotang.json) (苏小糖).

## How to run

### `slm/` (from [`README.md`](README.md))

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy tokenizers            # + transformers, only if you want the --judge metric

python -m slm.data                       # downloads ~22 MB (AO-CHILDES, TinyStories-valid, Zorro), trains a BPE tokenizer
python -m slm.train --recipe cradle --no-babble --no-sleep --name listen   # recommended "listen-only" Cradle, ~13 min
python -m slm.train --recipe cradle --name cradle        # full 4 phases incl. the babble/feedback loop
python -m slm.train --recipe baseline --name baseline    # same compute, plain shuffled next-token training
python -m slm.evaluate runs/listen/model.pt runs/cradle/model.pt runs/baseline/model.pt [--judge]
python -m slm.chat --checkpoint runs/listen/model.pt      # interactive; or add --prompt "once upon a time" -n 3
```

Smoke test: add `--budget-tokens 1e6` to train commands. Full comparison script: `scripts/run_experiments.sh 12e6 0`.

### `rp/` (from `rp/server.py`)

```bash
python -m rp.server [--port 8619] [--data-dir rp_data] [--model Qwen/Qwen2.5-0.5B-Instruct]
```

**SillyTavern:** API = Chat Completion, source = Custom (OpenAI-compatible), endpoint = `http://127.0.0.1:8619/v1`.
