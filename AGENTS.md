# SLM project brief

## Goal

Build a **Chinese role-play small model** used through **SillyTavern**, **learning from chat** (edits, swipes, feedback), on **CPU only**, with the working copy and runtime kept **in the cloud**. Public home: [https://github.com/How1Come/SLM](https://github.com/How1Come/SLM).

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

**SillyTavern:** API = Chat Completion, source = Custom (OpenAI-compatible), endpoint = `http://127.0.0.1:8619/v1`, model id **`qwen2.5-0.5b-rp`** (override with `--model-id`).

**Transformers 5:** `rp/engine.py` normalizes `tokenizer.apply_chat_template(..., tokenize=True)` so it works whether Hugging Face returns a token-id list (older `transformers`) or a `BatchEncoding` (`transformers` 5). Sleep and chat both depend on that path; without it, prompt encoding can fail on v5.

---

## Working mechanism

Two tracks share this repo: a **from-scratch English baby-LM experiment** (`slm/`) and a **Chinese role-play server** (`rp/`) that fine-tunes an existing instruct model from chat.

### Cradle (`slm/`): four phases, one fixed token budget

Training is split into phases that consume fixed fractions of the same compute budget (12M trained-token equivalents on the default run — about **14 minutes on 4 CPU cores**, **~3.4M parameters**, **~2.3M words** of AO-CHILDES + TinyStories). One optimizer and one cosine schedule span the whole run.

1. **Listen (child-directed speech)** — Transcripts are visited in **child-age order** (youngest first). Context is short (**64 tokens**). Next-token loss is mixed with **soft targets from a corpus bigram model** (transitional-probability prior).
2. **Listen (stories)** — Stories go **simple to complex**; context **grows** (128, then 192, then 256 tokens). The bigram prior **fades to zero**.
3. **Babble** — The model continues **caregiver prompts** from the corpus. A **corpus-only critic** (trigram fluency, adaptive discriminator, repetition/ending heuristics) ranks outputs. Training uses the **best** continuations and **recasts** (real corpus sentences) for the worst; real data is replayed each round.
4. **Sleep** — **Self-distillation** from an EMA copy of the model’s own weights (KL + CE on replayed real data).

**Recommended recipe today:** stop after the listening phases. In measured runs, **babble and sleep hurt** held-out perplexity (the critic only knows the same small corpus, so it does not add new information). Use `--no-babble --no-sleep` (listen-only Cradle).

### Role-play (`rp/`): pretrained Qwen + LoRA from chat

**Qwen2.5-0.5B-Instruct** is already pretrained. SillyTavern sends the **character card as the system prompt**; the server serves `/v1/chat/completions` and **logs every generation** (prompt, adapter version, timing).

On each request, `rp/learn.py` compares SillyTavern’s message history to the log:

- **Edited assistant message** (text differs from every logged swipe for that turn) → **preferred correction** (`chosen` = your edit, `rejected` = the model’s original). Strong learning signal.
- **Swipe / regenerate** (another reply for the same turn) → **weak preference** (kept reply preferred over the swiped-away one).
- **Unchanged match** to a logged reply → acceptance; other swipes for that turn become weak pairs.

**Sleep** (`python -m rp.sleep`, chat `!sleep` / `!睡觉`, or `POST /rp/sleep`) trains a **new LoRA adapter on CPU** from accumulated signals (SFT on preferred text, DPO on pairs, KTO-style term on rejections). Adapters are versioned under `rp_data/adapters/vN`; `current.json` selects the active one; the server **hot-reloads** after a successful sleep. This **changes weights** — it is not “just a longer prompt.” Roll back with `python -m rp.admin rollback`.

A fresh GitHub clone has **no adapter weights** (`rp_data/` is local); inference starts as the **base Qwen** model until you sleep at least once.

---

## How to train

### Cradle (from-scratch LM)

Setup and commands match [`README.md`](README.md):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy tokenizers            # + transformers, only if you want the --judge metric

python -m slm.data                       # downloads corpora, trains BPE tokenizer
python -m slm.train --recipe cradle --no-babble --no-sleep --name listen   # recommended listen-only Cradle
python -m slm.train --recipe baseline --name baseline                        # same budget, shuffled next-token baseline
```

- **Recommended:** `python -m slm.train --recipe cradle --no-babble --no-sleep --name listen` (~13–14 min at default budget).
- **Baseline comparison:** `python -m slm.train --recipe baseline --name baseline`.
- **Done when:** training finishes and writes **`runs/listen/model.pt`** (or `runs/<name>/model.pt`). Check quality with held-out perplexity, e.g. `python -m slm.evaluate runs/listen/model.pt runs/baseline/model.pt`.
- **Smoke test:** add `--budget-tokens 1e6` (~1 min per run). Full sweep: `scripts/run_experiments.sh 12e6 0`.

### Role-play (SillyTavern + sleep)

1. **Start the server** (install `torch`, `transformers`, `peft`, `fastapi`, `uvicorn`, `httpx` as needed for `rp/`):

   ```bash
   python -m rp.server [--port 8619] [--data-dir rp_data] [--model Qwen/Qwen2.5-0.5B-Instruct]
   ```

2. **SillyTavern:** Chat Completion → Custom (OpenAI-compatible) → `http://127.0.0.1:8619/v1`, model **`qwen2.5-0.5b-rp`**. Load a character (e.g. [`rp/characters/su_xiaotang.json`](rp/characters/su_xiaotang.json)); the card becomes the system prompt.

3. **Collect signals:** Chat normally. **Edit** an assistant bubble in SillyTavern to teach a better line. **Swipe/regenerate** to record a weak rejection of the previous reply. Optional chat hooks: `!good` / `!bad` / `!fix <text>` / `!status` (see `rp/learn.py`).

4. **Sleep (consolidate into LoRA):** any of:

   ```bash
   python -m rp.sleep [--data-dir rp_data] [--epochs 4] [--max-minutes 10]
   python -m rp.admin sleep [--wait] [--epochs 4]    # talks to the running server
   ```

   Or in chat: `!sleep` / `!睡觉`. Or HTTP: `POST /rp/sleep` (optional JSON `{"wait": true, "args": {"epochs": 4}}`). Status: `GET /rp/sleep`. Dashboard: `http://127.0.0.1:8619/`.

   Default server **auto-sleep** after **10** new signals and **300 s** idle (`--auto-sleep-signals`, `--auto-sleep-idle`).

---

## Current progress

**Nothing is training right now.** Cradle experiments on this recipe are **finished** (numbers below are from completed runs, not live jobs).

### Cradle (held-out perplexity; lower is better)

| Recipe | Stories ppl | CDS ppl |
|---|---|---|
| Baseline | 11.32 | 37.8 |
| Listen-only, natural mix (2 seeds) | 11.09 (−2%) | 34.3 (−9%) |
| Listen-only, matched mix (2 seeds) | 10.87 (−4%) | 43.9 (+16%) |
| All four phases (babble + sleep) | 11.60 (worse) | — |

Grammar (**Zorro**) and the optional **TinyStories-33M judge** were ties within noise. Details: [`docs/slm-design.md`](docs/slm-design.md), [`README.md`](README.md).

### Role-play (苏小糖)

- **14-edit eval** ([`runs/rp-demo/eval-v1-cli.md`](runs/rp-demo/eval-v1-cli.md)): persona score **0.417 → 0.858**; gold NLL **3.599 → 3.182**.
- **One live edit** ([`runs/rp-demo/live-one-edit.md`](runs/rp-demo/live-one-edit.md)): after sleep on「你是谁？」, chosen-reply NLL **2.77 → 0.57**; replies picked up **本座** / **小书生**. Held-out「你多大了？」**copied those phrases** but still **did not answer her age** — one memorized correction, not a broad persona shift.
- **Adapter weights are not in git**; reproducing v1 requires your own `rp_data/` or re-running sleep on your signals.

---

## How improvement shows up if muzi trains more

### Cradle

Success is **lower held-out perplexity** than the baseline at the **same token budget** (stories and/or CHILDES, depending on what you care about). Listen-only Cradle is the supported recipe; babble/sleep are experimental and currently regressed. This model is an English tiny LM on children’s text — it will **not** become a Chinese role-play chatbot.

### Role-play

- **Many consistent edits** on the same habit (what she calls the user, speech tics, tone) plus **sleep** nudge **later** replies toward that style — see the 14-edit eval for a broader shift than a single line.
- **One edit** can **memorize one correction** (as with「你是谁？」) without fixing other prompts; always check **held-out** lines, not only the edited turn.
- Compare **before/after** with `python -m rp.evaluate` or the metrics in `runs/rp-demo/`. **Older adapters stay on disk**; use `python -m rp.admin rollback` if a sleep pass drifts or overfits.
