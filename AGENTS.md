# AGENTS.md

Instructions for agents working in this checkout. Read this before changing code.

## Goal

muzi (GitHub user How1Come) wants a **Chinese role-play model used through SillyTavern**. It learns from the chat itself: edits, swipes, and short commands. It runs **CPU-only** and **stays in the cloud**. Do not move the checkout, weights, checkpoints, or chat database onto a local machine unless he asks.

The product is `rp/`: an OpenAI-compatible server in front of `Qwen/Qwen2.5-0.5B-Instruct`, with LoRA "sleep" that consolidates what he taught the character. The sample character is 苏小糖 (`rp/characters/su_xiaotang.json`): a three-hundred-year-old fox spirit who calls herself 本座, calls the user 小书生, writes *starred* actions, and never talks like an assistant.

`slm/` is a finished experiment, not the product. It trains a ~3.4M-parameter model from scratch on child-directed speech and TinyStories. The **listening** phases of the Cradle recipe beat a same-compute baseline. The **babble** and **sleep** phases did not. Leave that result alone unless asked to run a new experiment. The design and the numbers are in `docs/slm-design.md` and `runs/report.md`.

The intended public remote, once this environment is logged into GitHub as How1Come, is the empty repo https://github.com/How1Come/SLM. Until that login exists, do not push to github.com and do not create that repo.

## Constraints

- CPU only. Install the CPU wheel of PyTorch. Do not assume a GPU, CUDA, or a paid model API.
- Keep the running work in the cloud. SillyTavern is a client pointed at this server. It is not vendored here, and it is not a reason to copy the project off the machine.
- Every commit's author and committer must be `How1Come <118341976+How1Come@users.noreply.github.com>`. Never use `Cursor Agent` or `cursoragent@cursor.com`. Set the identity with repo-local `git config user.name` / `user.email`, or with `GIT_AUTHOR_*` and `GIT_COMMITTER_*` on the command. Do not change the global git identity.
- This checkout's `origin` is the Cursor remote. Push feature branches there. Never force-push `main`. Force-with-lease is only for a feature branch this work created.
- Do not commit `.venv/`, `data/raw/`, `data/prepared.pt`, `runs/**/*.pt`, or `rp_data/`. Those paths are in `.gitignore`. Hugging Face weights stay in the local cache.
- Do not retrain the Cradle model to "fix" babble or sleep. The measured result is that they hurt held-out perplexity because the caregiver critic only knows the same small corpus. The recommended recipe is listen-only.
- Do not point role-play learning at the 3.4M Cradle checkpoint. Learning is a fresh LoRA on Qwen2.5-0.5B-Instruct. Each sleep trains from the base weights plus the current signals. Adapters are not chained.
- Held-out prompts in `rp/characters/su_xiaotang_demo.json` (`heldout`) are for evaluation only. Do not feed those gold replies in as training edits.
- Python 3.10 or newer. This environment is 3.12.

## Setup

From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install transformers peft fastapi uvicorn httpx Pillow
```

`requirements.txt` covers the Cradle experiment (`torch`, `numpy`, `tokenizers`). `transformers` is optional for `slm.evaluate --judge` and required for `rp`. `rp/engine.py` calls `from_pretrained(..., dtype=...)`, so the installed `transformers` must accept `dtype` (the older `torch_dtype`-only releases will not load). `peft` is required once an adapter exists. `fastapi` and `uvicorn` run the server. `httpx` is the client for `rp.demo` and `rp.admin`. `Pillow` is only for `rp.make_card`.

No API keys. Corpus downloads and the Qwen weights are public Hugging Face files.

## Cradle experiment (`slm/`)

A 6-layer, 192-wide, 6-head decoder (about 3.4M parameters, RoPE, RMSNorm, tied embeddings, 256-token context). Byte-level BPE, 4,096 tokens, lowercased text. Data is about 2.3M words: AO-CHILDES caregiver speech in age order, and 12,000 TinyStories. Zorro is evaluation only.

Compute is counted in token-forward units. A trained token costs 3. A forward-only token costs 1. Every recipe stops at the same budget (default 12e6 trained-token equivalents, about 13–15 minutes on 4 CPU cores).

```bash
python -m slm.data
python -m slm.train --recipe cradle --no-babble --no-sleep --name listen
python -m slm.train --recipe baseline --name baseline
python -m slm.evaluate runs/listen/model.pt runs/baseline/model.pt
python -m slm.chat --checkpoint runs/listen/model.pt
```

`python -m slm.data` downloads about 22 MB and writes `data/prepared.pt` (gitignored) plus `data/tokenizer/` (committed). Add `--budget-tokens 1e6` for a one-minute smoke test. `--judge` on evaluate scores generations with TinyStories-33M and needs `transformers`.

Full comparison, then a table in `runs/report.md`:

```bash
scripts/run_experiments.sh 12e6 0
```

`ABLATIONS=0` runs only baseline, full Cradle, and listen-only. `JUDGE=` skips the TinyStories-33M judge.

Ablation flags on `--recipe cradle`: `--no-curriculum`, `--no-prior`, `--no-babble`, `--no-sleep`. Variant flags: `--babble-mode best-of-k`, `--cds-mix 0.07`, `--phases LISTEN_CDS LISTEN_STORIES BABBLE SLEEP`.

What the runs already show (held-out perplexity, lower is better; two-seed means):

| Recipe | Stories ppl | CHILDES ppl |
|---|---|---|
| Baseline | 11.32 | 37.8 |
| Listen-only, natural mix (`--no-babble --no-sleep`) | 11.09 | 34.3 |
| Listen-only, matched mix (`--no-babble --no-sleep --cds-mix 0.07 --phases 0.1 0.5 0.25 0.15`) | 10.87 | 43.9 |
| Full Cradle (babble + sleep) | 11.60 | 34.2 |

Use listen-only. Add the matched-mix flags only when story perplexity is the only thing that matters (CHILDES gets worse). Details, learning curves, and why babble and sleep failed are in `docs/slm-design.md`.

### Where Cradle results live

| Path | What it is |
|---|---|
| `runs/<name>/model.pt` | Checkpoint. Gitignored. `<name>` is `--name`, or the recipe name if `--name` is omitted. |
| `runs/<name>/log.json` | Per-run log, committed for the published comparison. |
| `runs/<name>.train.txt` | Training stdout. |
| `runs/results.json` | Aggregated metrics (`slm.evaluate --out`, `slm.report`). |
| `runs/report.md` | Markdown table and sample continuations. |
| `runs/judge_noise.json` | Resampled TinyStories-33M judge noise. |
| `docs/slm-design.md` | Design, literature, and the written analysis. |
| `README.md` | Short version of the same experiment. |

## Role-play server (`rp/`)

SillyTavern sends the whole chat on every request. The server logs each generation in SQLite, matches later history back to those generations, and records a learning signal when the user edits a reply, swipes, or types a command. Sleep trains a new LoRA from the active signals and hot-reloads it.

Default model id shown to SillyTavern: `qwen2.5-0.5b-rp`. Override the Hugging Face id with `--model` or `RP_MODEL`. Default data directory: `rp_data/` (gitignored). Default bind: `127.0.0.1:8619`.

```bash
python -m rp.server --port 8619
```

First start downloads `Qwen/Qwen2.5-0.5B-Instruct` into the Hugging Face cache. Inference is bfloat16 on CPU. The published held-out eval ran at about 17 tokens/s.

SillyTavern, on the same machine: API type Chat Completion, source Custom (OpenAI-compatible), endpoint `http://127.0.0.1:8619/v1`, model `qwen2.5-0.5b-rp`. Streaming is on. Context 4096, max reply tokens 250, temperature 0.7, top-p 0.9 match `rp.st_setup`.

If SillyTavern has been started once so `~/SillyTavern/data/default-user/settings.json` exists, close its browser tabs and run:

```bash
python -m rp.st_setup --st ~/SillyTavern --user muzi
```

That patches the settings and installs the 苏小糖 card. The card alone:

```bash
python -m rp.make_card rp/characters/su_xiaotang.json --install ~/SillyTavern/data/default-user/characters
```

### How learning works

Signals land in `rp_data/rp.db` (`generations` and `signals`). Kinds:

| Kind | How it happens | Preferred (`chosen`) | Rejected |
|---|---|---|---|
| `edit` | The user changes a reply in SillyTavern and continues | The edited text | The model text |
| `swipe` | Regenerate / swipe. Once one swipe is kept, it becomes the preferred side | Kept swipe, if any | Swiped-away replies |
| `good` | `!good` or `!好` | The previous reply | |
| `bad` | `!bad` or `!差` | | The previous reply |
| `fix` | `!fix <text>` or `!改 <text>` | The given text | The previous reply |

Other commands, typed as a user message (SillyTavern already uses `/`): `!sleep` / `!睡觉` starts sleep, `!status` / `!状态` prints stats. The server answers with a line starting `〔学习〕` and strips both the command and that acknowledgement from later prompts.

Whitespace-only differences are not edits. A turn is identified by the last non-system messages, so system-prompt and world-info changes do not split the turn.

Sleep (`python -m rp.sleep`, or from chat, or `python -m rp.admin sleep --wait`) trains a **fresh** rank-16 LoRA on the base model from active signals (SFT on preferred replies, DPO on preferred/rejected pairs, an unpaired undesirable term on rejected-only replies). Defaults: 4 epochs, lr 2e-4, at most 15 minutes, at most 400 examples. Adapters are written to `rp_data/adapters/vN` and never overwritten. `rp_data/adapters/current.json` is the active version. A canary (the base model's answers to generic prompts, `rp_data/canary.json`) blocks activation if NLL drifts by more than 0.5 nats/token. Roll back with `python -m rp.admin rollback`.

Auto-sleep is on by default: 10 new signals and 300 seconds without a chat request (`--auto-sleep-signals 0` turns it off).

Dashboard: `http://127.0.0.1:8619/`. Admin CLI (server must be running; `RP_SERVER` overrides the URL):

```bash
python -m rp.admin stats
python -m rp.admin sleep --wait
python -m rp.admin adapters
python -m rp.admin rollback          # previous version, or: rollback v1
python -m rp.admin good
python -m rp.admin bad
python -m rp.admin fix "更符合角色的回复"
```

HTTP equivalents: `GET /v1/models`, `POST /v1/chat/completions`, `GET /rp/stats`, `GET /rp/signals`, `POST /rp/feedback`, `POST /rp/sleep`, `GET /rp/sleep`, `GET /rp/adapters`, `POST /rp/activate`, `POST /rp/reload`.

### Scripted demo and held-out eval

Start the server first. `rp.demo teach` speaks to it the way SillyTavern does: stream a reply, replace it in the next history with the edited line from `su_xiaotang_demo.json`, and let the server record the edit. It does not write signals itself.

```bash
python -m rp.demo teach --chats 0 1
python -m rp.evaluate --versions base v1 --samples 3 --out runs/rp-demo/eval
```

`rp.evaluate` scores held-out prompts that were not in the teach chats. `gold_nll` is the mean per-token NLL of a reference in-character reply (lower is better). Persona rates are the share of samples that use 本座, say 书生, contain an `*action*`, and avoid assistant-speak. The same sample seeds are used for every version.

The committed before/after on that demo (`runs/rp-demo/eval-v1-cli.md`, 10 prompts × 3 samples) moved gold NLL from 3.60 to 3.18 and persona score from 0.42 to 0.86 after one sleep on the scripted edits. Treat that as the result of that run, not as a number to hit on every machine.

Tests that do not download a model:

```bash
python -m unittest tests.test_rp_learn
```

### Where role-play results live

| Path | What it is |
|---|---|
| `rp_data/rp.db` | Generations and signals. Gitignored. Created on first server start. |
| `rp_data/adapters/vN/` | LoRA weights plus `meta.json`. Gitignored. |
| `rp_data/adapters/current.json` | Active adapter version. |
| `rp_data/sleep.log` | Sleep subprocess log. |
| `rp_data/canary.json` | Base-model canary replies used by the drift guard. |
| `runs/rp-demo/eval-base.md` and `.json` | Base model only, before teaching. |
| `runs/rp-demo/eval-v1-cli.md` and `.json` | Base vs `v1` after the scripted teach and sleep. |
| `runs/rp-demo/teach.log` | Stdout of the scripted teach session. |

## Layout

```
slm/data.py        corpus download, tokenizer, curriculum scores
slm/model.py       tiny GPT
slm/caregiver.py   n-gram critic, discriminator, recaster
slm/train.py       Cradle phases and the matched baseline
slm/evaluate.py    perplexity, Zorro, optional TinyStories-33M judge
slm/chat.py        sample from a checkpoint
slm/report.py      rebuild runs/report.md from runs/results.json
rp/server.py       FastAPI: /v1 chat completions and /rp admin
rp/learn.py        SillyTavern history -> signals and !commands
rp/store.py        SQLite
rp/engine.py       Qwen2.5 + PEFT, streaming, hot-reload
rp/sleep.py        LoRA training and the canary guard
rp/admin.py        CLI over /rp/*
rp/demo.py         scripted edit sessions
rp/evaluate.py     held-out before/after
rp/st_setup.py     point a local SillyTavern install at this server
rp/make_card.py    character JSON -> SillyTavern PNG card
rp/stprompt.py     default SillyTavern system prompt
rp/characters/     苏小糖 card, portrait, and demo script
docs/slm-design.md Cradle writeup
scripts/run_experiments.sh
tests/test_rp_learn.py
```

## Git

Repo-local identity for this checkout:

```bash
git config --local user.name "How1Come"
git config --local user.email "118341976+How1Come@users.noreply.github.com"
```

Confirm with `git log --format='%an %ae'`. The only line that should appear is `How1Come 118341976+How1Come@users.noreply.github.com`.
