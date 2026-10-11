# SLM · Roleplay Studio

A CPU-friendly Chinese roleplay assistant that learns from your corrections, preferred replies and swipe choices. Chat in the web UI or SillyTavern, review what the model should learn, then run short, resumable LoRA training batches.

| Task | Where to do it |
|---|---|
| Chat during a lecture break | **对话 / Chat**: start the model service, add a character and send a message |
| Teach a better reply | **Chat**: like a reply, edit it, or choose a regenerated alternative |
| Keep the session quiet | **设置 / Settings**: choose **只收集反馈 / Collect only** |
| Learn when you are idle | **Settings**: choose **空闲时学习 / Idle** |
| Learn while continuing to chat | **Settings**: choose **后台短时学习 / Background**, allowing for slower CPU replies |
| Inspect training | **训练 / Training**: real loss, step, learning rate, elapsed time, checkpoints and saved logs |
| Remove bad examples | **学习记录 / Review**: exclude or restore a feedback record |
| Resume or roll back | **Training**: resume a complete checkpoint; **模型 / Models**: choose an older version or the base |
| Save the result | **Models**: export the adapter; **Review**: export the feedback dataset |

Learning happens in training batches. A single correction can be memorized without improving unfamiliar scenes. Keep reviewing examples and comparing held-out prompts. Negative feedback alone does not teach a replacement reply.

## Start locally

Use **Python 3.12**. The default model is `Qwen/Qwen2.5-0.5B-Instruct`. First model startup downloads its pretrained weights; allow persistent storage and enough RAM for both inference and training. For a cloud CPU instance, start with **8 GB RAM**, **4 CPU cores** and **10 GB free disk**; simultaneous inference/training needs more memory and disk grows with retained checkpoints. These are planning allowances, not a guarantee for every model or prompt length.

```bash
git clone https://github.com/How1Come/SLM.git
cd SLM
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m rp.ui
```

Open [http://127.0.0.1:8847](http://127.0.0.1:8847). Click **启动对话服务 / Start chat service** and wait for **已就绪 / Ready**. Set a character or scene, then chat. The training UI stays usable while the model loads.

The dependency file pins direct runtime versions. Linux CPU uses PyTorch **2.6.0**; Intel macOS uses **2.2.2** because newer PyTorch wheels are unavailable for that architecture. Tiny-model tests exercise real training, adapter reload and exact pause/resume behavior without downloading an LLM.

## Learn during spare time

1. Choose **Collect only** during class to retain feedback without automatic training.
2. Like useful replies or write a stronger correction. Rejection flags identify problems; corrections provide the behavior to imitate.
3. Open **Review** and exclude mistaken feedback. Legacy feedback created before session isolation is retained for review and starts excluded.
4. Choose **Idle** for automatic short batches after enough fresh feedback and a pause in chatting, or click **现在学习 / Learn now**.
5. Watch **Training**. A time limit pauses at a safe optimizer boundary and saves a checkpoint. Automatic learning continues that checkpoint on its next eligible pass; **Resume** also continues it manually.
6. Compare new conversations with fixed prompts. Choose **base**, **narration** or an older chat adapter if the latest version is worse.

Default automatic learning waits for **10 fresh signals** and **300 seconds idle**, uses **1 training thread**, trains for **2 epochs** and allows **5 minutes per batch**. Settings persist under the selected data directory. **Background** mode can consume CPU while you chat; **Idle** is the better starting choice on small machines.

Turning off automatic learning in the UI keeps feedback collection enabled. If you edit or exclude a training example while a batch is running, that batch's adapter is saved for inspection but cannot activate automatically; the updated feedback remains pending for a new batch.

Read the [beginner guide](docs/chat-learning.md) for cloud access, feedback interpretation, checkpoints, export, troubleshooting and API examples.

## Run on a cloud CPU instance

The UI listens on loopback by default. For remote access, generate an access token, expose only the authenticated UI behind HTTPS, and keep the model worker private.

```bash
cp .env.example .env
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Paste the generated value into `RP_AUTH_TOKEN` in `.env`. Set `RP_HOST=0.0.0.0` only when using a firewall and TLS reverse proxy. Set `RP_SECURE_COOKIE=1` for HTTPS browser access. Start using the launcher:

```bash
source .venv/bin/activate
sh scripts/launch-rp.sh
```

The UI-managed model worker listens on **127.0.0.1:8627**. The trainer UI listens on **8847**. A standalone `python -m rp.server` uses **8619**; use its port only for a deliberately separate server. Do not expose model, training-control or log endpoints unauthenticated.

For private access without a public endpoint, leave the UI on loopback and use an SSH tunnel:

```bash
ssh -L 8847:127.0.0.1:8847 your-user@your-server
```

Open `http://127.0.0.1:8847` on your own computer. Remote browser cookies require `RP_SECURE_COOKIE=0` for this HTTP tunnel.

## Run with Docker

The CPU image runs as UID **10001**, serves the UI with Waitress, requires an access token, and stores runtime files in a persistent volume. Docker publishes only the UI, bound to loopback on the host. Start it behind an SSH tunnel or a TLS reverse proxy.

```bash
cp .env.example .env
python -c 'import secrets; print(secrets.token_urlsafe(32))'
docker compose up --build -d
docker compose logs -f
```

Set the generated token in `.env` before `docker compose up`. Use the UI to stop active training and wait for a completed checkpoint before restarting the container:

```bash
docker compose down
docker compose up -d
```

`docker compose down` retains the named data volume. Removing that volume deletes chat feedback, adapters, checkpoints, job history and model cache. Back it up before moving or replacing the server. The image is designed for a single-user, single-controller service; it does not require a GPU or the NVIDIA runtime.

## Connect SillyTavern

| Setting | Value |
|---|---|
| API | Chat Completion |
| Source | Custom / OpenAI-compatible |
| Endpoint | Your trainer UI origin followed by `/v1`, such as `http://127.0.0.1:8847/v1` |
| Model | `qwen2.5-0.5b-rp` |
| API key | `RP_AUTH_TOKEN` when access is protected |
| Character | Your card, or [苏小糖](rp/characters/su_xiaotang.json) |

Send a distinct conversation identifier with each chat where the client supports `session_id`, `metadata.conversation_id`, `user` or `X-RP-Session`. The built-in UI supplies one automatically. Without one, the server falls back to the system prompt identity; independent chats using the same card cannot be reliably distinguished.

SillyTavern edits and kept swipe alternatives become feedback when its updated history reaches the server. `!good`, `!bad`, `!fix <better reply>`, `!sleep` and `!status` provide explicit hooks. See the [feedback guide](docs/chat-learning.md#teach-from-chat) before starting automatic learning.

## Repository layout

| Path | Purpose |
|---|---|
| `rp/ui/` | Web interface, persistent job journal, subprocess orchestration |
| `rp/server.py` | OpenAI-compatible model API and automatic training scheduler |
| `rp/learn.py`, `rp/store.py` | Scoped feedback detection, SQLite persistence and review |
| `rp/settings.py` | Atomic learning settings |
| `rp/engine.py` | CPU inference, pristine base selection and adapter loading |
| `rp/sleep.py` | Chat LoRA training, safety checks and checkpoint/resume |
| `rp/train_narration.py` | Supervised narration training |
| `data/rp-narration/` | Project-authored Chinese training examples and held-out prompts |
| `rp/bundled_adapters/narration/` | Bundled narration adapter |
| `tests/` | Offline tiny-model, feedback, persistence and API regression tests |
| `slm/` | Separate from-scratch English Cradle experiment |

The current release repairs and extends the CPU roleplay workflow. GPU QLoRA, arbitrary dataset/project management, external AI generation and multi-worker orchestration remain future modules. Adapters are exported for the matching Transformers/PEFT base; GGUF and other conversions are unavailable.

## Verify the installation

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest tests -q
node --check rp/ui/static/app.js
sh -n scripts/launch-rp.sh
```

Node is needed only for the JavaScript syntax check. It is not needed to serve the application.

For a focused training smoke test with real tiny-model optimizer steps, run `python -m pytest tests/test_sleep.py tests/test_adapters.py -q`.

## Historical Cradle experiment

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

The design rationale, the literature behind it, and measured results are in
[`docs/slm-design.md`](docs/slm-design.md). A short version of the results is at the bottom of this file.

## Run the Cradle experiment

Use the Python environment from the installation above. This downloads the public corpora and trains the separate English tiny model:

```bash
python -m slm.data
python -m slm.train --recipe cradle --no-babble --no-sleep --name listen
python -m slm.train --recipe cradle --name cradle
python -m slm.train --recipe baseline --name baseline
python -m slm.evaluate runs/listen/model.pt runs/cradle/model.pt runs/baseline/model.pt
python -m slm.chat --checkpoint runs/listen/model.pt
```

For a smoke test, add `--budget-tokens 1e6` to training commands. The experiment's recommended recipe remains **listen-only**. The full comparison runs with:

```bash
sh scripts/run_experiments.sh 12e6 0
```

Ablation flags for `slm.train --recipe cradle`: `--no-curriculum`, `--no-prior`, `--no-babble`, `--no-sleep`. Variant flags: `--babble-mode best-of-k`, `--cds-mix 0.07`, `--phases LISTEN_CDS LISTEN_STORIES BABBLE SLEEP`.

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

## Results (4-core CPU, 12M-token budget, about 14 minutes per run)

These are held-out perplexities, so lower is better. Two-seed rows show the mean.

| Recipe | Stories ppl | CHILDES ppl | Zorro | Command |
|---|---|---|---|---|
| Baseline (2 seeds) | 11.32 | 37.8 | 64.9% | `--recipe baseline` |
| Cradle v1, all 4 phases | 11.60 | 34.2 | 65.4% | `--recipe cradle` |
| **Listen-only Cradle, natural mix** (2 seeds) | **11.09 (−2.0%)** | **34.3 (−9%)** | 64.9% | `--recipe cradle --no-babble --no-sleep` |
| **Listen-only Cradle, matched mix** (2 seeds) | **10.87 (−4.0%)** | 43.9 (+16%) | 64.4% | `... --no-babble --no-sleep --cds-mix 0.07 --phases 0.1 0.5 0.25 0.15` |

What this shows:

- The **listening** part of the recipe helps at the same compute: age-ordered child-directed speech first,
  then simple-to-complex stories with a growing context, plus a bigram prior that fades out. The gain is
  consistent across seeds, and seed noise is about 0.3%.
- The **babble + feedback** self-improvement loop and the **sleep** (EMA self-distillation) phase both
  *hurt* perplexity in this setup. The likely reason is that the caregiver critic only knows the same small
  corpus, so it adds no new information.
- Generation quality, as judged by the independent TinyStories-33M model, is a tie within noise.

The full analysis, ablations, learning curves, and next experiments are in
[`docs/slm-design.md`](docs/slm-design.md). Every
run's raw numbers are in `runs/results.json` and `runs/report.md`.

## Open the Cradle trainer

The separate Cradle UI offers its experiment presets, logs and prompt continuation:

```bash
python -m slm.ui
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). This UI trains the English tiny model. Use **Roleplay Studio** for Chinese roleplay feedback learning.
