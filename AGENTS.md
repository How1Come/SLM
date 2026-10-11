# SLM engineering instructions

## Product scope

- Maintain `rp/` as a Chinese roleplay assistant that fine-tunes a pretrained model through deliberate chat feedback, with a beginner-facing training UI and SillyTavern-compatible API.
- Target a single-user Linux cloud instance with CPU inference and CPU LoRA training. Keep local development supported; keep cloud storage persistent.
- Keep `slm/` as the separate English Cradle experiment. Historical methods and results belong in [docs/slm-design.md](docs/slm-design.md) and [README.md](README.md).
- Preserve the distinction between teaching roleplay behavior and providing a character card at inference time. Changes must not silently lock general training to one character.
- Describe implemented functionality accurately. GPU QLoRA, an external AI dataset studio, full project/dataset management and distributed workers are future work, not available features.

## Module boundaries

| Module | Responsibility |
|---|---|
| `rp/learn.py` | Scoped edit, swipe and explicit-feedback interpretation |
| `rp/store.py` | SQLite generations, reviewable signals and schema migrations |
| `rp/settings.py` | Validated, atomic learning preferences |
| `rp/engine.py` | Pristine base inference, adapter compatibility and activation |
| `rp/sleep.py` | Chat-feedback training, immutable foundations and resumable checkpoints |
| `rp/train_narration.py` | Narration dataset training |
| `rp/server.py` | OpenAI-compatible inference API and automatic learning scheduler |
| `rp/ui/` | Trainer UI, job orchestration and persistent job journal |

## Reliability requirements

- Write no comments in code.
- Preserve session and character isolation. Explicit conversation IDs must survive prompt updates and trimming; fallback matching must never silently reuse unscoped legacy generations.
- Train from deliberate feedback. Unconfirmed swipes must not become negative examples. User review exclusions must survive repeated chat-history scanning.
- Keep reads and writes in short SQLite transactions. Protect edits made during training with signal snapshot versions when marking feedback consumed.
- Save immutable training inputs, parameters, foundation identity, optimizer/scheduler state, random state and progress before offering resume.
- Save checkpoints atomically. On interruption preserve the last complete checkpoint, keep the previous served adapter and report the job state accurately.
- Resume against the original dataset and foundation snapshot. Reject modified snapshots and incompatible checkpoints.
- Serve a pristine base when base is selected. Save adapters in a loadable directory and keep narration foundations identical between training and inference.
- Coordinate manual and automatic jobs with a storage-level training lock. Do not train two jobs against the same output concurrently.
- Display progress from real optimizer metrics. Mark unavailable actions honestly; never simulate successful learning, metrics or exports.
- Treat drift checks as limited regression checks, not proof of better roleplay. Preserve rollback and held-out human evaluation.

## Security and release requirements

- Bind the model worker to loopback. Bind the trainer UI to loopback by default.
- Require `RP_AUTH_TOKEN` with at least 24 characters for a non-loopback UI; use HTTPS and secure cookies for remote browser access.
- Keep the model API authenticated and private. Public deployments must expose only the trainer through a TLS reverse proxy.
- Validate request bodies, paths, adapter versions and subprocess arguments. Use argument arrays; never interpolate browser input into shell commands.
- Keep tokens out of source, logs and exported training data. Persist runtime data outside the application image.
- Keep direct dependency versions pinned and the CPU Docker image non-root.
- Synchronize [README.md](README.md), [docs/chat-learning.md](docs/chat-learning.md), launch options and UI behavior.
- Do not publish, deploy or expose a service without the user's authorization.

## Verification

```bash
python -m pytest tests -q
node --check rp/ui/static/app.js
sh -n scripts/launch-rp.sh
python -m pip check
```

- Use locally constructed tiny models for actual optimizer, adapter and checkpoint tests; tests must not require a GPU or a large model download.
- Cover base rollback, save/reload equivalence, narration foundation identity, pause/resume equivalence, review persistence, session isolation, SQLite concurrency, API validation, authentication and UI process lifecycle.
- Run focused checks while implementing, then run the complete suite before reporting completion.
- Distinguish verified tiny-model behavior, measured full-model results and untested cloud deployment paths in the release notes.
