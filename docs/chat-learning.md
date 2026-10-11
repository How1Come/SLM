# Chat, teach and train during spare time

Roleplay Studio runs a pretrained Chinese roleplay model on CPU. You provide the character card and scene at chat time. Feedback teaches preferences across a variety of characters and situations, then short LoRA training batches change model weights.

The browser is a control surface. SQLite feedback, adapters, checkpoint snapshots, logs and job history live on the server. Closing a browser tab does not stop the server or an active job. Losing the server's disk destroys unbacked-up runtime data.

## Install and start

Use Python 3.12. Linux CPU uses pinned PyTorch 2.6.0; Intel macOS uses 2.2.2. Direct dependencies are pinned in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m rp.ui
```

Open `http://127.0.0.1:8847`. The first screen works before any model is loaded. Click **启动对话服务 / Start chat service** and wait for **已就绪 / Ready**. Initial startup downloads `Qwen/Qwen2.5-0.5B-Instruct`; later launches reuse the model cache.

Plan for 8 GB RAM, four CPU cores and 10 GB initial free disk on a Linux cloud machine. Background inference plus training retains more than one model in memory; use 12–16 GB RAM or train while the inference service is stopped on a small machine. Training time depends on the CPU, sample length and amount of feedback. Each retained checkpoint and adapter consumes additional disk.

| Screen | Action |
|---|---|
| **对话 / Chat** | Start the service, create a scene, chat and give feedback |
| **训练 / Training** | Choose chat or narration training, inspect real metrics, stop and resume |
| **学习记录 / Review** | Compare chosen/rejected replies and exclude or restore examples |
| **模型 / Models** | Select base, narration or a saved chat version; export an adapter |
| **设置 / Settings** | Save the learning schedule, CPU limit, character card and reply length |

Set the character and scenario under **Settings**. Changing the character starts a new conversation ID, keeping independent chats isolated. Character settings and visible chat history are stored in the current browser; learning feedback and job history are stored on the backend.

## Teach from chat

| Feedback | Meaning | Training use |
|---|---|---|
| **喜欢 / Like**, `!good` | Keep a useful generated reply | A weaker positive example |
| **修改并教给模型 / Edit**, `!fix <reply>` | Provide a better reply in your own words | A strong preferred example and a preference pair |
| **不喜欢 / Dislike**, `!bad` | Flag an undesirable reply | A negative term alongside preferred examples |
| Regenerate, then keep an alternative | Prefer the kept reply over an earlier swipe | A weak preference pair |
| Regenerate without choosing a reply | Request another option | An audit record only; no training signal yet |

A dislike-only dataset cannot teach what to say instead. Add corrections or preferred examples before starting chat training. Likes reinforce the model's existing behavior; corrections provide new information. Keep examples that describe the behavior you want across different prompts.

In the built-in UI, feedback buttons target the exact generation. Corrections are saved immediately. Automatic inference does not treat every unedited reply as a training example.

SillyTavern resends conversation history. Edits and final swipe selections are detected when that updated history reaches the server. Returning to an earlier swipe or reverting an edit updates the active feedback instead of keeping contradictory training pairs. An unresolved new regeneration clears stale inferred preferences for that turn.

Explicit SillyTavern commands are removed from later model prompts:

| Command | Effect |
|---|---|
| `!good`, `!好` | Prefer the previous reply |
| `!bad`, `!差` | Reject the previous reply |
| `!fix 更合适的角色回复`, `!改 更合适的角色回复` | Supply a corrected previous reply |
| `!sleep`, `!睡觉` | Request a training batch |
| `!status`, `!状态` | Inspect learning counts and the active adapter |

## Fit learning around a lecture

Start with **Collect only** during a lecture and **Idle** between sessions. Both retain useful feedback. Automatic mode changes when training starts; it does not guarantee that the next reply has improved.

| Mode | Trigger | CPU behavior |
|---|---|---|
| **只收集反馈 / Collect only** | A manual **Learn now** or **Start training** action | Chat runs normally; automatic training is disabled |
| **空闲时学习 / Idle** | Enough new feedback, then the configured idle period | Training starts after a pause in chatting |
| **后台短时学习 / Background** | Enough new feedback, without waiting for idle | Training shares CPU/RAM with inference; replies can become slower |

The automatic-learning toggle controls scheduling. Turning it off in the UI keeps feedback collection enabled, and **Learn now** remains available. Selecting **Collect only** also disables automatic passes without deleting collected feedback.

Default automatic settings:

| Setting | Default | Meaning |
|---|---|---|
| Fresh feedback threshold | 10 | Collect enough new feedback before starting |
| Idle delay | 300 seconds | Wait five minutes after the last chat request |
| Maximum batch time | 5 minutes | Pause safely when the time allowance is used |
| Epochs | 2 | Visit the training examples twice |
| Training threads | 1 | Leave CPU capacity available for other work |
| Maximum canary drift | 0.3 | Prevent automatic activation when the regression check worsens too far |

A time allowance includes real training work; model loading and regression evaluation can add overhead. A running optimizer step finishes before a graceful pause, so the limit is not a hard wall-clock kill. Checkpointing has a storage cost. **Idle** and short replies are the better starting settings on a small CPU server.

If an automatic batch pauses at its time allowance, the next eligible automatic pass resumes its complete checkpoint. It preserves optimizer/scheduler state and the same dataset snapshot instead of beginning again. Eligibility still depends on the saved learning mode, feedback threshold, idle rule and scheduler cooldown. You can resume the checkpoint manually from **Training history** instead.

A new chat-learning batch rebuilds a fresh adapter from the same frozen base/narration foundation, replaying the currently included feedback up to the trainer's example cap. It does not repeatedly stack newly trained adapters. Fresh feedback triggers learning; earlier included feedback preserves previous lessons.

Use **Training → Quick** for a short manual test, **Balanced** for a longer break, and **Thorough** for a longer idle window. Expand **Advanced** to inspect epochs, time allowance, CPU threads and narration learning rate. Chat-learning guard settings are saved under **Settings**.

## Review data before training

Open **Review** and expand each record to see its preferred and rejected text. Excluding a record prevents its use in future fresh training passes and keeps the audit history. Restoring it makes it eligible again. Exclusion cannot remove influence already present in a currently active adapter; rebuild from reviewed feedback or choose an earlier adapter.

Old feedback from the pre-isolation schema is preserved with `needs_review` status and starts excluded. It may have associated a reply with the wrong character or chat. Read the saved context before restoring it. New requests never infer additional feedback by matching those old unscoped generations.

Train on diversity:

- Include characters with different personalities, relationships and vocabularies.
- Include quiet dialogue, conflict, emotional transitions, action and scene changes.
- Include short and longer conversations, with both narration and dialogue.
- Include initiative while preserving the user's control of their own actions.
- Include continuity across turns and clear writing perspective.

Reject repeated openings, copied phrases, broken formatting, contradictory character behavior and examples that control the user's actions. Ten carefully written corrections can teach more than hundreds of repeated poor replies. Training only one character is appropriate only when that is your intended specialization.

**Export learning data** downloads JSONL with normalized `messages`, provenance/review metadata and preference pairs when available. The export includes excluded audit records; filter `metadata.status` before using it in another trainer.

## Understand training and checkpoints

| Term | Meaning |
|---|---|
| Base model | The pretrained language model before your roleplay adapters |
| LoRA | Small additional trainable weights attached to the base model |
| SFT | Learning to produce a supplied preferred response |
| Preference training | Learning to prefer a chosen response over a rejected response |
| DPO | A preference objective used by chat training when a pair is available |
| Context length | The number of tokens the model can consider in one request |
| Epoch | One visit through the training examples |
| Checkpoint | Saved adapter weights plus the state needed to continue the same job |
| Token | A piece of text; Chinese characters, words and punctuation can span different numbers of tokens |
| QLoRA | LoRA using a quantized base model to reduce GPU memory; a future backend in this repository |

The training monitor displays actual optimizer metrics: steps, loss, learning rate, throughput, elapsed time and an estimate of time remaining. Estimated time varies during loading, checkpoint saves and evaluation. Logs remain downloadable after the job finishes.

A complete checkpoint saves adapter weights, optimizer and scheduler state, shuffle/random state, epoch/cursor, software versions, configuration and an immutable dataset snapshot. Chat checkpoints also preserve their exact frozen narration foundation. New feedback collected while a job runs belongs to a later batch; it is not silently inserted into a resumed job. An edit made during training remains unconsumed if the job trained an older version of that record.

Editing or withdrawing a signal already in a running job's snapshot blocks automatic activation of that job's adapter. Its metadata records `dataset_changed_since_start`; the previous model remains active, the candidate stays available for inspection and the changed feedback remains pending. Start a fresh batch to train the current reviewed examples. Adding unrelated new feedback does not alter the in-flight snapshot.

## Stop, reconnect and resume

1. Click **保存并停止 / Save and stop**.
2. Wait until the job is **Paused** and displays a checkpoint.
3. Keep the server data directory or Docker volume intact.
4. Restart the UI if needed.
5. Select the paused/interrupted job in **Training history**.
6. Click **从检查点恢复 / Resume from checkpoint**.

The previous active model remains served until a new adapter completes and passes the activation check. A browser disconnect does not interrupt backend training. A server restart marks in-flight jobs interrupted; the last complete checkpoint remains usable. A hard process kill can lose work since the latest checkpoint, so stop gracefully before shutting down a cloud VM.

Resume uses the original snapshot, including its epochs, seed and model foundation. It is not a way to switch datasets or model architecture midway through training. Start a new job for different settings. A missing or modified checkpoint produces a clear error instead of silently restarting from scratch.

The CLI also accepts a checkpoint directory:

```bash
python -m rp.sleep --help
python -m rp.train_narration --help
```

Use the saved job's checkpoint path with `--resume` and the same `--data-dir` or narration `--out` settings. Only load checkpoints you created or trust; optimizer state is part of the saved training artifacts.

## Check improvement and roll back

The activation guard compares fixed check responses before and after training on the same foundation. It detects some regressions. It does not score immersion, emotional continuity or all character-card behavior. A saved adapter may fail this guard and remain available for inspection without replacing the active version.

Keep fixed prompts outside the training examples. Compare the base, narration and chat adapter on:

| Dimension | Question |
|---|---|
| Character consistency | Does the reply follow the card's personality and perspective? |
| Immersion | Does the scene feel coherent and specific? |
| Dialogue and prose | Does the wording sound natural and varied? |
| Emotional continuity | Does the response preserve the established emotional state? |
| Initiative | Does the model contribute a useful next action? |
| User autonomy | Does it avoid deciding the user's actions or feelings? |
| Scene continuity | Does it preserve relevant facts from the recent conversation? |

Use **Compare base and narration** for the bundled held-out narration evaluation. Inspect the generated responses and individual metrics; a lower training loss alone is not proof of better roleplay. To compare a chat adapter, run the same held-out prompts while choosing different saved models.

Open **Models**, choose an earlier version or **base**, and wait for the server to activate it. Base selection reloads pristine pretrained weights. Saved narration and chat weights remain on disk. A change of adapter does not erase feedback or job history.

Historical roleplay evaluation numbers were collected before repairing baseline isolation. They remain historical artifacts; rerun comparisons with the repaired engine before using them to claim improvement.

## Export and keep your work

**Models → Export** downloads an adapter ZIP. A chat adapter includes metadata and the immutable narration foundation used during its training. Keep those files together: the adapter needs the matching base model and foundation. It is not a standalone full model.

| Storage item | Purpose |
|---|---|
| `rp.db` | Generation context, replies, feedback and review state |
| `learning.json` | Saved learning schedule |
| `.ui-secret` | Backend browser-session signing key |
| `adapters/current.json` | Selected adapter version |
| `adapters/narration/` | Narration adapter |
| `adapters/vN/` | Saved chat adapter, metadata and foundation snapshot |
| `checkpoints/` | Complete resumable training snapshots |
| `jobs/` | Persisted UI/automatic job records and logs |
| `server.log` | Current model-service log |
| `sleep.log` | Retained legacy automatic-learning log, when present |
| Hugging Face cache | Cached pretrained model/tokenizer downloads |

Paths are relative to the configured data directory unless a cache is configured elsewhere. Back up the entire data directory after gracefully stopping the trainer and model service. SQLite uses WAL, so copying only `rp.db` during live writes can omit uncheckpointed changes.

Each new manual or automatic job writes its own log under `jobs/`, preserving the training history instead of appending all automatic runs to `sleep.log`.

GGUF, Ollama, AWQ/GPTQ and merged full-model export are unavailable. The supplied export preserves a Transformers/PEFT adapter workflow. Character cards and external conversation memory are runtime inputs; they are not bundled into every trained adapter.

## Run privately on a cloud machine

Prefer SSH forwarding for personal use. Start `python -m rp.ui` on the cloud host without a public bind, then run this on your own computer:

```bash
ssh -L 8847:127.0.0.1:8847 your-user@your-server
```

The browser reaches the cloud UI at `http://127.0.0.1:8847`. The model and training stay on the cloud machine. Loopback traffic never requires opening the model worker port to the internet.

For remote HTTPS access:

1. Copy `.env.example` to `.env`.
2. Generate a token with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`.
3. Set `RP_AUTH_TOKEN` to that value, using at least 24 random characters.
4. Set `RP_HOST=0.0.0.0` and `RP_SECURE_COOKIE=1`.
5. Route HTTPS to UI port 8847 through your reverse proxy.
6. Restrict direct UI access with the firewall; leave worker port 8627 private.
7. Start `sh scripts/launch-rp.sh` from an activated virtual environment.

| Proxy requirement | Reason |
|---|---|
| TLS certificate | Protect the access token, session cookie and private conversations |
| Forwarded host consistency | Preserve same-origin request checks |
| At least 600-second upstream timeout | Allow long CPU-generated responses |
| Response buffering disabled on streaming routes | Preserve chat and training-event streams |
| One application controller | Avoid independent job managers competing for the same process/storage |

The application is single-user. Its token grants access to conversation logs, training and model management; it is not a read-only share link. Access tokens are supplied through the environment and are never included in exported feedback. Set secure cookies to `0` only for local HTTP or a private SSH tunnel.

## Run the CPU container

The Docker image uses Python 3.12, CPU PyTorch and a non-root user. The named volume retains checkpoints, job records, model cache and feedback through container replacement.

```bash
cp .env.example .env
python -c 'import secrets; print(secrets.token_urlsafe(32))'
docker compose up --build -d
docker compose logs -f
```

Set the printed token in `.env` before startup. The default Compose port mapping is `127.0.0.1:8847:8847`; reach it through SSH or a proxy running on the same host. Docker exposes no model-worker port.

Use **Save and stop** and wait for the checkpoint before `docker compose down`. The named volume survives that command. `docker compose down -v` removes persistent training data; do not use it to perform an ordinary restart.

For an existing bind-mounted storage directory, make it writable by container UID 10001. Never bind-mount a personal home directory as the entire training volume. The default named volume is prepared with the correct ownership.

The UI uses a single Waitress controller with eight request threads. The provided Docker configuration is ready for build verification on Linux. It has not been built in the current development session because a Docker daemon/CLI is unavailable.

## Connect SillyTavern

Start the model through the UI. Configure Chat Completion → Custom / OpenAI-compatible, endpoint `http://127.0.0.1:8847/v1` locally or the HTTPS UI origin followed by `/v1` remotely. Use model `qwen2.5-0.5b-rp`; fill the API key with `RP_AUTH_TOKEN` for protected access.

The UI proxies authenticated model traffic to its loopback worker. API clients need a Bearer token even if a browser is already signed in. Standalone `python -m rp.server` uses port 8619; UI-managed workers use 8627. Do not confuse those ports when using `rp.admin`.

For reliable conversation isolation send one stable, unique ID per chat through one of:

- `session_id` in the request body.
- `metadata.conversation_id` in the request body.
- `user` in the request body when it identifies this conversation.
- `X-RP-Session` in the request headers.

Keep the ID unchanged when editing or trimming that chat. Use a new ID for a new chat or character. If the client sends no conversation ID, the fallback uses system-prompt identity and cannot distinguish independent chats with the same card.

## Use the API

The browser API accepts authenticated JSON requests. The model worker exposes FastAPI OpenAPI at its private `/openapi.json`. Keep worker administration endpoints private and use the UI API for remote management.

| Method | UI route | Purpose |
|---|---|---|
| `GET` | `/api/overview` | Hardware, learning counts, server status and active job |
| `GET` / `PATCH` | `/api/settings` | Retrieve or update learning settings |
| `POST` | `/api/server/start`, `/api/server/stop` | Control the model worker |
| `POST` | `/api/chat` | Non-streaming chat from the UI |
| `POST` | `/api/feedback` | Like, reject or correct an exact generation |
| `GET` | `/api/signals?page=1&page_size=25` | Page through feedback |
| `PATCH` | `/api/signals/{id}` | Include/exclude feedback using `active` |
| `GET` | `/api/signals/export` | Download normalized audit JSONL |
| `POST` | `/api/sleep` | Start chat-feedback training |
| `POST` | `/api/narration/train` | Start supervised narration training |
| `POST` | `/api/narration/eval` | Evaluate base versus narration |
| `GET` | `/api/job`, `/api/jobs` | Active job and persistent history |
| `POST` | `/api/job/stop`, `/api/job/resume` | Stop safely or resume a saved job |
| `GET` | `/api/jobs/{id}/logs` | Download persistent logs |
| `POST` | `/api/adapters/activate` | Choose base, narration or a saved `vN` |
| `GET` | `/api/adapters/{version}/export` | Download an adapter ZIP |
| `GET` | `/api/events` | Stream training updates |
| `GET` | `/v1/models` | List the served model |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat, including streaming |

From an environment where `RP_AUTH_TOKEN` is set:

```bash
curl -H "Authorization: Bearer $RP_AUTH_TOKEN" http://127.0.0.1:8847/api/overview
curl -X PATCH -H "Authorization: Bearer $RP_AUTH_TOKEN" -H 'Content-Type: application/json' -d '{"mode":"collect_only"}' http://127.0.0.1:8847/api/settings
curl -X POST -H "Authorization: Bearer $RP_AUTH_TOKEN" -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8847/api/server/start
curl -X POST -H "Authorization: Bearer $RP_AUTH_TOKEN" -H 'Content-Type: application/json' -d '{"messages":[{"role":"system","content":"你是一位可靠的星际领航员。不要代替用户行动。"},{"role":"user","content":"求救信号来自那颗陌生星球。"}],"session_id":"space-scene-1","max_tokens":128}' http://127.0.0.1:8847/api/chat
```

The returned `rp_generation_id` identifies the reply. Submit `{"gen_id":"the returned ID","correction":"your better reply"}` to `/api/feedback`, or use `rating: "good"` / `rating: "bad"`. A request with invalid parameters produces an actionable validation error; it does not execute arbitrary shell input.

## Troubleshoot

| Symptom | Action |
|---|---|
| Model startup takes time | Keep the UI open and read the server log; first startup downloads the weights |
| Model download fails | Check internet access, model ID/path, provider permissions and free disk; restart the service after correcting the cause |
| Chat is slow | Shorten replies/context, reduce training threads, choose Collect only/Idle, or use a stronger CPU instance |
| No training examples are ready | Like or correct a reply; rejected replies alone do not supply replacement behavior |
| A new adapter is saved but not active | Inspect the drift guard result; evaluate before manually selecting it |
| A candidate records `dataset_changed_since_start` | A signal in its training snapshot was edited or excluded; train a fresh batch from the current reviewed feedback |
| A training job is paused | Resume its complete checkpoint or start a new job with different settings |
| An interrupted job cannot resume | Preserve the checkpoint directory, stop the old trainer, use the same model and verify the snapshot was not modified |
| Storage is nearly full | Stop training, back up older completed artifacts and free space without deleting the active foundation |
| RAM is exhausted | Stop competing jobs, stop inference during training, reduce context length or use more RAM |
| Remote login repeatedly returns to the token dialog | Use HTTPS with secure cookies, or set `RP_SECURE_COOKIE=0` for private HTTP tunneling |
| Public startup refuses to run | Set an access token with at least 24 random characters; keep the worker on loopback |
| Feedback from another chat appears | Ensure each chat sends a unique conversation ID; keep legacy feedback excluded until reviewed |
| Clearing browser chat does not clear learning records | Review and exclude the feedback on the server; browser state and training data are separate |

## Verify changes as a developer

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest tests -q
node --check rp/ui/static/app.js
sh -n scripts/launch-rp.sh
```

Tiny-model tests construct a local Qwen-compatible model and tokenizer, perform real optimizer steps, verify saved adapter reload, compare resumed and uninterrupted weights, and reject modified checkpoints. They require no GPU or model download. They verify training mechanics, not full-sized roleplay quality. UI/API tests exercise lifecycle, auth, validation and persistent review behavior with mockable model services.

For a focused training smoke test using actual tiny-model optimizer steps:

```bash
python -m pytest tests/test_sleep.py tests/test_adapters.py -q
```

The current product is the CPU roleplay workflow described above. External AI dataset generation, arbitrary project/dataset management, GPU QLoRA, cloud-provider orchestration and multi-user accounts remain separate future work.
