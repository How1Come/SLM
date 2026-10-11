# CPU chat-learning release

Roleplay Studio now supports a verified chat → correction → review → CPU LoRA training → checkpoint/resume → adapter export workflow. The interface separates chatting, training, learning review, model versions and settings.

## Install the supplied source archive

The repaired branch can be installed from `SLM-reliable-chat-learning.zip` without GitHub authentication:

```bash
unzip SLM-reliable-chat-learning.zip
cd SLM
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m rp.ui
```

Use this supplied archive for the repaired release when the branch has not been pushed to GitHub. For cloud access and persistence, follow the [chat-learning guide](chat-learning.md).

## Use it around lectures

1. Start `python -m rp.ui`, open `http://127.0.0.1:8847`, then start the chat service.
2. Choose **Collect only** during class. Chat feedback is retained without automatic training.
3. Like useful replies or edit weak replies into better examples. Dislikes alone cannot teach a replacement.
4. Review the examples and exclude mistaken feedback.
5. Choose **Idle** for short batches between sessions, or **Background** when CPU/RAM capacity allows chatting and training together.
6. Resume paused jobs from **Training history**. Automatic learning also continues its paused checkpoint on the next eligible pass.
7. Compare unfamiliar prompts and roll back in **Models** when the result is worse.

Automatic learning defaults to ten fresh signals, five minutes idle, one training thread, two epochs and a five-minute training allowance. Turning off automatic learning in the UI retains feedback collection. The model changes after a training batch completes and activates; each individual message does not guarantee an improvement.

## Reliability repairs

| Area | Result |
|---|---|
| Chat isolation | Distinct conversation IDs prevent independent chats from sharing inferred feedback; legacy unscoped records are retained for review |
| Feedback integrity | Reverted edits and changed swipe choices update the training signals; unresolved swipes remain audit records |
| Review integrity | Excluded examples stay excluded when history is resent |
| Background updates | Editing or withdrawing a signal in a running snapshot blocks automatic activation and leaves the changed feedback pending |
| Model versions | Base selection restores pristine pretrained weights; training and serving use the same immutable narration foundation |
| Adapter storage | Saved versions load correctly, retain foundation metadata and support export and rollback |
| Recovery | Complete checkpoints preserve optimizer, scheduler, random state, progress and the original dataset snapshot |
| Monitoring | Actual training metrics, persistent job history and separate job logs survive browser reconnection |
| Access control | Loopback worker, authenticated remote UI and validated paths |

## Verification

- **102 tests passed**, covering feedback, persistence, API behavior, authentication, adapters, training and recovery.
- **48 installed packages passed dependency compatibility checks**.
- A locally constructed tiny model completed real HTTP chat, correction/review and one-step adapter training.
- A browser narration job paused at **81/126 steps**, resumed at **82**, then completed at **126/126**.
- The interface was inspected at a **390 px mobile viewport**.
- Adapter export, rollback and version validation were checked.

Tiny-model tests exercise real optimizer mechanics and exact recovery. Their outputs do not establish full-sized roleplay quality. A drift check catches some regressions and does not replace human evaluation on held-out scenes.

## Default Qwen CPU smoke test

The full default `Qwen/Qwen2.5-0.5B-Instruct` workflow also passed locally on **Intel macOS, Python 3.12 and PyTorch 2.2.2 CPU**. It used one reviewed short example and rank-2 LoRA to check real training mechanics.

| Measurement | Result |
|---|---|
| Weight download | 988,097,824 bytes in 32.8 seconds |
| Initial narration-backed reply | 2.50 seconds; process RSS 2.11 GiB |
| One optimizer step | 3.55 seconds; 19.74 tokens/second |
| Full training stage | 101.49 seconds, including six generated regression checks and evaluation |
| Saved `v1` reload and reply | Reload 11.23 seconds; reply 3.98 seconds; process RSS 2.15 GiB |
| Download through checkpoint and reload | 191.79 seconds |
| Export | 19,585,946 bytes, including the immutable narration foundation; 1.91 additional seconds |
| Checkpoint verification | Restored step 1, optimizer, scheduler, random state, exact dataset, configuration and foundation |

These timings describe this short local smoke test, not a larger dataset or cloud performance guarantee. It confirms generation, weight updates, atomic checkpointing, reload and export. It does not establish improved roleplay quality from one example.

## Deployment status and scope

Docker/Compose, pinned CPU dependencies, an environment-aware launcher and cloud-access instructions are included. **The Docker image and remote deployment were not verified in this development session.** Keep persistent storage, use an access token and HTTPS or SSH forwarding, and validate the deployment before relying on it during a lecture.

This release covers the existing CPU roleplay application. GPU QLoRA, external AI dataset generation, arbitrary project/dataset management and distributed workers remain future modules.

Installation and day-to-day guidance: [README](../README.md), [chat-learning guide](chat-learning.md).
