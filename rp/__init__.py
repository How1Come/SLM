"""rp: a Chinese role-play chat server (OpenAI-compatible, for SillyTavern) that learns from chatting.

Modules:
  store    SQLite log of generations and learning signals (edits, swipes, explicit feedback)
  learn    turns a SillyTavern request history into learning signals
  engine   Qwen2.5 + PEFT LoRA inference on CPU, streaming, adapter hot-reload
  sleep    trains a LoRA adapter from accumulated signals (SFT + DPO + unpaired "undesirable" loss)
  server   FastAPI app: /v1/models, /v1/chat/completions, /rp/* admin endpoints
  admin    small CLI over the /rp/* endpoints
  evaluate held-out before/after measurement for a character
"""

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATA_DIR = "rp_data"
DEFAULT_PORT = 8619
