"""CPU inference: Qwen2.5 chat model + optional PEFT LoRA adapter, token streaming, adapter hot-reload."""
from __future__ import annotations

import json
import os
import shutil
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

BUNDLED_NARRATION = os.path.join(os.path.dirname(__file__), "bundled_adapters", "narration")
NARRATION_VERSION = "narration"


def adapters_dir(data_dir):
    return os.path.join(data_dir, "adapters")


def _adapter_ok(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def narration_adapter_dir(data_dir) -> str | None:
    """Local narration adapter in rp_data, else bundled weights shipped with the repo."""
    local = os.path.join(adapters_dir(data_dir), NARRATION_VERSION)
    if _adapter_ok(local):
        return local
    if _adapter_ok(BUNDLED_NARRATION):
        return BUNDLED_NARRATION
    return None


def ensure_narration_adapter(data_dir) -> str | None:
    """Copy bundled narration LoRA into rp_data once so sleep/chat paths can stack on it."""
    dest = os.path.join(adapters_dir(data_dir), NARRATION_VERSION)
    if _adapter_ok(dest):
        return dest
    if not _adapter_ok(BUNDLED_NARRATION):
        return None
    os.makedirs(adapters_dir(data_dir), exist_ok=True)
    if not os.path.isdir(dest):
        shutil.copytree(BUNDLED_NARRATION, dest)
    return dest


def read_current(data_dir) -> str:
    """Active adapter id: chat vN if current.json says so, else narration when available, else base."""
    p = os.path.join(adapters_dir(data_dir), "current.json")
    version = None
    if os.path.exists(p):
        with open(p) as f:
            version = json.load(f).get("version")
    if version and version.startswith("v") and _adapter_ok(os.path.join(adapters_dir(data_dir), version)):
        return version
    if version in (NARRATION_VERSION, "base", None) and narration_adapter_dir(data_dir):
        return NARRATION_VERSION
    return version or "base"


def write_current(data_dir, version):
    os.makedirs(adapters_dir(data_dir), exist_ok=True)
    p = os.path.join(adapters_dir(data_dir), "current.json")
    with open(p + ".tmp", "w") as f:
        json.dump({"version": version, "ts": time.time()}, f)
    os.replace(p + ".tmp", p)


def list_adapters(data_dir):
    d = adapters_dir(data_dir)
    out = []
    narr = narration_adapter_dir(data_dir)
    if narr:
        meta = os.path.join(narr, "meta.json")
        if os.path.exists(meta):
            with open(meta) as f:
                out.append({"version": NARRATION_VERSION, **json.load(f)})
        else:
            out.append({"version": NARRATION_VERSION, "kind": "narration"})
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d), key=lambda s: int(s[1:]) if s[1:].isdigit() else -1):
        meta = os.path.join(d, name, "meta.json")
        if name.startswith("v") and os.path.exists(meta):
            with open(meta) as f:
                out.append({"version": name, **json.load(f)})
    return out


def prompt_ids(tok, messages) -> list[int]:
    """Token ids for a chat prompt. Transformers 5 returns a BatchEncoding here, older versions a list."""
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(out, list):
        return out
    ids = out["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def truncate_messages(tok, messages, max_tokens):
    """Drop the oldest non-system messages until the rendered prompt fits."""
    msgs = list(messages)
    while True:
        n = len(prompt_ids(tok, msgs))
        droppable = [i for i, m in enumerate(msgs) if m["role"] != "system"]
        if n <= max_tokens or len(droppable) <= 1:
            return msgs
        del msgs[droppable[0]]


class _Cancel(StoppingCriteria):
    def __init__(self, ev):
        self.ev = ev

    def __call__(self, input_ids, scores, **kw):
        return self.ev.is_set()


class Engine:
    def __init__(self, model_name, data_dir, dtype="bfloat16", threads=None, max_context=3072):
        if threads:
            torch.set_num_threads(threads)
        self.model_name = model_name
        self.data_dir = data_dir
        self.max_context = max_context
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.base = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype))
        self.base.eval()
        self.model = self.base
        self.peft = False
        self.version = "base"
        self.lock = threading.Lock()
        ensure_narration_adapter(data_dir)
        self.activate(read_current(data_dir))

    # ---------- adapters ----------
    def _stack_paths(self, version: str) -> list[tuple[str, str]]:
        narr = ensure_narration_adapter(self.data_dir) or narration_adapter_dir(self.data_dir)
        if version == "base":
            return []
        if version == NARRATION_VERSION:
            if not narr:
                raise FileNotFoundError("no narration adapter (train with python -m rp.train_narration)")
            return [(NARRATION_VERSION, narr)]
        path = os.path.join(adapters_dir(self.data_dir), version)
        if not _adapter_ok(path):
            raise FileNotFoundError(f"no adapter at {path}")
        stack = []
        if narr:
            stack.append((NARRATION_VERSION, narr))
        stack.append((version, path))
        return stack

    def activate(self, version: str):
        """Hot-swap LoRA adapters. Chat vN stacks on frozen narration when both exist."""
        with self.lock:
            if version == "base":
                self.model = self.base
                self.peft = False
                self.version = "base"
                return
            stack = self._stack_paths(version)
            from peft import PeftModel
            slots = [f"{name}_{time.time_ns()}" for name, _ in stack]
            name0, path0 = stack[0]
            self.model = PeftModel.from_pretrained(self.base, path0, adapter_name=slots[0])
            self.peft = True
            active = [slots[0]]
            for (name, path), slot in zip(stack[1:], slots[1:]):
                self.model.load_adapter(path, adapter_name=slot)
                active.append(slot)
            self.model.set_adapter(active[-1] if len(active) == 1 else active)
            for name in list(self.model.peft_config):
                if name not in active:
                    self.model.delete_adapter(name)
            self.model.eval()
            self.version = version

    # ---------- generation ----------
    def render(self, messages, prefill=""):
        msgs = truncate_messages(self.tok, messages, self.max_context)
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + prefill

    def stream(self, messages, prefill="", max_tokens=300, temperature=0.7, top_p=0.9, top_k=0,
               repetition_penalty=1.08, seed=None, stop=None, cancel: threading.Event | None = None, info=None):
        """Yields text deltas. `info` (dict) is filled with timing/token stats and the adapter version used."""
        cancel = cancel or threading.Event()
        info = info if info is not None else {}
        stop = [s for s in (stop or []) if s]
        text = self.render(messages, prefill)
        ids = self.tok(text, return_tensors="pt")
        info["prompt_tokens"] = int(ids.input_ids.shape[1])
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True, timeout=600)
        gen_kw = dict(**ids, max_new_tokens=int(max_tokens), streamer=streamer,
                      stopping_criteria=StoppingCriteriaList([_Cancel(cancel)]),
                      repetition_penalty=float(repetition_penalty), pad_token_id=self.tok.eos_token_id)
        if temperature and temperature > 0:
            gen_kw.update(do_sample=True, temperature=float(temperature), top_p=float(top_p or 1.0))
            if top_k and top_k > 0:
                gen_kw["top_k"] = int(top_k)
            else:
                gen_kw["top_k"] = 0
        else:
            gen_kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        err = []

        def run():
            with self.lock:
                info["adapter"] = self.version
                info["t_start"] = time.time()
                try:
                    if seed is not None and seed >= 0:
                        torch.manual_seed(int(seed))
                    with torch.inference_mode():
                        if self.peft and self.version == "base":
                            with self.model.disable_adapter():
                                out = self.model.generate(**gen_kw)
                        else:
                            out = self.model.generate(**gen_kw)
                    info["completion_tokens"] = int(out.shape[1] - ids.input_ids.shape[1])
                except Exception as e:  # surfaced to the consumer below
                    err.append(e)
                    streamer.end()

        t0 = time.time()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        produced = ""
        first = None
        for delta in streamer:
            if not delta:
                continue
            if first is None:
                first = time.time()
            produced += delta
            if stop:
                hit = min((produced.find(s) for s in stop if s in produced), default=-1)
                if hit >= 0:
                    cancel.set()
                    keep = hit - (len(produced) - len(delta))
                    if keep > 0:
                        yield delta[:keep]
                    break
            yield delta
        th.join()
        if err:
            raise err[0]
        info["latency_s"] = time.time() - t0
        info["first_token_s"] = (first - t0) if first else None
        info["queue_s"] = info.get("t_start", t0) - t0

    def generate(self, messages, prefill="", info=None, **kw):
        info = info if info is not None else {}
        text = "".join(self.stream(messages, prefill, info=info, **kw))
        return text, info

    @torch.inference_mode()
    def response_nll(self, messages, response):
        """Mean per-token negative log-likelihood of `response` as the next assistant turn (adapter as active)."""
        prompt = self.render(messages)
        p_ids = self.tok(prompt, return_tensors="pt").input_ids
        r_ids = self.tok(response + "<|im_end|>", return_tensors="pt", add_special_tokens=False).input_ids
        x = torch.cat([p_ids, r_ids], 1)
        with self.lock:
            if self.peft and self.version == "base":
                with self.model.disable_adapter():
                    logits = self.model(x).logits
            else:
                logits = self.model(x).logits
        lp = torch.log_softmax(logits[0, p_ids.shape[1] - 1:-1].float(), -1)
        return float(-lp.gather(1, r_ids[0].unsqueeze(1)).mean())
