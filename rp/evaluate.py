"""Held-out before/after measurement for one character.

    python -m rp.evaluate --versions base v1 --samples 3 --out runs/rp-demo/eval

For every adapter version, on held-out prompts that were never used for training (fresh chat: SillyTavern's
system prompt + greeting + one user message):
  gold NLL     mean per-token NLL of a reference in-character reply (lower = the model finds her voice likelier)
  persona      share of sampled replies that use her self-reference 本座 / call the user 小书生 /
               contain an *action* description / avoid assistant-speak (有什么可以帮你, AI, 助手, 您, ...)
Sampling uses the same seeds for every version, so differences come from the adapter, not the dice.
"""
import argparse
import json
import os
import re
import resource
import time

from . import DEFAULT_DATA_DIR, DEFAULT_MODEL
from .engine import Engine
from .stprompt import build, default_system, greeting, system_from_db

HERE = os.path.dirname(__file__)
ASSISTANT = re.compile(r"有什么(我)?(可以|能)(帮|为)|帮助你|帮到你|需要帮助|AI|人工智能|助手|语言模型|作为一个|虚构的?角色|您")
ACTION = re.compile(r"\*[^*\n]{2,}\*")


def metrics(text):
    return {"benzuo": "本座" in text, "shusheng": "书生" in text, "action": bool(ACTION.search(text)),
            "no_assistant_speak": not ASSISTANT.search(text)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=["base"])
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--model", default=os.environ.get("RP_MODEL", DEFAULT_MODEL))
    ap.add_argument("--card", default=os.path.join(HERE, "characters", "su_xiaotang.json"))
    ap.add_argument("--demo", default=os.path.join(HERE, "characters", "su_xiaotang_demo.json"))
    ap.add_argument("--user", default="muzi")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="runs/rp-demo/eval")
    a = ap.parse_args()

    card = json.load(open(a.card, encoding="utf-8"))
    demo = json.load(open(a.demo, encoding="utf-8"))
    system = system_from_db(os.path.join(a.data_dir, "rp.db"), card["data"]["name"]) or default_system(card, a.user)
    greet = greeting(card, a.user)
    eng = Engine(a.model, a.data_dir, threads=a.threads)
    results = {"system_prompt": system, "versions": {}}
    for v in a.versions:
        eng.activate(v)
        rows, nlls, toks, secs, firsts = [], [], [], [], []
        for h in demo["heldout"]:
            msgs = build(system, greet, [(h["user"], None)])
            nll = eng.response_nll(msgs, h["gold"])
            nlls.append(nll)
            samples = []
            for s in range(a.samples):
                text, info = eng.generate(msgs, max_tokens=a.max_tokens, temperature=0.7, top_p=0.9, seed=1000 + s)
                toks.append(info.get("completion_tokens", 0)); secs.append(info["latency_s"]); firsts.append(info["first_token_s"] or 0)
                samples.append({"text": text, **metrics(text)})
            rows.append({"user": h["user"], "gold_nll": round(nll, 4), "samples": samples})
            print(f"[{v}] {h['user']} nll={nll:.3f} | {samples[0]['text'][:80]!r}", flush=True)
        allm = [s for r in rows for s in r["samples"]]
        rate = {k: round(sum(s[k] for s in allm) / len(allm), 3) for k in ("benzuo", "shusheng", "action", "no_assistant_speak")}
        rate["persona_score"] = round(sum(rate.values()) / 4, 3)
        results["versions"][v] = {
            "gold_nll": round(sum(nlls) / len(nlls), 4), **rate,
            "tokens_per_s": round(sum(toks) / max(1e-6, sum(secs) - sum(firsts)), 1),
            "first_token_s": round(sum(firsts) / len(firsts), 3), "avg_reply_s": round(sum(secs) / len(secs), 2),
            "rows": rows}
    results["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    results["created"] = time.ctime()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    cols = ["gold_nll", "benzuo", "shusheng", "action", "no_assistant_speak", "persona_score", "tokens_per_s", "avg_reply_s"]
    md = [f"# Held-out evaluation: {card['data']['name']}", "",
          f"{len(demo['heldout'])} held-out prompts x {a.samples} samples (T=0.7, top-p 0.9, same seeds per version). "
          "gold_nll: mean per-token NLL of a reference in-character reply (lower is better). Rates are shares of "
          "sampled replies.", "",
          "| version | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for v, r in results["versions"].items():
        md.append(f"| {v} | " + " | ".join(str(r[c]) for c in cols) + " |")
    md += ["", "## Samples (first sample per prompt)", ""]
    for i, h in enumerate(demo["heldout"]):
        md.append(f"**muzi:** {h['user']}")
        md.append("")
        for v, r in results["versions"].items():
            md.append(f"- *{v}:* {r['rows'][i]['samples'][0]['text'].strip()}".replace("\n", " "))
        md.append("")
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md[:8 + len(results["versions"])]))


if __name__ == "__main__":
    main()
