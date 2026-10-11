"""Held-out Chinese multi-character narration eval (base vs narration adapter).

    python -m rp.narration_eval --versions base narration --out runs/rp-narration/eval
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import time

from . import DEFAULT_DATA_DIR, DEFAULT_MODEL
from .engine import Engine

HERE = os.path.dirname(__file__)
DIALOGUE = re.compile(r"[「『\"]([^」』\"]{2,})[」』\"]")
NAME_LINE = re.compile(r"([一-龥]{1,4})(?:叹|沉声|低声|轻声|冷声|笑道|皱眉|咬牙|急道|答道|问|答|说|道)[：:]")


def score(text: str, rubric: list[str]) -> dict:
    names = set(NAME_LINE.findall(text))
    quotes = DIALOGUE.findall(text)
    rubric_hit = sum(1 for k in rubric if k in text) / max(1, len(rubric))
    return {
        "named_chars": len(names),
        "dialogue_lines": len(quotes),
        "rubric_hit": round(rubric_hit, 3),
        "has_narration": bool(re.search(r"[。！？…]", text) and len(text) > 80),
        "multi_char_dialogue": len(names) >= 2 or (len(quotes) >= 2 and rubric_hit >= 0.5),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", nargs="+", default=["base"])
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--model", default=os.environ.get("RP_MODEL", DEFAULT_MODEL))
    ap.add_argument("--heldout", default=os.path.join(os.path.dirname(HERE), "data", "rp-narration", "heldout.json"))
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default="runs/rp-narration/eval")
    a = ap.parse_args()

    held = json.load(open(a.heldout, encoding="utf-8"))
    eng = Engine(a.model, a.data_dir, threads=a.threads)
    results = {"heldout": a.heldout, "versions": {}, "created": time.ctime()}
    for v in a.versions:
        eng.activate(v)
        rows = []
        for p in held["prompts"]:
            msgs = [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}]
            samples = []
            for s in range(a.samples):
                text, info = eng.generate(msgs, max_tokens=a.max_tokens, temperature=0.7, top_p=0.9, seed=42 + s)
                samples.append({"text": text, **score(text, p.get("rubric", []))})
            rows.append({"user": p["user"], "samples": samples})
            print(f"[{v}] {p['user'][:40]}… | {samples[0]['text'][:100]!r}", flush=True)
        alls = [s for r in rows for s in r["samples"]]
        agg = {
            "multi_char_rate": round(sum(s["multi_char_dialogue"] for s in alls) / len(alls), 3),
            "avg_named_chars": round(sum(s["named_chars"] for s in alls) / len(alls), 2),
            "avg_dialogue_lines": round(sum(s["dialogue_lines"] for s in alls) / len(alls), 2),
            "rubric_hit": round(sum(s["rubric_hit"] for s in alls) / len(alls), 3),
            "rows": rows,
        }
        results["versions"][v] = agg
    results["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    md = ["# Narration held-out eval", "", f"Prompts: {len(held['prompts'])}", ""]
    md.append("| version | multi_char_rate | avg_named_chars | avg_dialogue | rubric_hit |")
    md.append("|---|---|---|---|---|")
    for v, r in results["versions"].items():
        md.append(f"| {v} | {r['multi_char_rate']} | {r['avg_named_chars']} | {r['avg_dialogue_lines']} | {r['rubric_hit']} |")
    md += ["", "## Samples", ""]
    for i, p in enumerate(held["prompts"]):
        md.append(f"**user:** {p['user']}")
        md.append("")
        for v, r in results["versions"].items():
            md.append(f"- *{v}:* {r['rows'][i]['samples'][0]['text'].strip().replace(chr(10), ' ')}")
        md.append("")
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md[:6]))


if __name__ == "__main__":
    main()
