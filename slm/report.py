"""Summarize runs/results.json and per-run logs into a Markdown table (runs/report.md)."""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("runs")


def main():
    results = json.loads((RUNS / "results.json").read_text())
    rows = [
        "| run | stories ppl | CHILDES ppl | Zorro acc | judge ppl (gen) | distinct-2 | rep-3 | trained tok | fwd-only tok | wall (min) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in sorted(results.items()):
        log_path = RUNS / name / "log.json"
        s = json.loads(log_path.read_text())["summary"] if log_path.exists() else {}
        judge = f"{r['gen_judge_ppl']:.2f}" if "gen_judge_ppl" in r else "-"
        rows.append(
            f"| {name} | {r['stories_val_ppl']:.2f} | {r['childes_val_ppl']:.2f} | {r['zorro_mean']:.3f} | {judge} | "
            f"{r['gen_distinct2']:.3f} | {r['gen_repetition3']:.3f} | {s.get('trained_tokens', 0)/1e6:.2f}M | "
            f"{s.get('forward_only_tokens', 0)/1e6:.2f}M | {s.get('wall_s', 0)/60:.1f} |"
        )
    out = ["# Results", "", *rows, ""]
    for name, r in sorted(results.items()):
        out += [f"## {name} samples", ""] + [f"- {t}" for t in r["samples"][:3]] + [""]
    (RUNS / "report.md").write_text("\n".join(out))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
