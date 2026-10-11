"""Training presets for the Cradle trainer UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    id: str
    title: str
    blurb: str
    recipe: str
    train_args: tuple[str, ...] = ()
    default_name: str = ""


PRESETS: dict[str, Preset] = {
    "listen": Preset(
        id="listen",
        title="Listen-only Cradle (recommended)",
        blurb=(
            "Age-ordered caregiver speech, then stories from simple to complex, with a fading bigram prior. "
            "Skips babble and sleep — the setup that beat the baseline on held-out story perplexity in this repo."
        ),
        recipe="cradle",
        train_args=("--no-babble", "--no-sleep"),
        default_name="listen",
    ),
    "cradle": Preset(
        id="cradle",
        title="Full four-phase Cradle",
        blurb=(
            "All four phases: listen (CDS + stories), babble with caregiver feedback, then sleep (EMA self-distillation). "
            "Interesting to compare, but babble and sleep tended to hurt perplexity on this small corpus."
        ),
        recipe="cradle",
        train_args=(),
        default_name="cradle",
    ),
    "baseline": Preset(
        id="baseline",
        title="Baseline",
        blurb=(
            "Plain shuffled next-token prediction on the same data, with the same compute budget as Cradle. "
            "The fair comparison point for the listen-only recipe."
        ),
        recipe="baseline",
        train_args=(),
        default_name="baseline",
    ),
    "smoke": Preset(
        id="smoke",
        title="Smoke test (~1 min)",
        blurb=(
            "Listen-only Cradle at a tiny budget (1M trained-token equivalents) to verify the UI and pipeline "
            "without waiting for a full ~13 minute run."
        ),
        recipe="cradle",
        train_args=("--no-babble", "--no-sleep", "--budget-tokens", "1e6"),
        default_name="smoke",
    ),
}

DEFAULT_BUDGET = 12e6
DEFAULT_SEED = 0
