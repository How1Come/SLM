#!/usr/bin/env bash
# Full comparison at a fixed compute budget: baseline vs Cradle variants and single-component ablations.
# Usage: scripts/run_experiments.sh [budget_tokens] [seed]      (each run is ~14 min on 4 CPU cores)
#   ABLATIONS=0  only run baseline, Cradle v1 and the recommended listen-only recipe
#   JUDGE=       disable the TinyStories-33M judge (no transformers needed)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=${PY:-.venv/bin/python}
BUDGET=${1:-12e6}
SEED=${2:-0}
JUDGE=${JUDGE---judge}
V2="--cds-mix 0.07 --phases 0.1 0.5 0.25 0.15 --babble-mode best-of-k --no-sleep"

[ -f data/prepared.pt ] || $PY -m slm.data
mkdir -p runs

run() {
  local name=$1; shift
  if [ ! -f "runs/$name/model.pt" ]; then
    $PY -m slm.train --budget-tokens "$BUDGET" --seed "$SEED" --name "$name" "$@" > "runs/$name.train.txt" 2>&1
  fi
  $PY -m slm.evaluate "runs/$name/model.pt" $JUDGE --out runs/results.json 2>&1 | grep "stories ppl" || true
}

run "baseline-s$SEED" --recipe baseline
run "cradle-s$SEED" --recipe cradle                                     # v1: all four phases
run "cradle-listen-only-s$SEED" --recipe cradle --no-babble --no-sleep  # curriculum + bigram prior
if [ "${ABLATIONS:-1}" = "1" ]; then
  run "cradle-no-curriculum-s$SEED" --recipe cradle --no-curriculum
  run "cradle-no-babble-s$SEED" --recipe cradle --no-babble
  run "cradle-no-prior-s$SEED" --recipe cradle --no-prior
  run "cradle-no-sleep-s$SEED" --recipe cradle --no-sleep
  run "cradle-v2-s$SEED" --recipe cradle $V2
  run "cradle-v2-no-babble-s$SEED" --recipe cradle $V2 --no-babble
  run "baseline-plus-prior-s$SEED" --recipe cradle --no-curriculum --no-babble --no-sleep
fi
$PY -m slm.report
