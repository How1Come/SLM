#!/usr/bin/env bash
# Full comparison: baseline vs Cradle vs single-component ablations, same compute budget each.
# Usage: scripts/run_experiments.sh [budget_tokens] [seed]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
BUDGET=${1:-12e6}
SEED=${2:-0}
JUDGE=${JUDGE:---judge}

[ -f data/prepared.pt ] || $PY -m slm.data

run() {
  local name=$1; shift
  if [ ! -f "runs/$name/model.pt" ]; then
    $PY -m slm.train --budget-tokens "$BUDGET" --seed "$SEED" --name "$name" "$@" 2>&1 | tee "runs/$name.train.txt"
  fi
  $PY -m slm.evaluate "runs/$name/model.pt" $JUDGE --out runs/results.json 2>&1 | grep -v -i "warn\|unexpected\|notes:\|can be ignored" || true
}

mkdir -p runs
run "baseline-s$SEED" --recipe baseline
run "cradle-s$SEED" --recipe cradle
if [ "${ABLATIONS:-1}" = "1" ]; then
  run "cradle-no-curriculum-s$SEED" --recipe cradle --no-curriculum
  run "cradle-no-babble-s$SEED" --recipe cradle --no-babble
  run "cradle-no-prior-s$SEED" --recipe cradle --no-prior
  run "cradle-no-sleep-s$SEED" --recipe cradle --no-sleep
fi
$PY -m slm.report
