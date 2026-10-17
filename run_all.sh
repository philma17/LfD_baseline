#!/usr/bin/env bash
# Limit-Analyse S2-NNDS, Step 0.
#
#   ./run_all.sh            # Phase-0-Tests
#   ./run_all.sh phase1     # Phase-1-Sweep (288 Laeufe, ~8 min auf 4 Kernen) + Auswertung
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/lfd_env/bin/python}"
export PYTHONDONTWRITEBYTECODE=1   # nichts in S2NNDS/ (read-only) schreiben

cd "$ROOT"
if [[ "${1:-}" == "phase1" ]]; then
    "$PY" -u experiments/run_grid.py configs/stage1_clean.yaml --workers "$(nproc)" \
        --out results/phase1/runs_clean.jsonl
    "$PY" -u experiments/phase1_report.py
    "$PY" -u experiments/phase1_traj_plot.py
    exit 0
fi
mkdir -p results/phase0
echo "[run_all] Python: $("$PY" --version 2>&1) ($PY)"
echo "[run_all] Phase 0: Tests"
"$PY" -u -m unittest discover -s tests -v 2>&1 | tee results/phase0/tests.log
