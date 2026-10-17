#!/usr/bin/env bash
# Limit-Analyse S2-NNDS, Step 0.
# Aktuell nur Phase 0 (Fundament-Tests). Stufe 1-5 folgen nach Freigabe.
#
#   ./run_all.sh            # Phase-0-Tests
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/lfd_env/bin/python}"
export PYTHONDONTWRITEBYTECODE=1   # nichts in S2NNDS/ (read-only) schreiben

cd "$ROOT"
mkdir -p results/phase0
echo "[run_all] Python: $("$PY" --version 2>&1) ($PY)"
echo "[run_all] Phase 0: Tests"
"$PY" -u -m unittest discover -s tests -v 2>&1 | tee results/phase0/tests.log
