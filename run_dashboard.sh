#!/usr/bin/env bash
# M3TRICS Dashboard Launcher
# Run from git_exp/ after completing analysis:
#   bash m3trics/run_dashboard.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║        M3TRICS Interactive Dashboard         ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Python interpreter ──────────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Activate your conda environment first." >&2
    exit 1
fi

# ── Configuration ───────────────────────────────────────────────────────────
# Analysis directories (auto-detected from repo root; override if needed)
# ANALYSIS_DIR="${SCRIPT_DIR}/analysis/progressive_missingness_analysis_outputs"

# Models ending in _KD are detected automatically.
# Optionally pass custom/legacy distillation model names as a comma-separated list.
DISTILLATION_MODELS="${DISTILLATION_MODELS:-}"

# Output HTML file (relative to repo root)
OUTPUT="m3trics/dashboard/m3trics_dashboard.html"

# ── Generate dashboard ──────────────────────────────────────────────────────
cd "${REPO_ROOT}"
${PYTHON} m3trics/dashboard/generate_dashboard.py \
    --distillation_models "${DISTILLATION_MODELS}" \
    --output              "${OUTPUT}"

# ── Open in browser ─────────────────────────────────────────────────────────
if [[ -f "${OUTPUT}" ]]; then
    echo ""
    if [[ "$(uname)" == "Darwin" ]]; then
        open "${OUTPUT}"
    elif command -v xdg-open &>/dev/null; then
        xdg-open "${OUTPUT}"
    else
        echo "Open manually: ${REPO_ROOT}/${OUTPUT}"
    fi
fi
