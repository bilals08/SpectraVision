#!/usr/bin/env bash
# SpectraVision inference and batch evaluation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -gt 0 && "$1" == SpectraVision-* ]]; then
  exec uv run python run_model.py infer "$@"
fi

DATA="inference_data"
if [[ $# -gt 0 && "$1" != -* ]]; then
  DATA="$1"
  shift
fi

exec uv run python -m spectravision evaluate \
  --data "$DATA" \
  --output outputs/predictions_batch1024 \
  "$@"
