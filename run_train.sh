#!/usr/bin/env bash
# SpectraVision training.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -eq 0 ]]; then
  echo "Usage: ./run_train.sh MODEL [--data TRAIN_H5_OR_DIR]" >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo "Usage: ./run_train.sh MODEL [--data TRAIN_H5_OR_DIR]"
  echo "       ./run_train.sh CONFIG.json [--profiles configs/models.json]"
  exit 0
fi

if [[ "$1" == SpectraVision-* ]]; then
  exec uv run python run_model.py train "$@"
fi

CONFIG="$1"
shift
exec uv run python -m spectravision train --config "$CONFIG" "$@"
