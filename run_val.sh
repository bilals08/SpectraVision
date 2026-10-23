#!/usr/bin/env bash
# SpectraVision validation.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  echo "Usage: ./run_val.sh VALIDATION.h5 [--model MODEL] [other evaluate options]"
  exit 0
fi
if [[ $# -eq 0 || "$1" == -* ]]; then
  echo "Usage: ./run_val.sh VALIDATION.h5 [--model MODEL] [other evaluate options]" >&2
  exit 2
fi

DATA="$1"
shift
exec uv run python -m spectravision evaluate \
  --data "$DATA" \
  --output outputs/validation_batch1024 \
  "$@"
