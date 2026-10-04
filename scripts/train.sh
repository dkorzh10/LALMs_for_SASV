#!/usr/bin/env bash
# Launch training / evaluation with torchrun.
# Usage:
#   ./scripts/train.sh configs/salmon_hard_label.yaml [NUM_GPUS]
# Loads .env from the repo root if present (KEY=value, no export needed).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

CONFIG="${1:?Usage: $0 <config.yaml> [num_gpus]}"
NUM_GPUS="${2:-${NUM_GPUS:-1}}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

# Fail early if YAML placeholders would be left unexpanded.
missing=()
for var in DATA_ROOT SALMONN_CKPT LLAMA_PATH WHISPER_PATH BEATS_PATH; do
  if [[ -z "${!var:-}" ]]; then
    missing+=("$var")
  fi
done
if ((${#missing[@]})); then
  echo "Missing environment variables: ${missing[*]}" >&2
  echo "Set them in ${ROOT}/.env or export them before running." >&2
  exit 1
fi

if [[ "$NUM_GPUS" -le 1 ]]; then
  python -m src.runner --config "$CONFIG"
else
  torchrun --standalone --nproc_per_node="$NUM_GPUS" -m src.runner --config "$CONFIG"
fi
