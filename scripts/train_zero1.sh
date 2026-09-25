#!/usr/bin/env bash
set -euo pipefail

: "${1:?Usage: bash scripts/train_zero1.sh <nproc_per_node> [hydra_overrides...]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RIFT_ACCELERATE_CONFIG_FILE="${SCRIPT_DIR}/accelerate_configs/accelerate_zero1_ds.yaml"
exec bash "${SCRIPT_DIR}/train_zero2.sh" "$@"
