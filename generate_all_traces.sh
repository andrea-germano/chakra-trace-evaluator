#!/usr/bin/env bash

# Regenerate MLSynth Chakra traces for every model config in configs/mlsynth/,
# by invoking generate_traces.sh once per .yaml. Stops at the first failure.

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for cfg in "$BASE_DIR"/configs/mlsynth/*.yaml; do
  model="$(basename "$cfg" .yaml)"
  echo "==> Regenerating traces for: $model"
  bash "$BASE_DIR/generate_traces.sh" "$model"
done