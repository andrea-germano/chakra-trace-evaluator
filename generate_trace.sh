#!/usr/bin/env bash

# Phase 1/2: MLSynth trace generation only.
#
# Run this ONCE per model before launching one or more generate_log_ns3.sh jobs
# (even in parallel across different ns3 subdirs) for that model. Running
# MLSynth concurrently for the SAME model would corrupt the shared
# output/mlsynth/<model>/et/ directory (concurrent writers, partial files).

MODEL_FILE_NAME="$1"

if [ -t 0 ]; then
  [ -z "$MODEL_FILE_NAME" ] && read -p "Enter the model config file name without extension (e.g., bert_base): " MODEL_FILE_NAME
else
  if [ -z "$MODEL_FILE_NAME" ]; then
    echo "ERROR: No terminal attached to prompt for input, and required argument is missing."
    echo "Usage: $0 <model_file_name>"
    exit 1
  fi
fi

if [ -z "$MODEL_FILE_NAME" ]; then
  echo "ERROR: Model config file name is required. Exiting."
  exit 1
fi

echo "---------------------------------------------------"

# Paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MLSYNTH_SCRIPT="$BASE_DIR/../mlsynth/synthesise_inference.py"
MLSYNTH_CFG_PATH="$BASE_DIR/configs/mlsynth/${MODEL_FILE_NAME}.yaml"
TRACES_DIR="$BASE_DIR/output/mlsynth"

# ESSENTIAL CHECK: config file exists
if [ ! -f "$MLSYNTH_CFG_PATH" ]; then
  echo "ERROR: MLSynth config file not found at $MLSYNTH_CFG_PATH"
  exit 1
fi

echo "==> Running MLSynth to generate traces for model: $MODEL_FILE_NAME"
python3 "$MLSYNTH_SCRIPT" -c "$MLSYNTH_CFG_PATH" -o "$TRACES_DIR"

if [ $? -ne 0 ]; then
  echo "ERROR: MLSynth execution failed. Check the python errors above. Exiting."
  exit 1
fi

echo "==> Traces successfully generated in $TRACES_DIR/$MODEL_FILE_NAME"

# Sanity check: the number of NPUs (one .et per NPU) must equal the number of
# COMPUTE nodes in physical_topology.txt, and logical_topology.json must use that count.
NUM_NPUS=$(find "$TRACES_DIR/$MODEL_FILE_NAME/et/" -maxdepth 1 -name "${MODEL_FILE_NAME}.*.et" | wc -l | tr -d ' ')
echo "==> Detected $NUM_NPUS NPU trace file(s) (.et)."
echo "    -> physical_topology.txt must have exactly $NUM_NPUS compute nodes"
echo "    -> logical_topology.json must use [\"$NUM_NPUS\"]"
echo ""
echo "==> You can now run generate_log_ns3.sh $MODEL_FILE_NAME <ns3_subdir> for one"
echo "    or more ns3 subdirs, including in parallel with each other."
echo "---------------------------------------------------"
