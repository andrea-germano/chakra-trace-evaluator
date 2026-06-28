#!/usr/bin/env bash

# 1. Ask for user input (supports command line args or interactive prompt)
MODEL_FILE_NAME="$1"
OUTPUT_DIR_NAME="$2"

[ -z "$MODEL_FILE_NAME" ] && read -p "Enter the model config file name without extension (e.g., bert_base): " MODEL_FILE_NAME
[ $# -lt 2 ] && read -p "Enter the output directory name (leave blank to use analytical_<model>): " OUTPUT_DIR_NAME

# ESSENTIAL CHECK 1: Ensure inputs are not empty
if [ -z "$MODEL_FILE_NAME" ]; then
  echo "ERROR: Model config file name is required. Exiting."
  exit 1
fi

if [ -z "$OUTPUT_DIR_NAME" ]; then
  OUTPUT_DIR_NAME="analytical_${MODEL_FILE_NAME}"
  echo "Output directory not specified, using: $OUTPUT_DIR_NAME"
fi

echo "---------------------------------------------------"

# 2. Get paths based on the project structure
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Absolute paths for the executables
MLSYNTH_SCRIPT="$BASE_DIR/../mlsynth/synthesise_inference.py"
ASTRA_BIN="$BASE_DIR/../astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"

# 3. Define paths for base configurations
MLSYNTH_CFG_PATH="$BASE_DIR/configs/mlsynth/${MODEL_FILE_NAME}.yaml"
SYSTEM_CFG="$BASE_DIR/configs/astra_sim/system/h100_2D.json"
NETWORK_CFG="$BASE_DIR/configs/astra_sim/analytical/network2D.yml"
REMOTE_MEM_CFG="$BASE_DIR/configs/astra_sim/no_remote_memory.json"
LOGGING_CFG="$BASE_DIR/configs/astra_sim/logging_config.toml"

# 4. Define output paths
TRACES_DIR="$BASE_DIR/output/mlsynth"
WORKLOAD_PREFIX="$TRACES_DIR/$MODEL_FILE_NAME/et/$MODEL_FILE_NAME"
COMM_GROUPS="$TRACES_DIR/$MODEL_FILE_NAME/comm_groups.json"
OUTPUT_DIR="$BASE_DIR/output/astra_logs/${OUTPUT_DIR_NAME}"

# ESSENTIAL CHECK 2: Verify all required files and binaries exist
if [ ! -f "$MLSYNTH_CFG_PATH" ]; then
  echo "ERROR: MLSynth config file not found at $MLSYNTH_CFG_PATH"
  exit 1
fi

if [ ! -x "$ASTRA_BIN" ]; then
  echo "ERROR: Analytical binary not found or not executable at:"
  echo "       $ASTRA_BIN"
  exit 1
fi

for f in "$SYSTEM_CFG" "$NETWORK_CFG" "$REMOTE_MEM_CFG"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Required ASTRA-sim config file missing: $f"
    exit 1
  fi
done

# --- EXECUTION ---

# Step 1: MLSynth
echo "==> [1/2] Running MLSynth to generate traces for model: $MODEL_FILE_NAME"
python3 "$MLSYNTH_SCRIPT" -c "$MLSYNTH_CFG_PATH" -o "$TRACES_DIR"

# ESSENTIAL CHECK 3: Stop if MLSynth failed
if [ $? -ne 0 ]; then
  echo "ERROR: MLSynth execution failed. Check the python errors above. Exiting."
  exit 1
fi

echo "==> Traces successfully generated in $TRACES_DIR/$MODEL_FILE_NAME"
echo ""

# Ensure we run ASTRA-sim from the root
cd "$BASE_DIR" || exit

if [ -d "$OUTPUT_DIR" ]; then
  rm -rf "$OUTPUT_DIR"
  echo "Existing output directory $OUTPUT_DIR removed."
fi

# Step 2: ASTRA-sim
echo "==> [2/2] Starting ASTRA-sim simulation (Analytical backend)..."
"$ASTRA_BIN" \
  --workload-configuration="$WORKLOAD_PREFIX" \
  --system-configuration="$SYSTEM_CFG" \
  --network-configuration="$NETWORK_CFG" \
  --remote-memory-configuration="$REMOTE_MEM_CFG" \
  --comm-group-configuration="$COMM_GROUPS" \
  --logging-folder="$OUTPUT_DIR" 
  # --logging-configuration="$LOGGING_CFG" \
RC=$?

if [ $RC -ne 0 ]; then
  echo "==> ERROR: ASTRA-sim exited with code $RC."
  echo "    Check the terminal output above for the failing argument/file."
  if [ -d "$OUTPUT_DIR" ]; then
    rm -rf "$OUTPUT_DIR"
    echo "==> NOTE: Removed the 'log' folder produced by ASTRA-sim due to the error."
  fi
  exit 1
fi
echo "==> SUCCESS: System-layer logs saved in $OUTPUT_DIR"
echo "---------------------------------------------------"