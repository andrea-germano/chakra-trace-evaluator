#!/usr/bin/env bash

# 1. Ask for user input
read -p "Enter the name of the MLSynth config (e.g., config_base.yaml): " MLSYNTH_CFG_NAME
read -p "Enter the model name contained within it (e.g., streaming_push): " MODEL_NAME

# ESSENTIAL CHECK 1: Ensure inputs are not empty
if [ -z "$MLSYNTH_CFG_NAME" ] || [ -z "$MODEL_NAME" ]; then
  echo "ERROR: Both config name and model name are required. Exiting."
  exit 1
fi

echo "---------------------------------------------------"

# 2. Get paths based on the project structure
#BASE_DIR is simply the script's location
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Absolute paths for the executables
MLSYNTH_SCRIPT="$BASE_DIR/../mlsynth/synthesise_inference.py"
ASTRA_BIN="$BASE_DIR/../astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"

# 3. Define paths for base configurations
MLSYNTH_CFG_PATH="$BASE_DIR/configs/mlsynth/base_templates/$MLSYNTH_CFG_NAME"
SYSTEM_CFG="$BASE_DIR/configs/astra_sim/base_templates/system_base.json"
NETWORK_CFG="$BASE_DIR/configs/astra_sim/base_templates/network_base.yml"
REMOTE_MEM_CFG="$BASE_DIR/configs/astra_sim/base_templates/remote_memory_base.json"

# ESSENTIAL CHECK 2: Verify the MLSynth config file actually exists
if [ ! -f "$MLSYNTH_CFG_PATH" ]; then
  echo "ERROR: Config file not found at $MLSYNTH_CFG_PATH"
  exit 1
fi

# 4. Define output paths
TRACES_DIR="$BASE_DIR/mlsynth_traces"
WORKLOAD_PREFIX="$TRACES_DIR/$MODEL_NAME/et/$MODEL_NAME"
COMM_GROUPS="$TRACES_DIR/$MODEL_NAME/comm_groups.json"

# --- EXECUTION ---

# Step 1: MLSynth
echo "==> [1/2] Running MLSynth to generate traces for model: $MODEL_NAME"
python3 "$MLSYNTH_SCRIPT" -c "$MLSYNTH_CFG_PATH" -o "$TRACES_DIR"

# ESSENTIAL CHECK 3: Stop if MLSynth failed (exit code is not 0)
if [ $? -ne 0 ]; then
  echo "ERROR: MLSynth execution failed. Check the python errors above. Exiting."
  exit 1
fi

echo "==> Traces successfully generated in $TRACES_DIR/$MODEL_NAME"
echo ""

# Ensure we run ASTRA-sim from the root so the 'log' folder spawns here
cd "$BASE_DIR" || exit

# Step 2: ASTRA-sim
echo "==> [2/2] Starting ASTRA-sim simulation..."
"$ASTRA_BIN" \
  --workload-configuration="$WORKLOAD_PREFIX" \
  --system-configuration="$SYSTEM_CFG" \
  --network-configuration="$NETWORK_CFG" \
  --remote-memory-configuration="$REMOTE_MEM_CFG" \
  --comm-group-configuration="$COMM_GROUPS"

# Step 3: Log management
# ESSENTIAL CHECK 4: Verify ASTRA-sim succeeded by checking if 'log' folder exists
echo "==> Simulation finished. Moving logs..."
if [ -d "$BASE_DIR/log" ]; then

 # If a log folder for this model already exists in astra_logs, remove it to avoid confusion
  if [ -d "$BASE_DIR/astra_logs/log_${MODEL_NAME}" ]; then
    rm -rf "$BASE_DIR/astra_logs/log_${MODEL_NAME}"
  fi
  
  mv "$BASE_DIR/log" "$BASE_DIR/astra_logs/${MODEL_NAME}"
  echo "==> SUCCESS: Data saved in $BASE_DIR/astra_logs/${MODEL_NAME}"
else
  echo "==> ERROR: 'log' folder not found! ASTRA-sim might have crashed."
  echo "Check $BASE_DIR/astra_logs/terminal_output_${MODEL_NAME}.log for details."
  exit 1
fi

echo "---------------------------------------------------"