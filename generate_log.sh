#!/usr/bin/env bash

# 1. Ask for user input
read -p "Enter the model folder name (e.g., bert, gpt3): " MODEL_FOLDER
read -p "Enter the config file name without extension (e.g., config_base): " CONFIG_FILE

# ESSENTIAL CHECK 1: Ensure inputs are not empty
if [ -z "$MODEL_FOLDER" ] || [ -z "$CONFIG_FILE" ]; then
  echo "ERROR: Both model folder and config file name are required. Exiting."
  exit 1
fi

MODEL_NAME="${MODEL_FOLDER}_${CONFIG_FILE}"

echo "---------------------------------------------------"

# 2. Get paths based on the project structure
#BASE_DIR is simply the script's location
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Absolute paths for the executables
MLSYNTH_SCRIPT="$BASE_DIR/../mlsynth/synthesise_inference.py"
ASTRA_BIN="$BASE_DIR/../astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"

# 3. Define paths for base configurations
MLSYNTH_CFG_PATH="$BASE_DIR/configs/mlsynth/$MODEL_FOLDER/$CONFIG_FILE.yaml"
SYSTEM_CFG="$BASE_DIR/configs/astra_sim/system/h100_1D.json"
NETWORK_CFG="$BASE_DIR/configs/astra_sim/analytical/network2D.yml"
REMOTE_MEM_CFG="$BASE_DIR/configs/astra_sim/no_remote_memory.json"
LOGGING_CFG="$BASE_DIR/configs/astra_sim/logging_config.toml"

# ESSENTIAL CHECK 2: Verify the MLSynth config file actually exists
if [ ! -f "$MLSYNTH_CFG_PATH" ]; then
  echo "ERROR: Config file not found at $MLSYNTH_CFG_PATH"
  exit 1
fi

# 4. Define output paths
TRACES_DIR="$BASE_DIR/output/mlsynth"
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
  --comm-group-configuration="$COMM_GROUPS" \
  --logging-configuration="$LOGGING_CFG"
RC=$?

if [ $RC -ne 0 ]; then
  echo "==> ERROR: ASTRA-sim exited with code $RC."
  echo "    Check the terminal output above for the failing argument/file."
  exit 1
fi

# Step 3: Log management
# ESSENTIAL CHECK 4: Verify ASTRA-sim succeeded by checking if 'log' folder exists
echo "==> Simulation finished. Moving logs..."
if [ -d "$BASE_DIR/log" ]; then

 # If a log folder for this model already exists in astra_logs, remove it to avoid confusion
  if [ -d "$BASE_DIR/output/astra_logs/analytical_${MODEL_NAME}" ]; then
    rm -rf "$BASE_DIR/output/astra_logs/analytical_${MODEL_NAME}"
  fi
  
  mv "$BASE_DIR/log" "$BASE_DIR/output/astra_logs/analytical_${MODEL_NAME}"
  echo "==> SUCCESS: Data saved in $BASE_DIR/output/astra_logs/analytical_${MODEL_NAME}"
else
  echo "==> ERROR: 'log' folder not found! ASTRA-sim might have crashed."
  echo "Check logs for details."
  exit 1
fi

echo "---------------------------------------------------"