#!/usr/bin/env bash

echo "==================================================="
echo "  Batch Simulation Runner - Disaggregated LLM"
echo "==================================================="

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLSYNTH_SCRIPT="$BASE_DIR/../mlsynth/synthesise_inference.py"
ASTRA_BIN="$BASE_DIR/../astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"

AUTO_GEN_MLSYNTH_DIR="$BASE_DIR/configs/mlsynth/auto_generated"
AUTO_GEN_ASTRASIM_DIR="$BASE_DIR/configs/astra_sim/auto_generated"
ASTRA_LOGS_DIR="$BASE_DIR/astra_logs/auto_generated"
TRACES_BASE_DIR="$BASE_DIR/mlsynth_traces/auto_generated"

REMOTE_MEM_CFG="$BASE_DIR/configs/astra_sim/base_templates/remote_memory_base.json"

# --- GLOBAL CLEANUP ---
echo "=> Cleaning up old simulation traces and logs..."
rm -rf "$TRACES_BASE_DIR"
mkdir -p "$TRACES_BASE_DIR"
rm -rf "$ASTRA_LOGS_DIR"
mkdir -p "$ASTRA_LOGS_DIR"

# Check available configurations
CONFIG_FILES=("$AUTO_GEN_MLSYNTH_DIR"/*_mlsynth.yaml)
TOTAL_SCENARIOS=${#CONFIG_FILES[@]}

if [ $TOTAL_SCENARIOS -eq 0 ] || [ ! -e "${CONFIG_FILES[0]}" ]; then
    echo "No configuration files found. Please run generate_configs.py first. Exiting."
    exit 0
fi

echo "Found $TOTAL_SCENARIOS scenarios to simulate."
echo "---------------------------------------------------"

COUNTER=1
for MLSYNTH_CFG_PATH in "${CONFIG_FILES[@]}"; do
    
    FILENAME=$(basename -- "$MLSYNTH_CFG_PATH")
    SCENARIO_ID="${FILENAME%_mlsynth.yaml}"

    echo "==> [$COUNTER/$TOTAL_SCENARIOS] Starting Scenario: $SCENARIO_ID"

    SYSTEM_CFG="$AUTO_GEN_ASTRASIM_DIR/${SCENARIO_ID}_system.json"
    NETWORK_CFG="$AUTO_GEN_ASTRASIM_DIR/${SCENARIO_ID}_network.yml"

    SCENARIO_TRACES_DIR="$TRACES_BASE_DIR/$SCENARIO_ID"
    WORKLOAD_PREFIX="$SCENARIO_TRACES_DIR/$SCENARIO_ID/et/$SCENARIO_ID"
    COMM_GROUPS="$SCENARIO_TRACES_DIR/$SCENARIO_ID/comm_groups.json"

    # --- A: MLSynth ---
    python3 "$MLSYNTH_SCRIPT" -c "$MLSYNTH_CFG_PATH" -o "$SCENARIO_TRACES_DIR" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "    [!] MLSynth ERROR. Skipping to the next one."
      let COUNTER++
      continue
    fi

    # --- B: ASTRA-sim ---
    cd "$BASE_DIR" || exit
    "$ASTRA_BIN" \
      --workload-configuration="$WORKLOAD_PREFIX" \
      --system-configuration="$SYSTEM_CFG" \
      --network-configuration="$NETWORK_CFG" \
      --remote-memory-configuration="$REMOTE_MEM_CFG" \
      --comm-group-configuration="$COMM_GROUPS" > /dev/null 2>&1

    # --- C: Log Management ---
    if [ -d "$BASE_DIR/log" ]; then
        TARGET_LOG_DIR="$ASTRA_LOGS_DIR/$SCENARIO_ID"
        mv "$BASE_DIR/log" "$TARGET_LOG_DIR"
        echo "    -> DONE. Log saved in astra_logs/$SCENARIO_ID"
    else
        echo "    [!] ASTRA-sim ERROR: Log folder not generated."
    fi

    echo "---------------------------------------------------"
    let COUNTER++
done

echo "All simulations completed successfully!"