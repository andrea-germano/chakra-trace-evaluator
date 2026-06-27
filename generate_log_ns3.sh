#!/usr/bin/env bash
# 1. User input
read -p "Enter the model folder name (e.g., bert, gpt3): " MODEL_FOLDER
read -p "Enter the config file name without extension (e.g., config_base): " CONFIG_FILE
read -p "Enter the name of the ns3 configuration subdirectory: " NS3_SUBDIR
read -p "Enter the output directory name (leave blank to use <model>_<ns3subdir>): " OUTPUT_DIR

if [ -z "$MODEL_FOLDER" ] || [ -z "$CONFIG_FILE" ] || [ -z "$NS3_SUBDIR" ]; then
  echo "ERROR: Model folder, config file name, and ns3 subdirectory are required. Exiting."
  exit 1
fi

MODEL_NAME="${MODEL_FOLDER}_${CONFIG_FILE}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${MODEL_NAME}_${NS3_SUBDIR}"
  echo "Output directory not specified, using: $OUTPUT_DIR"
fi

echo "---------------------------------------------------"

# 2. Paths
# BASE_DIR is the script's location
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MLSYNTH_SCRIPT="$BASE_DIR/../mlsynth/synthesise_inference.py"

# ns-3 backend lives inside astra-sim/extern/network_backend/ns-3
NS3_DIR="$BASE_DIR/../astra-sim/extern/network_backend/ns-3"
NS3_BIN="$NS3_DIR/build/scratch/ns3.42-AstraSimNetwork-default"

# 3. NEW ns-3 config templates (these are the files this script points to,
#    NOT the astra-sim example files in scratch/config).
NS3_CFG_DIR="$BASE_DIR/configs/astra_sim/ns3"
NET_CFG="$NS3_CFG_DIR/$NS3_SUBDIR/config.txt"
TOPOLOGY_FILE="$NS3_CFG_DIR/$NS3_SUBDIR/physical_topology.txt"
LOGICAL_TOPO="$NS3_CFG_DIR/$NS3_SUBDIR/logical_topology.json"
FLOW_FILE="$NS3_CFG_DIR/add_request/flow.txt"
TRACE_FILE="$NS3_CFG_DIR/add_request/trace.txt"

# Reuse the system / remote-memory / logging configs you already use for the
# analytical backend. IMPORTANT: the system config must replicate the same 
# topology described in logical_topology.json, otherwise the simulation will fail.
SYSTEM_CFG="$BASE_DIR/configs/astra_sim/system/h100_1D.json"
REMOTE_MEM_CFG="$BASE_DIR/configs/astra_sim/no_remote_memory.json"
LOGGING_CFG="$BASE_DIR/configs/astra_sim/logging_config.toml"

MLSYNTH_CFG_PATH="$BASE_DIR/configs/mlsynth/$MODEL_FOLDER/$CONFIG_FILE.yaml"

# ESSENTIAL CHECKS
if [ ! -f "$MLSYNTH_CFG_PATH" ]; then
  echo "ERROR: Config file not found at $MLSYNTH_CFG_PATH"
  exit 1
fi
if [ ! -x "$NS3_BIN" ]; then
  echo "ERROR: ns-3 binary not found / not executable at:"
  echo "       $NS3_BIN"
  exit 1
fi
for f in "$NET_CFG" "$TOPOLOGY_FILE" "$LOGICAL_TOPO" "$FLOW_FILE" "$TRACE_FILE"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: required ns-3 config file missing: $f"
    exit 1
  fi
done

# 4. Output paths
TRACES_DIR="$BASE_DIR/output/mlsynth"
WORKLOAD_PREFIX="$TRACES_DIR/$MODEL_NAME/et/$MODEL_NAME"
COMM_GROUPS="$TRACES_DIR/$MODEL_NAME/comm_groups.json"
NS3_OUT_DIR="$BASE_DIR/output/ns3/$NS3_SUBDIR"
mkdir -p "$NS3_OUT_DIR"

# --- EXECUTION ---

# Step 1: MLSynth (identical to the analytical flow)
echo "==> [1/2] Running MLSynth to generate traces for model: $MODEL_NAME"
python3 "$MLSYNTH_SCRIPT" -c "$MLSYNTH_CFG_PATH" -o "$TRACES_DIR"
if [ $? -ne 0 ]; then
  echo "ERROR: MLSynth execution failed. Check the python errors above. Exiting."
  exit 1
fi
echo "==> Traces successfully generated in $TRACES_DIR/$MODEL_NAME"

# Sanity check: the number of NPUs (one .et per NPU) must equal the number of
# COMPUTE nodes in physical_topology.txt, and logical_topology.json must use that count.
NUM_NPUS=$(find "$TRACES_DIR/$MODEL_NAME/et/" -maxdepth 1 -name "${MODEL_NAME}.*.et" | wc -l | tr -d ' ')
echo "==> Detected $NUM_NPUS NPU trace file(s) (.et)."
echo "    -> physical_topology.txt must have exactly $NUM_NPUS compute nodes"
echo "    -> logical_topology.json must use [\"$NUM_NPUS\"]"
echo ""

echo "==> [2/2] Starting ASTRA-sim (ns-3 backend)..."
"$NS3_BIN" \
  --workload-configuration="$WORKLOAD_PREFIX" \
  --system-configuration="$SYSTEM_CFG" \
  --network-configuration="$NET_CFG" \
  --remote-memory-configuration="$REMOTE_MEM_CFG" \
  --logical-topology-configuration="$LOGICAL_TOPO" \
  --comm-group-configuration="$COMM_GROUPS" \
  --logging-configuration="$LOGGING_CFG"
RC=$?

if [ $RC -ne 0 ]; then
  echo "==> ERROR: ASTRA-sim ns-3 backend exited with code $RC."
  echo "    Check the terminal output above for the failing argument/file."
  if [ -d "$BASE_DIR/log" ]; then
    rm -rf "$BASE_DIR/log"
    echo "==> NOTE: Removed the 'log' folder produced by ASTRA-sim due to the error."
  fi
  exit 1
fi

# Step 4: Log management (same logic as the analytical script)
echo "==> Simulation finished. Moving logs..."
if [ -d "$BASE_DIR/log" ]; then
  if [ -d "$BASE_DIR/output/astra_logs/${OUTPUT_DIR}" ]; then
    rm -rf "$BASE_DIR/output/astra_logs/${OUTPUT_DIR}"
  fi
  mkdir -p "$BASE_DIR/output/astra_logs"
  mv "$BASE_DIR/log" "$BASE_DIR/output/astra_logs/${OUTPUT_DIR}"
  echo "==> SUCCESS: System-layer logs saved in $BASE_DIR/output/astra_logs/${OUTPUT_DIR}"
else
  echo "==> NOTE: no 'log' folder was produced in $BASE_DIR."
fi

echo "==> ns-3 packet-level outputs: $NS3_OUT_DIR"
echo "---------------------------------------------------"