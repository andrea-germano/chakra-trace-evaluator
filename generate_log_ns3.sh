#!/usr/bin/env bash

# 1. Ask for user input (supports command line args or interactive prompt)
MODEL_FILE_NAME="$1"
NS3_SUBDIR="$2"
OUTPUT_DIR_NAME="$3"

[ -z "$MODEL_FILE_NAME" ] && read -p "Enter the model config file name without extension (e.g., bert_base): " MODEL_FILE_NAME
[ -z "$NS3_SUBDIR" ] && read -p "Enter the name of the ns3 configuration subdirectory: " NS3_SUBDIR
[ $# -lt 3 ] && read -p "Enter the output directory name (leave blank to use <model>/<ns3subdir>): " OUTPUT_DIR_NAME

# ESSENTIAL CHECK 1: Ensure inputs are not empty
if [ -z "$MODEL_FILE_NAME" ] || [ -z "$NS3_SUBDIR" ]; then
  echo "ERROR: Model config file name and ns3 subdirectory are required. Exiting."
  exit 1
fi

if [ -z "$OUTPUT_DIR_NAME" ]; then
  OUTPUT_DIR_NAME="${MODEL_FILE_NAME}/${NS3_SUBDIR}"
  echo "Output directory not specified, using: $OUTPUT_DIR_NAME"
fi

echo "---------------------------------------------------"

# 2. Paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MLSYNTH_SCRIPT="$BASE_DIR/../mlsynth/synthesise_inference.py"
NS3_DIR="$BASE_DIR/../astra-sim/extern/network_backend/ns-3"
NS3_BIN="$NS3_DIR/build/scratch/ns3.42-AstraSimNetwork-default"

# 3. Configurations
NS3_CFG_DIR="$BASE_DIR/configs/astra_sim/ns3"
NET_CFG="$NS3_CFG_DIR/$NS3_SUBDIR/config.txt"
TOPOLOGY_FILE="$NS3_CFG_DIR/$NS3_SUBDIR/physical_topology.txt"
LOGICAL_TOPO="$NS3_CFG_DIR/$NS3_SUBDIR/logical_topology.json"
FLOW_FILE="$NS3_CFG_DIR/add_request/flow.txt"
TRACE_FILE="$NS3_CFG_DIR/add_request/trace.txt"

SYSTEM_CFG="$BASE_DIR/configs/astra_sim/system/h100_1D.json"
REMOTE_MEM_CFG="$BASE_DIR/configs/astra_sim/no_remote_memory.json"
LOGGING_CFG="$BASE_DIR/configs/astra_sim/logging_config.toml"

MLSYNTH_CFG_PATH="$BASE_DIR/configs/mlsynth/${MODEL_FILE_NAME}.yaml"

# 4. Define output paths
TRACES_DIR="$BASE_DIR/output/mlsynth"
WORKLOAD_PREFIX="$TRACES_DIR/$MODEL_FILE_NAME/et/$MODEL_FILE_NAME"
COMM_GROUPS="$TRACES_DIR/$MODEL_FILE_NAME/comm_groups.json"
NS3_OUT_DIR="$BASE_DIR/output/ns3/$NS3_SUBDIR"
OUTPUT_DIR="$BASE_DIR/output/astra_logs/${OUTPUT_DIR_NAME}"

# ESSENTIAL CHECK 2: Verify all required files and binaries exist
if [ ! -f "$MLSYNTH_CFG_PATH" ]; then
  echo "ERROR: MLSynth config file not found at $MLSYNTH_CFG_PATH"
  exit 1
fi

if [ ! -x "$NS3_BIN" ]; then
  echo "ERROR: ns-3 binary not found or not executable at:"
  echo "       $NS3_BIN"
  exit 1
fi

for f in "$SYSTEM_CFG" "$REMOTE_MEM_CFG" "$NET_CFG" "$TOPOLOGY_FILE" "$LOGICAL_TOPO" "$FLOW_FILE" "$TRACE_FILE"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Required config file missing: $f"
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

# Sanity check: the number of NPUs (one .et per NPU) must equal the number of
# COMPUTE nodes in physical_topology.txt, and logical_topology.json must use that count.
NUM_NPUS=$(find "$TRACES_DIR/$MODEL_FILE_NAME/et/" -maxdepth 1 -name "${MODEL_FILE_NAME}.*.et" | wc -l | tr -d ' ')
echo "==> Detected $NUM_NPUS NPU trace file(s) (.et)."
echo "    -> physical_topology.txt must have exactly $NUM_NPUS compute nodes"
echo "    -> logical_topology.json must use [\"$NUM_NPUS\"]"
echo ""

# Ensure we run from the root
cd "$BASE_DIR" || exit

if [ -d "$OUTPUT_DIR" ]; then
  rm -rf "$OUTPUT_DIR"
  echo "Existing output directory $OUTPUT_DIR removed."
fi
mkdir -p "$NS3_OUT_DIR"

# Step 2: ASTRA-sim
echo "==> [2/2] Starting ASTRA-sim (ns-3 backend)..."
"$NS3_BIN" \
  --workload-configuration="$WORKLOAD_PREFIX" \
  --system-configuration="$SYSTEM_CFG" \
  --network-configuration="$NET_CFG" \
  --remote-memory-configuration="$REMOTE_MEM_CFG" \
  --logical-topology-configuration="$LOGICAL_TOPO" \
  --comm-group-configuration="$COMM_GROUPS" \
  --logging-folder="$OUTPUT_DIR" 
  # --logging-configuration="$LOGGING_CFG" \
RC=$?

if [ $RC -ne 0 ]; then
  echo "==> ERROR: ASTRA-sim ns-3 backend exited with code $RC."
  echo "    Check the terminal output above for the failing argument/file."
  if [ -d "$OUTPUT_DIR" ]; then
    rm -rf "$OUTPUT_DIR"
    echo "==> NOTE: Removed the 'log' folder produced by ASTRA-sim due to the error."
  fi
  exit 1
fi

echo "==> SUCCESS: System-layer logs saved in $OUTPUT_DIR"
echo "==> ns-3 packet-level outputs: $NS3_OUT_DIR"
echo "---------------------------------------------------"