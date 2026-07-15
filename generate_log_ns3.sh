#!/usr/bin/env bash

# Phase 2/2: ns-3/ASTRA-sim execution only.
#
# Requires MLSynth traces to already exist for MODEL_FILE_NAME (run
# generate_traces.sh once beforehand). Safe to run in parallel across
# different NS3_SUBDIR values; do NOT run two instances with the SAME
# NS3_SUBDIR (or the same OUTPUT_DIR_NAME) concurrently, since their
# ns-3 output files and logging folder would collide.

# 1. Ask for user input (supports command line args or interactive prompt)
MODEL_FILE_NAME="$1"
NS3_SUBDIR="$2"
OUTPUT_DIR_NAME="$3"

if [ -t 0 ]; then
  [ -z "$MODEL_FILE_NAME" ] && read -p "Enter the model config file name without extension (e.g., bert_base): " MODEL_FILE_NAME
  [ -z "$NS3_SUBDIR" ] && read -p "Enter the name of the ns3 configuration subdirectory: " NS3_SUBDIR
  [ $# -lt 3 ] && read -p "Enter the output directory name (leave blank to use <model>/<ns3subdir>): " OUTPUT_DIR_NAME
else
  if [ -z "$MODEL_FILE_NAME" ] || [ -z "$NS3_SUBDIR" ]; then
    echo "ERROR: No terminal attached to prompt for input, and required arguments are missing."
    echo "Usage: $0 <model_file_name> <ns3_subdir> [output_dir_name]"
    exit 1
  fi
fi

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

NS3_DIR="$BASE_DIR/../astra-sim/extern/network_backend/ns-3"
NS3_BIN="$NS3_DIR/build/scratch/ns3.42-AstraSimNetwork-default"

# 3. Configurations
NS3_CFG_DIR="$BASE_DIR/configs/astra_sim/ns3"
NET_CFG="$NS3_CFG_DIR/$NS3_SUBDIR/config.txt"
TOPOLOGY_FILE="$NS3_CFG_DIR/$NS3_SUBDIR/physical_topology.txt"
LOGICAL_TOPO="$NS3_CFG_DIR/$NS3_SUBDIR/logical_topology.json"
FLOW_FILE="$NS3_CFG_DIR/additional_configs/add_flow.txt"
TRACE_FILE="$NS3_CFG_DIR/additional_configs/trace_link.txt"

SYSTEM_CFG="$BASE_DIR/configs/astra_sim/system/h100_1D.json"
REMOTE_MEM_CFG="$BASE_DIR/configs/astra_sim/no_remote_memory.json"
LOGGING_CFG="$BASE_DIR/configs/astra_sim/logging_config.toml"

# 4. Define output paths
TRACES_DIR="$BASE_DIR/output/mlsynth"
WORKLOAD_PREFIX="$TRACES_DIR/$MODEL_FILE_NAME/et/$MODEL_FILE_NAME"
COMM_GROUPS="$TRACES_DIR/$MODEL_FILE_NAME/comm_groups.json"
NS3_OUT_DIR="$BASE_DIR/output/ns3/$NS3_SUBDIR"
OUTPUT_DIR="$BASE_DIR/output/astra_logs/${OUTPUT_DIR_NAME}"

# ESSENTIAL CHECK 2: Verify all required files and binaries exist
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

# ESSENTIAL CHECK 3: MLSynth traces must already exist for this model
# (run ./generate_traces.sh "$MODEL_FILE_NAME" once beforehand)
if ! find "$TRACES_DIR/$MODEL_FILE_NAME/et/" -maxdepth 1 -name "${MODEL_FILE_NAME}.*.et" -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: No MLSynth traces found for model '$MODEL_FILE_NAME' in:"
  echo "       $TRACES_DIR/$MODEL_FILE_NAME/et/"
  echo "       Run './generate_traces.sh $MODEL_FILE_NAME' first."
  exit 1
fi

if [ ! -f "$COMM_GROUPS" ]; then
  echo "ERROR: comm_groups.json not found at $COMM_GROUPS"
  echo "       Run './generate_traces.sh $MODEL_FILE_NAME' first."
  exit 1
fi

# --- EXECUTION ---

# Ensure we run from the root
cd "$BASE_DIR" || exit

if [ -d "$OUTPUT_DIR" ]; then
  rm -rf "$OUTPUT_DIR"
  echo "Existing output directory $OUTPUT_DIR removed."
fi
mkdir -p "$NS3_OUT_DIR"

echo "==> Starting ASTRA-sim (ns-3 backend)..."
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