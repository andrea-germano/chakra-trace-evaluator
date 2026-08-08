#!/usr/bin/env bash

# Phase 2/2: ns-3/ASTRA-sim execution only.
#
# Requires MLSynth traces to already exist for MODEL_FILE_NAME (run
# generate_traces.sh once beforehand).
#
# ns-3 packet-level outputs always land at output/ns3/<OUTPUT_DIR_NAME>/,
# mirroring output/astra_logs/<OUTPUT_DIR_NAME>/ exactly, whatever
# OUTPUT_DIR_NAME is (independent of NS3_SUBDIR's own folder structure under
# configs/astra_sim/ns3/). The config carries a __RUN_DIR__ placeholder that is
# resolved here, into a per-run temporary config, just before launching ns-3.
# As a result:
#   - the SAME NS3_SUBDIR (fabric config) reused with a DIFFERENT
#     OUTPUT_DIR_NAME no longer collides (each writes to its own
#     output/ns3/<OUTPUT_DIR_NAME>/... subtree);
#   - two instances sharing the same OUTPUT_DIR_NAME still collide on ns-3
#     outputs and must not run concurrently.
#
# All-or-nothing outputs: a run leaves both output trees complete, or neither.
# Both are wiped before the run and after any failure, so a half-written run can
# never reach the analysis scripts. KEEP_FAILED=1 keeps them for post-mortem.

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

# It ends up inside rm -rf targets: keep it strictly under output/.
case "$OUTPUT_DIR_NAME" in
  /*|*..*) echo "ERROR: Output directory name must be relative and free of '..': $OUTPUT_DIR_NAME"; exit 1 ;;
esac

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
NS3_OUT_DIR="$BASE_DIR/output/ns3/$OUTPUT_DIR_NAME"
OUTPUT_DIR="$BASE_DIR/output/astra_logs/${OUTPUT_DIR_NAME}"
NS3_DROPS="$NS3_OUT_DIR/drops.txt"

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

# One .et per rank: how many stats_sys*.csv a complete run must produce.
EXPECTED_RANKS=$(find "$TRACES_DIR/$MODEL_FILE_NAME/et/" -maxdepth 1 -name "${MODEL_FILE_NAME}.*.et" | wc -l)

# --- EXECUTION ---

# Ensure we run from the root
cd "$BASE_DIR" || exit

purge_outputs() {
  rm -rf "$OUTPUT_DIR" "$NS3_OUT_DIR"
  echo "==> outputs removed ($1)"
}

purge_outputs "pre-run cleanup"
mkdir -p "$NS3_OUT_DIR"

# Resolve the __RUN_DIR__ placeholder into a per-run temporary config so the
# fabric config itself stays run-agnostic and reusable. Fail fast if the
# config was generated before the placeholder existed (otherwise ns-3 would
# silently write to a fixed path and different runs would overwrite each
# other).
if ! grep -q "__RUN_DIR__" "$NET_CFG"; then
  echo "ERROR: '$NET_CFG' has no __RUN_DIR__ placeholder."
  echo "       Regenerate it with config_generator.py (updated template), or add"
  echo "       '__RUN_DIR__' in place of the model/tag segment in its output-file lines."
  exit 1
fi

RESOLVED_CFG="$(mktemp "${TMPDIR:-/tmp}/ns3_config.${OUTPUT_DIR_NAME//\//_}.XXXXXX.txt")"
STDOUT_FIFO="$(mktemp -u "${TMPDIR:-/tmp}/ns3_stdout.${OUTPUT_DIR_NAME//\//_}.XXXXXX")"
sed "s#__RUN_DIR__#${OUTPUT_DIR_NAME}#g" "$NET_CFG" > "$RESOLVED_CFG"

# RUN_OK flips to 1 only once the run is verified complete; every other exit
# path takes the outputs down with it.
RUN_OK=0
PIDS=()

cleanup() {
  if [ ${#PIDS[@]} -gt 0 ]; then
    kill -TERM "${PIDS[@]}" 2>/dev/null
    for _ in $(seq 1 20); do
      kill -0 "${PIDS[0]}" 2>/dev/null || break
      sleep 0.05
    done
    kill -KILL "${PIDS[@]}" 2>/dev/null
  fi
  rm -f "$RESOLVED_CFG" "$STDOUT_FIFO"
  if [ "$RUN_OK" -ne 1 ]; then
    if [ "${KEEP_FAILED:-0}" = "1" ]; then
      echo "==> KEEP_FAILED=1: partial outputs kept in $OUTPUT_DIR and $NS3_OUT_DIR"
    else
      purge_outputs "run did not complete"
    fi
  fi
}
trap cleanup EXIT
trap 'echo "==> Interrupted — killing ns-3 and cleaning up."; exit 130' INT TERM HUP

echo "==> Starting ASTRA-sim (ns-3 backend)... ($EXPECTED_RANKS ranks expected)"
# Packet drops ("Headroom full" + a companion per-port state line) are printed by
# ns-3 ONLY to stdout, and a lossy run prints thousands of them. Keep the console
# clean: route the drop lines to drops.txt, discard the companion state dump, and
# let all other (info) output through to the screen. Convention: drops.txt ABSENT
# == lossless
mkfifo "$STDOUT_FIFO" || { echo "ERROR: cannot create FIFO $STDOUT_FIFO"; exit 1; }

awk -v drops="$NS3_DROPS" '
    /Headroom full/ { print > drops; expect=1; next }  # drop -> drops.txt (created here), off stdout
    expect && /^\(/ { expect=0; next }                 # its (a,b)... companion line -> discarded
    { expect=0; print }                                # info -> stdout (on screen)
  ' < "$STDOUT_FIFO" &
AWK_PID=$!

# ns-3 goes to the background so bash can sit in `wait`: while it waits on a
# FOREGROUND command it defers trap handlers, so a signal used to be queued for
# the rest of the run and the cleanup never ran.
"$NS3_BIN" \
  --workload-configuration="$WORKLOAD_PREFIX" \
  --system-configuration="$SYSTEM_CFG" \
  --network-configuration="$RESOLVED_CFG" \
  --remote-memory-configuration="$REMOTE_MEM_CFG" \
  --logical-topology-configuration="$LOGICAL_TOPO" \
  --comm-group-configuration="$COMM_GROUPS" \
  --logging-folder="$OUTPUT_DIR" > "$STDOUT_FIFO" 2>&1 &
  # --logging-configuration="$LOGGING_CFG" \
NS3_PID=$!
PIDS=("$NS3_PID" "$AWK_PID")

wait "$NS3_PID"; RC=$?
wait "$AWK_PID" 2>/dev/null   # let awk drain the FIFO and flush drops.txt
PIDS=()

FAILURE=""
GOT_RANKS=$(find "$OUTPUT_DIR" -maxdepth 1 -name 'stats_sys*.csv' 2>/dev/null | wc -l)

if [ $RC -ne 0 ]; then
  FAILURE="ns-3 backend exited with code $RC"
elif [ ! -f "$OUTPUT_DIR/log.log" ]; then
  FAILURE="log.log was never written"
elif ! grep -q "All ranks have finished" "$OUTPUT_DIR/log.log"; then
  FAILURE="log.log has no 'All ranks have finished' marker (ended early)"
elif [ "$GOT_RANKS" -ne "$EXPECTED_RANKS" ]; then
  FAILURE="only $GOT_RANKS/$EXPECTED_RANKS stats_sys*.csv produced"
elif [ -s "$OUTPUT_DIR/err.log" ]; then
  FAILURE="err.log is not empty ($(wc -c < "$OUTPUT_DIR/err.log") bytes)"
fi

if [ -n "$FAILURE" ]; then
  echo "==> ERROR: $FAILURE"
  echo "    Set KEEP_FAILED=1 to keep the partial outputs for inspection."
  exit 1
fi

if [ -f "$NS3_DROPS" ]; then
  echo "==> WARNING: $(wc -l < "$NS3_DROPS") dropped packet(s) ('Headroom full') — run is NOT lossless (see $NS3_DROPS)."
else
  echo "==> lossless: 0 dropped packets."
fi

RUN_OK=1
echo "==> SUCCESS: System-layer logs saved in $OUTPUT_DIR ($EXPECTED_RANKS/$EXPECTED_RANKS ranks)"
echo "==> ns-3 packet-level outputs: $NS3_OUT_DIR"
echo "---------------------------------------------------"
