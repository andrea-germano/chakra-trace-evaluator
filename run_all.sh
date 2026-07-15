#!/usr/bin/env bash

# Convenience wrapper: runs the full pipeline (MLSynth trace generation +
# ns-3/ASTRA-sim execution) sequentially for a single job, equivalent to the
# old all-in-one generate_log_ns3.sh.
#
# For sweeps (e.g. several ns3 subdirs for the same model, possibly in
# parallel), call generate_traces.sh once and then generate_log_ns3.sh
# per subdir instead -- running MLSynth once and reusing its output avoids
# the corruption risk of concurrent writers to the same trace directory.

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

if [ -z "$MODEL_FILE_NAME" ] || [ -z "$NS3_SUBDIR" ]; then
  echo "ERROR: Model config file name and ns3 subdirectory are required. Exiting."
  exit 1
fi

if [ -z "$OUTPUT_DIR_NAME" ]; then
  OUTPUT_DIR_NAME="${MODEL_FILE_NAME}/${NS3_SUBDIR}"
  echo "Output directory not specified, using: $OUTPUT_DIR_NAME"
fi

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$BASE_DIR/generate_traces.sh" "$MODEL_FILE_NAME" || exit 1
"$BASE_DIR/generate_log_ns3.sh" "$MODEL_FILE_NAME" "$NS3_SUBDIR" "$OUTPUT_DIR_NAME"
