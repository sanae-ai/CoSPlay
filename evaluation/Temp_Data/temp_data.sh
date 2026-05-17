#!/usr/bin/env bash
set -euo pipefail

# Offline temp_data metric recomputation.
# Reads downloaded temp_data JSON matrices and recomputes paper metrics without
# regenerating model outputs or re-executing candidate code.
#
# Expected downloaded roots:
#   temp_data/main            Full Dataset main-table runs
#   temp_data/generalization  Small Dataset transfer/generalization runs
#   temp_data/scaling         k={2,4,8,16,32,64} 7B scaling runs
#   temp_data/tts             full k=16 CoSPlay reference / TTS-related runs
#
# Important JSON fields:
#   generated_code            sampled code candidates
#   case_input                generated unit-test inputs
#   case_bool_table           candidate x generated-UT execution matrix for BoN
#   case_is_valid             optional generated-UT validity mask
#   test_bool_table           candidate x held-out-test matrix for pass@k
#   new_bon_cluster_info      cached output-consensus selector metadata
#
# Paper-facing naming:
#   new_bon_cluster_info["new_bon"] -> cluster_acc / cluster_accumulate
#   new_bon_front and new_bon_back  -> debug only, not used in paper results
#
# Input file patterns:
#   outputs_results_eval_*.json
#   self_play_v2_rounds/round_XX_results_eval_*.json
# Use ROUND_ID=05 for the final saved self-play round in round ablations.
#
# Example usages:
#   bash evaluation/Temp_Data/temp_data.sh
#   ROOT=/path/to/temp_data/scaling OUT_DIR=outputs/scaling_metrics bash evaluation/Temp_Data/temp_data.sh
#   ROOT=/path/to/temp_data/tts KIND=round ROUND_ID=05 OUT_DIR=outputs/tts_round05_metrics bash evaluation/Temp_Data/temp_data.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Python executable. Override with PYTHON_BIN=python3 if needed.
PYTHON_BIN="${PYTHON_BIN:-python}"

# Path passed to temp_data.py.
# It can be:
# - the full temp_data root;
# - one split, e.g. temp_data/scaling;
# - one setting, e.g. temp_data/scaling/Cosplay_7b_mix_k16;
# - one JSON file.
TEMP_DATA_ROOT="${TEMP_DATA_ROOT:-${REPO_ROOT}/temp_data}"
ROOT="${ROOT:-${TEMP_DATA_ROOT}}"

# Output directory for:
# - per_file_metrics.csv
# - per_run_metrics.csv
# - per_setting_metrics.csv
# - metrics_summary.json
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/temp_data_metrics}"

# Which JSON files to read:
# - final: outputs_results_eval_*.json
# - round: self_play_v2_rounds/round_XX_results_eval_*.json
# - both: both final and round files
KIND="${KIND:-final}"

# Round id used when KIND=round. Use "05" for final self-play snapshots.
# Leave empty to include every saved round.
ROUND_ID="${ROUND_ID:-}"

# Optional comma-separated split filter. Valid values:
# main,generalization,scaling,tts
# Example: SPLITS=scaling,tts
SPLITS="${SPLITS:-}"

# Optional comma-separated setting directory filter.
# Example: SETTINGS=Cosplay_7b_mix_k16,Cosplay_14b_mix_k16
SETTINGS="${SETTINGS:-}"

# pass@k values to recompute from test_bool_table.
PASS_AT_K="${PASS_AT_K:-1,2,4,8,16,32,64}"

# BoN scales to compute from case_bool_table.
# Use "auto" to compute every supported K up to the saved matrix size.
# Other examples: SCALES=16 or SCALES=2,4,8,16 or SCALES=16x32.
SCALES="${SCALES:-auto}"

# Debug limit. 0 means process every matching JSON file.
MAX_FILES="${MAX_FILES:-0}"

ARGS=(
  --root "${ROOT}"
  --out-dir "${OUT_DIR}"
  --kind "${KIND}"
  --pass-at-k "${PASS_AT_K}"
  --scales "${SCALES}"
)

if [[ -n "${ROUND_ID}" ]]; then
  ARGS+=(--round "${ROUND_ID}")
fi

if [[ -n "${SPLITS}" ]]; then
  IFS=',' read -r -a SPLIT_LIST <<< "${SPLITS}"
  for split in "${SPLIT_LIST[@]}"; do
    if [[ -n "${split}" ]]; then
      ARGS+=(--split "${split}")
    fi
  done
fi

if [[ -n "${SETTINGS}" ]]; then
  IFS=',' read -r -a SETTING_LIST <<< "${SETTINGS}"
  for setting in "${SETTING_LIST[@]}"; do
    if [[ -n "${setting}" ]]; then
      ARGS+=(--setting "${setting}")
    fi
  done
fi

if [[ "${MAX_FILES}" != "0" ]]; then
  ARGS+=(--max-files "${MAX_FILES}")
fi

echo "[temp_data] repo root: ${REPO_ROOT}"
echo "[temp_data] root: ${ROOT}"
echo "[temp_data] kind: ${KIND}"
if [[ -n "${ROUND_ID}" ]]; then
  echo "[temp_data] round: ${ROUND_ID}"
fi
echo "[temp_data] out dir: ${OUT_DIR}"
echo "[temp_data] pass@k: ${PASS_AT_K}"
echo "[temp_data] scales: ${SCALES}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/temp_data.py" "${ARGS[@]}"
