#!/usr/bin/env bash
set -euo pipefail

# Signal recomputation wrapper.
# Reads generated-UT and code-candidate JSON directories, executes generated UTs
# against the code pool, and recomputes BoN plus optional Cluster metrics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

CASE_DIR="${CASE_DIR:-${REPO_ROOT}/temp_data/main/Cosplay-14b/Cosplay_CodeContests_14b_0/self_play_v2_rounds}"
CODE_DIR="${CODE_DIR:-${REPO_ROOT}/temp_data/main/Qwen2.5-7B-Ins/Qwen2_5_Instruct_CodeContests_7b_0}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/signal}"

OUTPUTS_PREFIX="${OUTPUTS_PREFIX:-CODE_Qwen2.5_Ins_UT_CoSPlay_round_05_CodeContests}"
MODE="${MODE:-CODE_Qwen2.5_Ins_UT_CoSPlay_round_05_CodeContests}"
RESULT_SUBDIR="${RESULT_SUBDIR:-signal}"

K_CODE="${K_CODE:-16}"
K_CASE="${K_CASE:-16}"
CASE_CONTAINS="${CASE_CONTAINS:-round_05}"
CODE_CONTAINS="${CODE_CONTAINS:-}"
GENERATION_MODE="${GENERATION_MODE:-exp-atk}"
COMPUTE_CLUSTER="${COMPUTE_CLUSTER:-True}"
STRICT="${STRICT:-True}"

ARGS=(
  --case_dir "${CASE_DIR}"
  --code_dir "${CODE_DIR}"
  --out_dir "${OUT_DIR}"
  --outputs_prefix "${OUTPUTS_PREFIX}"
  --mode "${MODE}"
  --result_subdir "${RESULT_SUBDIR}"
  --k_code "${K_CODE}"
  --k_case "${K_CASE}"
  --case_contains "${CASE_CONTAINS}"
  --generation_mode "${GENERATION_MODE}"
  --compute_cluster "${COMPUTE_CLUSTER}"
)

if [[ -n "${CODE_CONTAINS}" ]]; then
  ARGS+=(--code_contains "${CODE_CONTAINS}")
fi

if [[ "${STRICT}" == "True" || "${STRICT}" == "true" || "${STRICT}" == "1" ]]; then
  ARGS+=(--strict)
else
  ARGS+=(--no_strict)
fi

echo "[signal] case dir: ${CASE_DIR}"
echo "[signal] code dir: ${CODE_DIR}"
echo "[signal] out dir: ${OUT_DIR}"
echo "[signal] mode: ${MODE}"
echo "[signal] generation mode: ${GENERATION_MODE}"
echo "[signal] compute cluster: ${COMPUTE_CLUSTER}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/signal.py" "${ARGS[@]}"
