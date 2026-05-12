#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${REPO_ROOT}/CURE_data"

cd "${SCRIPT_DIR}"

# Optional runtime overrides:
# - PYTHON_BIN=python3 uses a different Python executable.
# - CONDA_ENV_NAME=cosplay activates a conda environment before running.
# - MODEL=/path/to/model or MODEL=org/model changes the evaluated model.
# - CUDA_VISIBLE_DEVICES and GPU_GROUPS control the vLLM engine placement.
PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

if [[ -n "${CONDA_ENV_NAME}" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

rm -rf "${HOME}/.cache/torch_extensions"

# Use one isolated compile/cache directory per run. This avoids cache conflicts
# when several evaluation jobs are launched on the same machine.
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
JOB_CACHE_DIR="${JOB_CACHE_DIR:-${REPO_ROOT}/.cache/${RUN_ID}}"

cleanup() {
  if [[ -d "${JOB_CACHE_DIR}" ]]; then
    rm -rf "${JOB_CACHE_DIR}"
  fi
}
trap cleanup EXIT

mkdir -p "${JOB_CACHE_DIR}"

export TRITON_CACHE_DIR="${JOB_CACHE_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_CACHE_DIR}/torchinductor"
export PYTORCH_KERNEL_CACHE_PATH="${JOB_CACHE_DIR}/kernels"
export VLLM_CONFIG_ROOT="${JOB_CACHE_DIR}/vllm_config"
export VLLM_TORCH_COMPILE_CACHE_DIR="${JOB_CACHE_DIR}/vllm_torch_compile_cache"
mkdir -p \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${PYTORCH_KERNEL_CACHE_PATH}" \
  "${VLLM_CONFIG_ROOT}" \
  "${VLLM_TORCH_COMPILE_CACHE_DIR}"

# Model under evaluation. The public default is used for open-source release;
# set MODEL to a local checkpoint path when reproducing internal runs.
MODELS=(
  "${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
)

# Dataset names are file stems under CURE_data. main.py expands
# each name to ../CURE_data/<dataset>.json, so do not pass absolute paths here.
DATASETS=(
  "CodeContests_chunk_0"
  "CodeContests_chunk_1"
  "CodeContests_chunk_2"
  "CodeContests_chunk_3"
  "CodeContests_chunk_4"
  "CodeForces_chunk_0"
  "CodeForces_chunk_1"
  "CodeForces_chunk_2"
  "CodeForces_chunk_3"
  "CodeForces_chunk_4"
  "CodeForces_chunk_5"
  "CodeForces_chunk_6"
  "CodeForces_chunk_7"
  "CodeForces_chunk_8"
  "CodeForces_chunk_9"
  "LiveBench_chunk_0"
  "LiveBench_chunk_1"
  "LiveBench_chunk_2"
  "LiveCodeBench_chunk_0"
  "LiveCodeBench_chunk_1"
  "LiveCodeBench_chunk_2"
  "LiveCodeBench_chunk_3"
  "LiveCodeBench_chunk_4"
  "LiveCodeBench_chunk_5"
  "LiveCodeBench_chunk_6"
  "LiveCodeBench_chunk_7"
  "LiveCodeBench_chunk_8"
  "LiveCodeBench_chunk_9"
  "LiveCodeBench_chunk_10"
)

# Final CosPlay setting: generate 16 candidate programs and 16 candidate tests.
SINGLE_EVAL=False
USE_API=False
K_CODE="${K_CODE:-16}"
K_CASE="${K_CASE:-16}"

# Final BoN sweep evaluated from the same k=16 generation budget.
SCALE_TUPLE_LIST="${SCALE_TUPLE_LIST:-[(2, 2), (4, 4), (8, 8), (16, 16)]}"
PASS_AT_K_LIST="${PASS_AT_K_LIST:-[1,2,4,8,16]}"
GENERATION_MODE="${GENERATION_MODE:-plansearch}"
EVAL_MODE="${EVAL_MODE:-bon}"

# Final method configuration. These flags describe the reported CosPlay setup:
# direct second-order-observation code generation, self-play, idea-level attack
# unit tests, and all second-order observations enabled.
MAX_OBS="${MAX_OBS:-4}"
PROMPT_ROLE_MODE="${PROMPT_ROLE_MODE:-3}"
ABLATION="${ABLATION:-only_stage2}"
USE_ALL_SECOND_ORDER_OBS="${USE_ALL_SECOND_ORDER_OBS:-True}"
USE_IDEA_ATTACK_UT="${USE_IDEA_ATTACK_UT:-True}"
SELF_CONSISTENCY_NUM="${SELF_CONSISTENCY_NUM:-4}"
UT_VOTE_BY_CODE="${UT_VOTE_BY_CODE:-False}"
USE_SELF_PLAY="${USE_SELF_PLAY:-True}"
SELF_PLAY_ROUND="${SELF_PLAY_ROUND:-5}"
VERBOSE_LOGGING="${VERBOSE_LOGGING:-False}"
UT_ACCURACY_TARGET="${UT_ACCURACY_TARGET:-0.5}"
UT_REGEN_MAX_ATTEMPTS="${UT_REGEN_MAX_ATTEMPTS:-5}"

# GPU layout. The default starts two one-GPU vLLM engines on GPU 0 and GPU 1.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_GROUPS="${GPU_GROUPS:-[[0],[1]]}"
export CUDA_VISIBLE_DEVICES

# Repeated experiments. Use a comma-separated list, for example:
#   REPEAT_IDS=1,2,3 bash evaluation/eval.sh
# An empty repeat id keeps MODE exactly equal to MODE_PREFIX.
MODE_PREFIX="${MODE_PREFIX:-Cosplay_7b}"
REPEAT_IDS="${REPEAT_IDS:-}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/CURE_logs/final_eval_logs/final/Cosplay_7b}"

IFS=',' read -r -a RUN_SUFFIXES <<< "${REPEAT_IDS}"
if [[ ${#RUN_SUFFIXES[@]} -eq 0 ]]; then
  RUN_SUFFIXES=("")
fi

for dataset in "${DATASETS[@]}"; do
  if [[ ! -f "${DATA_DIR}/${dataset}.json" ]]; then
    echo "Missing dataset file: ${DATA_DIR}/${dataset}.json" >&2
    exit 1
  fi
done

for suffix in "${RUN_SUFFIXES[@]}"; do
  if [[ -n "${suffix}" ]]; then
    MODE="${MODE_PREFIX}_${suffix}"
  else
    MODE="${MODE_PREFIX}"
  fi

  LOG_DIR="${LOG_ROOT}/${MODE}"
  mkdir -p "${LOG_DIR}"

  for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      if [[ "${model}" == /* || "${model}" == ./* || "${model}" == ../* ]]; then
        base_model_name="$(basename "${model}")"
      else
        base_model_name="${model}"
      fi
      sanitized_model_name="$(echo "${base_model_name}" | tr '/' '_')"
      log_dataset_name="$(echo "${dataset}" | tr '/' '_')"
      LOG_FILE="${LOG_DIR}/eval_${sanitized_model_name}_${log_dataset_name}.log"

      {
        echo
        echo "=================================================="
        echo "Model:   ${model}"
        echo "Dataset: ${dataset}"
        echo "Mode:    ${MODE}"
        echo "=================================================="

        "${PYTHON_BIN}" -u main.py \
          --use_api "${USE_API}" \
          --pretrained_model "${model}" \
          --single_eval "${SINGLE_EVAL}" \
          --dataset "${dataset}" \
          --k_code "${K_CODE}" \
          --k_case "${K_CASE}" \
          --scale_tuple_list "${SCALE_TUPLE_LIST}" \
          --gpu_groups "${GPU_GROUPS}" \
          --mode "${MODE}" \
          --generation_mode "${GENERATION_MODE}" \
          --eval_mode "${EVAL_MODE}" \
          --pass_at_k_list "${PASS_AT_K_LIST}" \
          --max_obs "${MAX_OBS}" \
          --prompt_role_mode "${PROMPT_ROLE_MODE}" \
          --ablation "${ABLATION}" \
          --use_all_second_order_obs "${USE_ALL_SECOND_ORDER_OBS}" \
          --self_consistency_num "${SELF_CONSISTENCY_NUM}" \
          --ut_vote_by_code "${UT_VOTE_BY_CODE}" \
          --use_idea_attack_ut "${USE_IDEA_ATTACK_UT}" \
          --self_play_round "${SELF_PLAY_ROUND}" \
          --use_self_play "${USE_SELF_PLAY}" \
          --ut_accuracy_target "${UT_ACCURACY_TARGET}" \
          --ut_regen_max_attempts "${UT_REGEN_MAX_ATTEMPTS}" \
          --verbose_logging "${VERBOSE_LOGGING}"

        echo
        echo "Completed ${model} on ${dataset}"
      } 2>&1 | tee "${LOG_FILE}"
    done
  done
done

echo
echo "All evaluation jobs completed."
