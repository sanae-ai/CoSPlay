#!/usr/bin/env bash
set -euo pipefail

# Dataset download wrapper for CoSPlay benchmark JSON files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

REPO_ID="${REPO_ID:-yomi017/CosPlay}"
GROUP="${GROUP:-full-dataset-chunked}"
DATASETS="${DATASETS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/CURE_data}"
REVISION="${REVISION:-}"
CACHE_DIR="${CACHE_DIR:-}"
LIST_ONLY="${LIST_ONLY:-False}"
FORCE="${FORCE:-False}"

ARGS=(
  --repo-id "${REPO_ID}"
  --group "${GROUP}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${DATASETS}" ]]; then
  IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"
  ARGS+=(--dataset)
  for dataset in "${DATASET_LIST[@]}"; do
    if [[ -n "${dataset}" ]]; then
      ARGS+=("${dataset}")
    fi
  done
fi

if [[ -n "${REVISION}" ]]; then
  ARGS+=(--revision "${REVISION}")
fi

if [[ -n "${CACHE_DIR}" ]]; then
  ARGS+=(--cache-dir "${CACHE_DIR}")
fi

if [[ "${LIST_ONLY}" == "True" || "${LIST_ONLY}" == "true" || "${LIST_ONLY}" == "1" ]]; then
  ARGS+=(--list)
fi

if [[ "${FORCE}" == "True" || "${FORCE}" == "true" || "${FORCE}" == "1" ]]; then
  ARGS+=(--force)
fi

echo "[download_data] repo: ${REPO_ID}"
echo "[download_data] group: ${GROUP}"
if [[ -n "${DATASETS}" ]]; then
  echo "[download_data] datasets: ${DATASETS}"
fi
echo "[download_data] output dir: ${OUTPUT_DIR}"
echo "[download_data] list only: ${LIST_ONLY}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/download_data.py" "${ARGS[@]}"
