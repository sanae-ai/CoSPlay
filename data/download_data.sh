#!/usr/bin/env bash
set -euo pipefail

# Dataset download wrapper for CoSPlay benchmark JSON files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

REPO_ID="${REPO_ID:-yomi017/CosPlay}"
GROUP="${GROUP:-full-dataset-chunk}"
DATASETS="${DATASETS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/CURE_data}"

case "${GROUP}" in
  full-dataset-chunk)
    DOWNLOAD_GROUP="full-dataset-chunked"
    ;;
  full-dataset)
    DOWNLOAD_GROUP="full-dataset-complete"
    ;;
  small-dataset)
    DOWNLOAD_GROUP="${GROUP}"
    ;;
  *)
    echo "[download_data] unknown GROUP=${GROUP}" >&2
    echo "[download_data] expected one of: full-dataset, full-dataset-chunk, small-dataset" >&2
    exit 2
    ;;
esac

ARGS=(
  --repo-id "${REPO_ID}"
  --group "${DOWNLOAD_GROUP}"
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

echo "[download_data] repo: ${REPO_ID}"
echo "[download_data] group: ${GROUP}"
if [[ -n "${DATASETS}" ]]; then
  echo "[download_data] datasets: ${DATASETS}"
fi
echo "[download_data] output dir: ${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/download_data.py" "${ARGS[@]}"
