#!/usr/bin/env bash

# Stop immediately when a command fails so the first incompatible CUDA/package
# step is visible instead of being hidden by later install errors.
set -euo pipefail

# Let users override the environment name without editing this script.
ENV_NAME="${CONDA_ENV_NAME:-CosPlay}"

# Keep Python pinned to the version used in our experiments.
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Torch wheels are CUDA-specific; override this URL for a different CUDA stack.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# The vLLM wheel is pinned to CUDA 12.8; replace it if your GPU driver/CUDA differs.
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0+cu128-cp38-abi3-manylinux_2_35_x86_64.whl}"

# vLLM may still resolve CUDA PyTorch dependencies from this extra index.
VLLM_EXTRA_INDEX_URL="${VLLM_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# flash-attn compiles native code; lowering MAX_JOBS can avoid RAM/CPU pressure.
export MAX_JOBS="${MAX_JOBS:-4}"

# Find conda in the current shell; non-interactive bash does not always load it.
if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH. Please install Miniconda/Anaconda or load conda first." >&2
  exit 1
fi

# Initialize conda for non-interactive bash; this is the most portable way to
# make 'conda activate' available when running this script with 'bash setup_env.sh'.
if ! eval "$(conda shell.bash hook)"; then
  # Fall back to sourcing conda.sh directly for older conda installations.
  CONDA_BASE="$(conda info --base)"

  # Git Bash on Windows may return a C:\... path, which bash cannot source
  # unless it is converted to /c/... first.
  if command -v cygpath >/dev/null 2>&1; then
    CONDA_BASE="$(cygpath -u "${CONDA_BASE}")"
  fi

  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

# Reuse an existing environment when present so rerunning the script is practical.
if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda environment '${ENV_NAME}' already exists; reusing it."
else
  # Create the isolated reproduction environment with the pinned Python version.
  conda create --name "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

# Activate the environment after loading conda.sh; this is the bash-safe form.
conda activate "${ENV_NAME}"

# Use the environment's Python explicitly, which avoids accidentally calling system pip.
python -m pip install --upgrade pip

# Install PyTorch first because xformers, flashinfer, vLLM, and flash-attn are CUDA-sensitive.
python -m pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url "${TORCH_INDEX_URL}"

# Install the repository's regular Python dependencies after PyTorch is available.
python -m pip install -r requirements.txt

# Install xformers separately because its wheels are tightly coupled to PyTorch/CUDA.
python -m pip install xformers==0.0.33.post1

# Install flashinfer separately because available wheels depend on GPU/CUDA support.
python -m pip install flashinfer-python==0.5.3

# Install the pinned vLLM CUDA 12.8 wheel; change VLLM_WHEEL_URL for other platforms.
python -m pip install --no-cache-dir "${VLLM_WHEEL_URL}" \
  --extra-index-url "${VLLM_EXTRA_INDEX_URL}"

# Install flash-attn last because it may compile locally and is the most machine-sensitive.
python -m pip install flash-attn==2.8.3 --no-build-isolation

# Print the active Python and torch/CUDA state to make support logs easier to read.
python - <<'PY'
import torch

print("Environment setup finished.")
print(f"Torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version reported by torch: {torch.version.cuda}")
PY
