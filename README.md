# CosPlay

<p align="center">
  <a href="#citation"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/yomi017/CosPlay"><img src="https://img.shields.io/badge/Hugging%20Face-Data%20%26%20Logs-yellow?logo=huggingface" alt="Data and logs on Hugging Face"></a>
  <a href="#evaluation"><img src="https://img.shields.io/badge/Evaluation-eval.sh-blue" alt="Evaluation"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-TODO-lightgrey" alt="License"></a>
</p>

CosPlay is a test-time scaling (TTS) method for code generation. It uses LLM-generated unit tests, self-play refinement, and best-of-N selection to improve a base model at inference time without releasing or requiring a separate trained model.

The current release provides the evaluation code and scripts for reproducing the CosPlay pipeline over programming benchmarks split into smaller JSON chunks. Generated data and logs are hosted on Hugging Face.

## Method Overview

CosPlay evaluates code generation with an iterative test-time pipeline:

1. Generate candidate reasoning paths and programs with PlanSearch-style prompting.
2. Generate candidate unit tests for each task.
3. Execute generated programs against public, generated, and ground-truth tests.
4. Use self-play rounds to refine unit-test feedback and candidate selection.
5. Select final answers with BoN evaluation over multiple code/test budgets.

The default configuration in `evaluation/eval.sh` is the final CosPlay setting used by this repository:

| Setting | Default |
| --- | --- |
| Code candidates | `K_CODE=16` |
| Unit-test candidates | `K_CASE=16` |
| Self-play rounds | `SELF_PLAY_ROUND=5` |

## Data and Logs

This repository does not release trained models. CosPlay is a TTS method that can be applied to a chosen base model at evaluation time.

Generated data and evaluation logs are available on Hugging Face:

```text
TODO
```

The paper contains the official results. The paper link will be added when available.

## Getting Started

Clone the repository:

```bash
git clone TODO
cd CosPlay
```

Create an environment:

```bash
conda create -n cosplay python=3.10
conda activate cosplay
```

No pinned environment file is included yet. Expected packages include:

```bash
pip install torch transformers vllm openai numpy termcolor jinja2
```

Depending on your CUDA, PyTorch, vLLM, and cluster setup, you may need to install these packages from the official wheels for your platform.

## Evaluation Data

CosPlay expects evaluation data under `CURE_data/`. The evaluation script passes dataset names as file stems, and `main_self_play_v3.py` resolves each name as:

```text
../CURE_data/<dataset>.json
```

Do not pass absolute paths to `--dataset` unless you also modify the loader.

The default evaluation script covers these chunk families:

| Family | Chunks |
| --- | --- |
| CodeContests | `CodeContests_chunk_0` to `CodeContests_chunk_4` |
| CodeForces | `CodeForces_chunk_0` to `CodeForces_chunk_9` |
| LiveBench | `LiveBench_chunk_0` to `LiveBench_chunk_2` |
| LiveCodeBench | `LiveCodeBench_chunk_0` to `LiveCodeBench_chunk_10` |

## Evaluation

Run the default open-source evaluation entrypoint:

```bash
CONDA_ENV_NAME=cosplay bash evaluation/eval.sh
```

Evaluate a base model by local path or model ID:

```bash
MODEL=/path/to/model bash evaluation/eval.sh
MODEL=Qwen/Qwen2.5-7B-Instruct bash evaluation/eval.sh
```

Run repeated experiments with separate output modes:

```bash
REPEAT_IDS=1,2,3 bash evaluation/eval.sh
```

Override GPU placement:

```bash
CUDA_VISIBLE_DEVICES=0,1 GPU_GROUPS='[[0],[1]]' bash evaluation/eval.sh
```

Common runtime overrides:

| Variable | Meaning |
| --- | --- |
| `PYTHON_BIN` | Python executable, for example `python3` |
| `CONDA_ENV_NAME` | Optional conda environment to activate |
| `MODEL` | Model name or local checkpoint path |
| `CUDA_VISIBLE_DEVICES` | GPUs visible to the process |
| `GPU_GROUPS` | vLLM engine GPU grouping |
| `REPEAT_IDS` | Comma-separated repeated run suffixes |
| `LOG_ROOT` | Output directory for evaluation logs |

Logs are written under `CURE_logs/final_eval_logs/final/Cosplay_7b/` by default.

## Repository Layout

```text
CosPlay/
  CURE_data/      Benchmark JSON files used by the evaluator
  evaluation/     Generation, execution, metrics, prompts, and eval script
  README.md       Project overview and usage notes
```

Key evaluation files:

| File | Purpose |
| --- | --- |
| `evaluation/eval.sh` | Main evaluation entrypoint |
| `evaluation/main_self_play_v3.py` | Argument parsing and end-to-end evaluation orchestration |
| `evaluation/generator_v3.py` | Candidate code and unit-test generation pipeline |
| `evaluation/self_play_v3.py` | Self-play refinement loop |
| `evaluation/execution.py` | Code execution and test running |
| `evaluation/metrics.py` | Metric computation and logging |

## Citation

If you use CosPlay, please cite the paper below once citation information is available:

```bibtex
TODO
```

## Acknowledgement

TODO: Add acknowledgements for upstream codebases, datasets, model providers, and evaluation frameworks.