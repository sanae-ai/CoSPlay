# Evaluation

This directory contains the open-source evaluation pipeline for **CoSPlay:
Cooperative Self-Play at Test-Time with Self-Generated Code and Unit Test**.
CoSPlay is a **GT-free** and **training-free** test-time scaling method for code
generation. It does not require ground-truth unit tests during inference and does
not update model weights. Instead, it uses the model itself to generate code,
generate unit tests, execute code-test pairs, refine both pools, and select the
final answer from execution evidence.

The default script, [`eval.sh`](eval.sh), runs the final CoSPlay setting used by
this release on the benchmark chunks under `../CURE_data`.

## What CoSPlay Evaluates

Given one competitive-programming problem, CoSPlay maintains two generated pools:

| Pool | Meaning |
| --- | --- |
| Code pool | Candidate Python programs sampled from solution plans. |
| Unit-test pool | Self-generated input-output tests used for repair and selection. |

The pipeline has three stages.

### Stage 1: Exploration-Attack Idea Generation

CoSPlay first asks the model to explore solution ideas rather than immediately
sampling code. It generates high-level algorithmic hints, expands small hint
subsets into detailed solution plans, and then derives **failure-oriented unit
test ideas** from those plans. These attack ideas describe edge cases,
implementation pitfalls, or hidden failure modes that are likely to distinguish
correct and buggy solutions.

This stage is important because directly sampled unit tests are often generic or
weakly discriminative. By grounding tests in concrete solution hypotheses,
CoSPlay obtains a stronger initial code-test pool for subsequent self-play.

### Stage 2: Execution-Matrix-Driven Self-Play

CoSPlay executes every generated code candidate against every generated unit
test and builds a binary execution matrix:

```text
M[i, j] = 1 if code_i passes unit_test_j, else 0
```

Row statistics estimate code quality, while column statistics estimate unit-test
reliability and discriminative power. Each self-play round uses these statistics
to update both pools:

| Step | Purpose |
| --- | --- |
| Code cleaning | Resample code candidates that fail all current tests. |
| Coupling break | Regenerate low-support non-trivial tests that may be spuriously coupled with a few wrong codes. |
| Code fixing | Repair code candidates using a high-support but still non-trivial test. |
| UT replacement | Replace invalid or zero-pass tests after code repair, and separately refresh all-pass tests when they no longer discriminate among candidates. |

After each update, the code-test execution matrix is refreshed, so later
decisions use current execution evidence rather than stale signals.

### Stage 3: Output-Consensus Cluster Selection

After self-play, CoSPlay performs BoN-style scoring with the evolved unit-test
pool. If multiple code candidates tie, it generates random valid inputs and
clusters tied candidates by their output behavior. Correct code is expected to
form a larger and more internally consistent output cluster, while wrong code
tends to fail in diverse ways. The final program is selected from the strongest
execution-consensus cluster.

## Benchmarks

The release evaluates on four coding benchmarks:

| Benchmark | Description |
| --- | --- |
| `LiveBench` | Recent coding problems from LiveBench, used to test instruction-following and programming ability on less-contaminated tasks. |
| `LiveCodeBench` | Recent competitive-programming style problems with hidden tests and temporal separation from many training corpora. |
| `CodeContests` | Algorithmic programming problems from the CodeContests benchmark, emphasizing classical contest reasoning. |
| `CodeForces` | Codeforces-style competitive-programming problems, typically the hardest split in our evaluation. |

Datasets are stored as JSON chunks under:

```text
../CURE_data/<dataset_name>.json
```

The loader in [`main_self_play_v3.py`](main_self_play_v3.py) resolves
`--dataset CodeContests_chunk_0` as:

```text
../CURE_data/CodeContests_chunk_0.json
```

Do not pass an absolute path to `--dataset` unless you also modify the loader.

The default script evaluates these chunk families:

| Family | Chunks |
| --- | --- |
| `CodeContests` | `CodeContests_chunk_0` to `CodeContests_chunk_4` |
| `CodeForces` | `CodeForces_chunk_0` to `CodeForces_chunk_9` |
| `LiveBench` | `LiveBench_chunk_0` to `LiveBench_chunk_2` |
| `LiveCodeBench` | `LiveCodeBench_chunk_0` to `LiveCodeBench_chunk_10` |

## Metrics

The evaluation logs several code and unit-test metrics. The most commonly used
ones are:

| Metric | Meaning |
| --- | --- |
| `Code Acc.` | Fraction of generated code candidates that pass official tests. This is used for analysis, not for test-time supervision. |
| `UT Acc.` | Fraction of generated unit tests whose expected outputs are correct. This is also analysis-only. |
| `Signal` / `Sig.` | Accuracy when generated unit tests are used to select among code candidates; it measures UT discriminative power. |
| `BoN` | Best-of-N accuracy using self-generated code candidates and generated unit tests. |
| `Cluster` | Output-consensus selection among BoN-tied candidates using random valid inputs. |

Official tests are only used for final evaluation and analysis. CoSPlay itself
uses self-generated tests during inference.

## Quick Start

From the repository root:

```bash
CONDA_ENV_NAME=cosplay bash evaluation/eval.sh
```

By default, this evaluates:

```text
MODEL=Qwen/Qwen2.5-7B-Instruct
K_CODE=16
K_CASE=16
SELF_PLAY_ROUND=5
EVAL_MODE=bon
GENERATION_MODE=plansearch
```

You can evaluate a local checkpoint or a different Hugging Face model:

```bash
MODEL=/path/to/model bash evaluation/eval.sh
MODEL=Qwen/Qwen2.5-14B-Instruct bash evaluation/eval.sh
```

Run repeated experiments with separate output modes:

```bash
REPEAT_IDS=1,2,3 bash evaluation/eval.sh
```

Control GPU placement:

```bash
CUDA_VISIBLE_DEVICES=0,1 GPU_GROUPS='[[0],[1]]' bash evaluation/eval.sh
```

## Main Configuration

Most options can be overridden through environment variables in `eval.sh`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Model name or local checkpoint path. |
| `K_CODE` | `16` | Number of candidate programs generated per problem. |
| `K_CASE` | `16` | Number of generated unit tests maintained per problem. |
| `SELF_PLAY_ROUND` | `5` | Number of code-test self-play rounds. |
| `GENERATION_MODE` | `plansearch` | Uses exploration-guided plan search before code generation. |
| `EVAL_MODE` | `bon` | Runs BoN-style selection with generated unit tests. |
| `SCALE_TUPLE_LIST` | `[(2, 2), (4, 4), (8, 8), (16, 16)]` | BoN scales evaluated from the same generated pools. |
| `PASS_AT_K_LIST` | `[1,2,4,8,16]` | Pass@k values to report. |
| `USE_IDEA_ATTACK_UT` | `True` | Enables failure-oriented UT ideas from Stage 1. |
| `USE_SELF_PLAY` | `True` | Enables execution-guided iterative self-play. |
| `SELF_CONSISTENCY_NUM` | `4` | Number of self-consistency samples for UT output validation. |
| `GPU_GROUPS` | `[[0],[1]]` | vLLM worker GPU grouping. |
| `LOG_ROOT` | `../CURE_logs/final_eval_logs/final/Cosplay_7b` | Root directory for run logs. |

## Running One Dataset Manually

For debugging, you can call the Python entrypoint directly from this directory:

```bash
python -u main_self_play_v3.py \
  --use_api False \
  --pretrained_model Qwen/Qwen2.5-7B-Instruct \
  --dataset CodeContests_chunk_0 \
  --k_code 16 \
  --k_case 16 \
  --generation_mode plansearch \
  --eval_mode bon \
  --scale_tuple_list '[(2, 2), (4, 4), (8, 8), (16, 16)]' \
  --pass_at_k_list '[1,2,4,8,16]' \
  --use_idea_attack_ut True \
  --use_self_play True \
  --self_play_round 5 \
  --gpu_groups '[[0],[1]]'
```

## Output Layout

`eval.sh` writes a tee log for each model-dataset pair:

```text
<LOG_ROOT>/<MODE>/eval_<model>_<dataset>.log
```

During evaluation, the Python pipeline also writes intermediate and final result
artifacts under the configured result/log directories. Common artifacts include:

| Artifact | Meaning |
| --- | --- |
| `results_eval_<model>_<dataset>.json` | Main per-problem output after generation/evaluation. |
| `self_play_v2_rounds/` | Round-by-round snapshots of the evolving code and UT pools. |
| `round_*_metrics` / result text logs | Per-round BoN, pass@k, execution, and usage summaries. |

Exact filenames depend on `MODE`, model name sanitization, and whether the run is
resumed.

## File Guide

| File | Purpose |
| --- | --- |
| [`eval.sh`](eval.sh) | Main shell entrypoint for full benchmark evaluation. |
| [`main_self_play_v3.py`](main_self_play_v3.py) | Argument parsing, dataset loading, orchestration, resume logic, and final evaluation. |
| [`generator_v3.py`](generator_v3.py) | Stage-1 idea exploration, code generation, unit-test generation, extraction, and validation helpers. |
| [`self_play_v3.py`](self_play_v3.py) | Execution-matrix-driven self-play loop. |
| [`execution.py`](execution.py) | Sandboxed execution of generated code on generated and official tests. |
| [`metrics.py`](metrics.py) | BoN, pass@k, cluster selection, and evaluation metrics. |
| [`inference.py`](inference.py) | vLLM and API model backends. |
| [`UT_config.py`](UT_config.py) | Prompt templates and helper functions for unit-test generation, refinement, and code fixing. |
| [`usage_tracking.py`](usage_tracking.py) | Token and request accounting across generation stages. |

## Local vLLM and API Modes

The default open-source script uses local vLLM workers:

```text
USE_API=False
```

For API evaluation, set `--use_api True` and configure `api_key`, `base_url`,
and `api_model_name` in [`evaluation_config.py`](evaluation_config.py) or pass
them through your own wrapper. The public config keeps API keys as placeholders.

## Resume Support

`main_self_play_v3.py` supports resuming from saved outputs:

| Argument | Purpose |
| --- | --- |
| `--resume_round00` | Resume from the initial generated round-0 file. |
| `--resume_round` | Resume from a later self-play round snapshot. |
| `--start_round` | Round index already completed. |

This is useful for long-running jobs where generation completed but later
self-play rounds or metrics need to be rerun.

## Practical Notes

- Make sure every dataset listed in `eval.sh` exists under `../CURE_data`.
- Keep `K_CODE` and `K_CASE` aligned with the largest values in
  `SCALE_TUPLE_LIST`; the default `(16, 16)` assumes both pools contain 16
  candidates.
- The script creates per-job Torch/vLLM cache directories to avoid conflicts
  when multiple jobs run on the same machine.
- Generated unit tests are noisy by design; CoSPlay relies on self-consistency,
  execution pass counts, and self-play replacement to filter that noise.
- Official tests should not be used during generation or self-play. They are
  only used by the evaluator to compute final correctness.

## Citation

If you use this evaluation pipeline, please cite the CoSPlay paper once the
official citation is available in the repository root README.
