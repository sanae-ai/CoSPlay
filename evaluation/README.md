# Evaluation

This directory contains the open-source evaluation pipeline for **CoSPlay:
Cooperative Self-Play at Test-Time with Self-Generated Code and Unit Test**.
CoSPlay is a **GT-free** and **training-free** test-time scaling method for code
generation. It does not require ground-truth unit tests during inference and does
not update model weights. Instead, it uses the model itself to generate code,
generate unit tests, execute code-test pairs, refine both pools, and select the
final answer from execution evidence.

The default script, [`eval.sh`](eval.sh), runs the final CoSPlay setting used by
this release on the benchmark files under `../CURE_data`.

## Benchmarks

The release covers `CodeContests`, `CodeForces`, `LiveBench`, and
`LiveCodeBench`. Dataset files are stored under `../CURE_data`, and
[`main.py`](main.py) resolves `--dataset` as a file stem:

```text
--dataset CodeContests_chunk_0 -> ../CURE_data/CodeContests_chunk_0.json
```

Use file stems rather than absolute paths unless you also modify the loader.

## Metrics

Common logged metrics:

- `Code Acc.`: official-test pass rate of generated code candidates.
- `UT Acc.`: correctness rate of generated unit tests.
- `Signal` / `Sig.`: generated-test selection accuracy.
- `BoN`: best-of-N selection using generated tests.

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

## Running One Dataset Manually

For debugging, you can call the Python entrypoint directly from this directory:

```bash
python -u main.py \
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

## Resume Support

`main.py` supports resuming from saved outputs:

| Argument | Purpose |
| --- | --- |
| `--resume_round00` | Resume from the initial generated round-0 file. |
| `--resume_round` | Resume from a later self-play round snapshot. |
| `--start_round` | Round index already completed. |

