# CoSPlay: Cooperative Self-Play at Test-Time with Self-Generated Code and Unit Test

<p align="center">
  <a href="#-citation"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b" alt="Paper"></a>
  <a href="https://github.com/sanae-ai/CosPlay"><img src="https://img.shields.io/badge/Code-GitHub-000000?logo=github" alt="Code"></a>
  <a href="https://huggingface.co/datasets/yomi017/CosPlay"><img src="https://img.shields.io/badge/Hugging%20Face-Data%20%26%20Logs-ffcc00?logo=huggingface" alt="Data and logs"></a>
  <a href="assets/methodology.pdf"><img src="https://img.shields.io/badge/Methodology-PDF-2b6cb0" alt="Methodology PDF"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="MIT license"></a>
</p>

<p align="center">
  🎉 <a href="#-news">News</a> •
  🔗 <a href="#-links">Links</a> •
  🧠 <a href="#-method-overview">Method Overview</a> •
  📦 <a href="#-data-and-logs">Data & Logs</a> •
  ✨ <a href="#-getting-started">Getting Started</a> •
  🛠️ <a href="#-evaluation">Evaluation</a> •
  🗂️ <a href="#-repository-layout">Repository Layout</a> •
  📌 <a href="#-citation">Citation</a> •
  🌻 <a href="#-acknowledgement">Acknowledgement</a> •
  📬 <a href="#-contact">Contact</a>
</p>

CoSPlay is a **test-time scaling (TTS)** method for code generation. It improves a chosen base model at inference time by coupling self-generated code and unit tests through cooperative self-play, without releasing or requiring a separately trained model.

The current release provides the evaluation code, benchmark chunk loader, and scripts for reproducing the CoSPlay pipeline. Generated data and evaluation logs are hosted on Hugging Face.

## 🎉 News

- **2026-05**: Code, evaluation scripts, generated data, and logs are being prepared for public release.
- **TODO**: Add the paper link after the paper page is available.

## 🔗 Links

| Resource | URL |
| --- | --- |
| 📄 Paper | Coming soon |
| 💻 Code | [github.com/sanae-ai/CosPlay](https://github.com/sanae-ai/CosPlay) |
| 🤗 Data & Logs | [huggingface.co/datasets/yomi017/CosPlay](https://huggingface.co/datasets/yomi017/CosPlay) |
| 🖼️ Methodology Figure | [assets/methodology.pdf](assets/methodology.pdf) |

## 🧠 Method Overview

CoSPlay treats code generation as a test-time interaction between a code pool and a unit-test pool:

1. **Explore** candidate code ideas and failure-oriented unit-test ideas.
2. **Generate** code candidates and adversarial/random unit tests.
3. **Self-play** between code and tests to clean weak code, replace weak tests, and fix useful failures.
4. **Cluster** output behavior using random valid test inputs.
5. **Select** the final answer with BoN-style code selection.

<p align="center">
  <img src="assets/methodology.png" alt="CoSPlay methodology" width="95%">
</p>

<p align="center">
  <a href="assets/methodology.pdf">📎 Open the high-resolution PDF</a>
</p>

The default configuration in `evaluation/eval.sh` is the final CoSPlay setting used by this repository:

| Setting | Default |
| --- | --- |
| Code candidates | `K_CODE=16` |
| Unit-test candidates | `K_CASE=16` |
| Self-play rounds | `SELF_PLAY_ROUND=5` |
| Evaluation mode | `bon` |
| Default base model | `Qwen/Qwen2.5-7B-Instruct` |

## 📦 Data and Logs

This repository does **not** release trained models. CoSPlay is a TTS method that can be applied to a selected base model at evaluation time.

Generated data and evaluation logs are available here:

```text
https://huggingface.co/datasets/yomi017/CosPlay
```

The official benchmark numbers will be reported in the paper.

## ✨ Getting Started

Clone the repository:

```bash
git clone https://github.com/sanae-ai/CosPlay.git
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

## 🧩 Benchmark Chunks

CoSPlay expects evaluation files under `CURE_data/`. The evaluation script passes dataset names as file stems, and `main_self_play_v3.py` resolves each name as:

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

## 🛠️ Evaluation

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

## 🗂️ Repository Layout

```text
CosPlay/
  assets/         Methodology figure source and README preview image
  CURE_data/      Benchmark JSON files used by the evaluator
  evaluation/     Generation, execution, metrics, prompts, and eval script
  LICENSE         MIT license
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

## 📌 Citation

If you use CoSPlay, please cite the paper below once citation information is available:

```bibtex
TODO
```

## 🌻 Acknowledgement

TODO: Add acknowledgements for upstream codebases, datasets, model providers, and evaluation frameworks.

## 📜 License

This project is released under the [MIT License](LICENSE).

## 📬 Contact

TODO.
