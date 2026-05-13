# CoSPlay: Cooperative Self-Play at Test-Time with Self-Generated Code and Unit Test

<p align="center">
  <a href="#-citation"><img src="https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b" alt="Paper"></a>
  <a href="https://github.com/sanae-ai/CosPlay"><img src="https://img.shields.io/badge/Code-GitHub-000000?logo=github" alt="Code"></a>
  <a href="https://huggingface.co/datasets/yomi017/CosPlay"><img src="https://img.shields.io/badge/Hugging%20Face-Data%20%26%20Logs-ffcc00?logo=huggingface" alt="Data and logs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="MIT license"></a>
</p>

<p align="center">
  📌 <a href="#-introduction">Introduction</a> •
  🎯 <a href="#-motivation">Motivation</a> •
  🧠 <a href="#-methods">Methods</a> •
  ✨ <a href="#-contributions">Contributions</a> •
  🚀 <a href="#-quick-start">Quick Start</a> •
  📦 <a href="#-data-and-logs">Data & Logs</a> •
  🛠️ <a href="#-comprehensive-evaluation">Evaluation</a> •
  📊 <a href="#-interesting-results">Results</a> •
  📚 <a href="#-citation">Citation</a>
</p>

<p align="center">
  <img src="assets/radar.png" alt="CoSPlay code and unit-test capability comparison" width="78%">
</p>

<p align="center">
  <sub><em><strong>Capability Comparison.</strong> Code and UT capability comparison against RLVR baselines.</em></sub>
</p>

CoSPlay improves code and unit-test capabilities without GT data or training, while competing RLVR methods require costly weight updating or massive GT labels.

<p align="center">
  <img src="assets/tts_cost_vs_pass1_combined.png" alt="TTS cost versus pass@1 comparison" width="78%">
</p>

<p align="center">
  <sub><em><strong>Efficiency.</strong> Token cost versus Pass@1 on Qwen2.5-Instruct models.</em></sub>
</p>

Darker markers indicate scaled variants with larger budgets; CoSPlay reaches a stronger cost-accuracy trade-off than GT-free TTS baselines.

## 📌 Introduction

CoSPlay is a **GT-free, training-free test-time scaling (TTS)** framework for code generation. It improves a chosen base model at inference time by coupling self-generated code and unit tests through cooperative self-play, without requiring ground-truth unit tests or releasing a separately trained model.

The current release provides evaluation code, benchmark loaders, paper figures, and scripts for reproducing the CoSPlay pipeline. Generated data and evaluation logs are hosted on Hugging Face.

## 🎯 Motivation

CoSPlay targets the middle ground: high-accuracy code generation with **no GT tests** and **no additional training**.

<p align="center">
  <img src="assets/motivation.png" alt="CoSPlay motivation: GT-free and training-free test-time scaling" width="88%">
</p>

<p align="center">
  <sub><em><strong>Motivation.</strong> GT-free and training-free code generation setting.</em></sub>
</p>

Existing RLVR methods can rely on costly ground-truth unit tests and weight updates, while GT-free TTS methods often spend more compute on sampling without reliably filtering noisy self-generated tests.

## 🧠 Methods

CoSPlay treats code generation as a test-time interaction between a code pool and a unit-test pool. The pipeline has three stages: idea-level exploration and attack, execution-matrix-driven self-play, and output-consensus cluster selection.

1. **Explore** candidate code ideas and failure-oriented unit-test ideas.
2. **Generate** code candidates and adversarial/random unit tests.
3. **Self-play** between code and tests to clean weak code, replace weak tests, and fix useful failures.
4. **Cluster** output behavior using random valid test inputs.
5. **Select** the final answer with BoN-style code selection.

<p align="center">
  <img src="assets/methodology.png" alt="CoSPlay methodology" width="88%">
</p>

<p align="center">
  <sub><em><strong>Full CoSPlay Pipeline.</strong> Exp&Atk idea generation, execution-matrix self-play, and output-consensus clustering.</em></sub>
</p>

CoSPlay first generates solution-oriented code ideas and failure-oriented UT ideas to bootstrap reliable and discriminative pools. It then co-evolves code and UTs through execution-matrix pass-count signals, and resolves BoN ties with output-consensus clustering.

## ✨ Contributions

- **GT-free and training-free self-play:** CoSPlay builds a cooperative loop between self-generated code and self-generated unit tests, improving inference-time performance without ground-truth unit tests or model weight updates.
- **Execution-matrix signals:** Code and unit-test pass counts provide internal reliability signals, allowing the method to clean weak code, refresh noisy tests, repair useful failures, and co-evolve both pools.
- **Consensus-based final selection:** Output-consensus clustering resolves BoN ties using random valid inputs, improving robustness, scaling behavior, and transfer across base models.

## 🚀 Quick Start

Clone the repository:

```bash
git clone https://github.com/sanae-ai/CosPlay.git
cd CosPlay
```

Create an environment and install dependencies:

```bash
conda create --name CosPlay python=3.10 -y
conda activate CosPlay

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

pip install xformers==0.0.33.post1
pip install flashinfer-python==0.5.3

pip install --no-cache-dir \
  https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0+cu128-cp38-abi3-manylinux_2_35_x86_64.whl \
  --extra-index-url https://download.pytorch.org/whl/cu128

MAX_JOBS=4 pip install flash-attn==2.8.3 --no-build-isolation
```

## 📦 Data and Logs

Generated data and evaluation logs are available here:

```text
https://huggingface.co/datasets/yomi017/CosPlay
```

## 🧩 Datasets

The dataset is also in: 

```text
https://huggingface.co/datasets/yomi017/CosPlay/Datasets
```

## 🛠️ Comprehensive Evaluation

Run the default open-source evaluation entrypoint:

```bash
CONDA_ENV_NAME=cosplay bash evaluation/eval.sh
```

## 📊 Interesting Results

<p align="center">
  <img src="assets/generalization_of_cosplay_on_various_models_cropped.png" alt="Generalization of CoSPlay on various base models" width="88%">
</p>

<p align="center">
  <sub><em><strong>Generalization.</strong> BoN gains across base and RL models.</em></sub>
</p>

CoSPlay generalizes across different model families and scales, showing that the self-play mechanism is not tied to a single checkpoint.

---

<p align="center">
  <img src="assets/Cosplay_scaling_cropped.png" alt="CoSPlay scaling with BoN candidate budget" width="70%">
</p>

<p align="center">
  <sub><em><strong>Scaling.</strong> BoN accuracy versus candidate-pool budget.</em></sub>
</p>

CoSPlay continues to scale with larger candidate-pool budgets, improving the BoN accuracy ceiling beyond strong baselines.

---

<p align="center">
  <img src="assets/bon_diversity_tradeoff_curved_frontier_stronger_bon.png" alt="Accuracy-diversity tradeoff with CoSPlay" width="88%">
</p>

<p align="center">
  <sub><em><strong>UT Diversity Trade-off.</strong> UT accuracy-rank frontier with BoN and Signal encoded by marker size.</em></sub>
</p>

CoSPlay improves the accuracy-diversity frontier, indicating that better self-generated tests can help select stronger and more diverse code candidates.

## 📚 Citation

If you use CoSPlay, please cite the paper:

```bibtex
@article{hu2026cosplay,
  title={CoSPlay: Cooperative Self-Play at Test-Time with Self-Generated Code and Unit Test},
  author={Hu, Zhangyi and Liu, Chenhui and Huang, Tian and Li, Jindong and Yang, Yang and Wu, Jiemin and Zhong, Zining and Yang, Menglin and Yue, Yutao},
  journal={arXiv preprint},
  year={2026}
}
```

## 🌻 Acknowledgement

This work was supported in part by Guangzhou-HKUST(GZ) Joint Funding Program (Grant No. 2023A03J0008), Education Bureau of Guangzhou Municipality.

## 📜 License

This project is released under the [MIT License](LICENSE).

## 📬 Contact

For questions, please contact zhangyi_hu@whu.edu.cn, cliu9168@gmail.com, or yfield017@gmail.com.
