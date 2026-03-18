# Open-Source Reproduction and Explainability Analysis of CRAG

[![arXiv](https://img.shields.io/badge/arXiv-2603.16169-b31b1b.svg)](https://arxiv.org/abs/2603.16169)

This repository contains the code for our paper:

**"Open-Source Reproduction and Explainability Analysis of Corrective Retrieval Augmented Generation"**
Surya Vardhan Yalavarthi, University of Cincinnati

## Overview

We reproduce [CRAG (Yan et al., 2024)](https://arxiv.org/abs/2401.15884) using fully open-source components:

| Component | Original | Ours |
|-----------|----------|------|
| Generator | LLaMA-2-7B (fine-tuned) | Phi-3-mini-4k-instruct |
| Web Search | Google Search API (paid) | Wikipedia API (free) |
| Keyword Extraction | GPT-3.5 Turbo | Rule-based |
| Retrieval Evaluator | T5-large (fine-tuned) | Same checkpoint |

## Key Results

| Method | PopQA | ARC-Challenge |
|--------|-------|---------------|
| Vanilla RAG | 51.4% | 84.8% |
| CRAG (ours) | 54.4% | 85.2% |
| CRAG (original) | 54.9% | 53.7% |

## Novel Contributions

1. **Open-source reproduction** — All proprietary components replaced with free alternatives
2. **Wikipedia retrieval pipeline** — 82.3% hit rate on PopQA AMBIGUOUS questions, 99% on ARC-Challenge
3. **SHAP explainability** — First token-level analysis of CRAG's T5 retrieval evaluator, revealing it functions as a named entity alignment detector

## Repository Structure

```
├── scripts/
│   ├── wikipedia_ambiguous_v2.py    # Wikipedia retrieval pipeline
│   ├── validate_wiki.py             # Wikipedia result validation
│   ├── build_ground_truth.py        # PopQA GT construction
│   ├── shap_compute.py              # SHAP computation on T5 evaluator
│   ├── shap_9samples.py             # 9-sample SHAP analysis
│   ├── error_analysis.py            # Question type error analysis
│   └── retrieve_arc_wikipedia.py    # ARC-Challenge Wikipedia retrieval
├── figures/
│   ├── shap_9_summary.png           # SHAP token attribution figure
│   ├── shap_token_heatmap.png       # SHAP heatmap figure
│   ├── error_analysis_plot.png      # Error analysis bar chart
│   └── error_analysis_action_heatmap.png  # Error analysis heatmap
├── paper/
│   └── main.tex                     # LaTeX source
└── README.md
```

## Setup

```bash
git clone https://github.com/suryayalavarthi/crag-reproduction
cd crag-reproduction
pip install transformers torch shap matplotlib numpy sentencepiece wikipedia-api wikipedia
```

## Reproducing Results

### 1. Download T5 Evaluator Checkpoint
Download from the [original CRAG repository](https://github.com/HuskyInSalt/CRAG) and place in `models/finetuned_t5_evaluator/`.

### 2. Run CRAG Pipeline
Run the Kaggle notebook `crag-mistral-baseline` (see `scripts/crag_kaggle_notebook.py`) with the Contriever retrieval results from the original CRAG repo.

### 3. Wikipedia Search for AMBIGUOUS Questions
```bash
python scripts/wikipedia_ambiguous_v2.py
python scripts/validate_wiki.py
```

### 4. SHAP Analysis
```bash
python scripts/shap_9samples.py
```

### 5. Error Analysis
```bash
python scripts/error_analysis.py
```

## Citation

If you use this work, please cite:
```bibtex
@article{yalavarthi2026crag,
  title={Open-Source Reproduction and Explainability Analysis of Corrective Retrieval Augmented Generation},
  author={Yalavarthi, Surya Vardhan},
  journal={arXiv preprint arXiv:2603.16169},
  year={2026}
}
```

And the original CRAG paper:
```bibtex
@article{yan2024corrective,
  title={Corrective Retrieval Augmented Generation},
  author={Yan, Shi-Qi and Gu, Jia-Chen and Zhu, Yun and Ling, Zhen-Hua},
  journal={arXiv preprint arXiv:2401.15884},
  year={2024}
}
```

## Acknowledgments

This work builds on [CRAG](https://github.com/HuskyInSalt/CRAG) by Yan et al. The T5 retrieval evaluator checkpoint is from the original authors.
