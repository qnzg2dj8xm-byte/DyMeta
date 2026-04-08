# DynaMeta: A Prototype Framework for Dynamic Neural-State Inference with Metacognitive Routing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

**DynaMeta** is a prototype-level framework for **dynamic neural-state inference** in multimodal biological time series. The project combines interpretable concept extraction, temporal state decoding, transition anticipation, calibration, and conditional re-evaluation into a closed-loop pipeline.

Rather than claiming a fully learned metacognitive agent, DynaMeta is best viewed as a **metacognitively inspired** system: it monitors uncertainty, identifies when evidence is unstable or out-of-distribution, and routes samples to different intervention strategies. In that sense, the project explores how metacognitive-style control can be embedded into biologically grounded sequence modeling.

---

## Dataset Provenance & Reproducibility Note

The core architecture is designed for a continuous **24-hour multimodal neural activity and natural-behavior dataset** targeting the medial prefrontal cortex (mPFC) in freely moving mice.

- **Data Source:** The original dataset used for development and evaluation is based on the open-access dataset provided by Han et al. (2025). If you use this codebase, please cite:
  > Han, J., Zhou, F., Zhao, Z. et al. 24-hour simultaneous mPFC’s miniature 2-photon imaging, EEG-EMG, and video recording during natural behaviors. *Sci Data* 12, 1226 (2025).

- **Repository Note:** Because the full dataset is too large for GitHub distribution, this repository includes **curated truncated slices** under `data/raw/` for demonstration and reproducibility. The full-pipeline development and evaluation logic are documented in `notebooks/original_dataset_sample.ipynb`, while the cleaned demo workflow is presented in `notebooks/data_segments sample.ipynb`.

---

## Project Objectives

This repository is organized around three complementary goals:

1. **Neuroscience:** study how multimodal signals encode stable states and pre-transition dynamics in the mPFC.
2. **Interpretability:** map predictions to explicit biological concepts such as EEG band power, EMG activity, and behavioral signals.
3. **Closed-Loop Inference:** test whether uncertainty-aware routing can improve robustness by deciding when to accept, smooth, repair, or defer a prediction.

---

## Core Architecture

DynaMeta separates the pipeline into four modules:

### 1. State Identification (System 1 Baseline)
A sequence model continuously decodes the latent physiological state (Wake, NREM, REM, Microarousal) from learned representations of multimodal inputs.

### 2. Dynamic Pre-warning (Transition Head)
A dedicated transition head extracts temporal change signals such as trend, variance, and energy dynamics to estimate whether a state transition is likely to occur.

### 3. Metacognitive Arbitration Layer
A lightweight routing layer evaluates multiple uncertainty signals and assigns each sample to one mutually exclusive intervention strategy:

- **Direct Report:** return the current prediction when confidence is high and the evidence is stable.
- **Keep Watching:** apply temporal smoothing or cooldown when the system appears stable but uncertain.
- **Invoke Repair:** trigger concept-level masking or secondary reasoning when the evidence is noisy but still within known-state regions.
- **Defer:** abstain or pass the sample downstream when the input appears out-of-distribution or too far from learned prototypes.

### 4. Concept Bottleneck & Explanation Interface
Intermediate decisions are linked to interpretable biological concepts so that routing decisions and corrections can be inspected rather than treated as opaque outputs.

---

## Current Scope & Limitations

DynaMeta is currently a **prototype** rather than a production system. Its main contribution is to demonstrate that dynamic neural-state decoding can be augmented with interpretable concept extraction, uncertainty calibration, and structured routing.

Planned future directions include:

- **Learned Meta-Policy:** replacing some rule-based routing decisions with a trainable policy network.
- **Explicit Abstention:** making the defer action more explicit and cost-aware.
- **Structured Diagnostic Output:** generating concise decision summaries that explain why a sample was accepted, smoothed, repaired, or deferred.
- **Cost-Aware Routing:** balancing predictive confidence against the computational cost of additional reasoning.

---

## Implementation Notes

The codebase includes several design choices intended to support stable experimentation on long biological recordings:

- **Memory-Efficient Windowing:** efficient sequence slicing for long recordings.
- **Leakage Prevention:** consistent preprocessing and scaler handling across train/validation/test splits.
- **Loss Balancing:** multi-task losses designed to reduce the impact of severe class imbalance.
- **Debug-Friendly Routing:** print-based and mask-based inspection of routing outcomes during development.

---

## Project Structure

```text
DynaMeta/
├── README.md                             
├── requirements.txt                      
├── checkpoints/
│   └── dynameta_best.pth                 # Serialized model weights
├── configs/
│   └── default_config.yaml               # Central configuration (hyperparams, labels)
├── data/
│   ├── processed/                        # Serialized concept bottleneck tensors (.npy)
│   └── raw/                              # Truncated multimodal recordings (Matlab/CSV)
├── notebooks/
│   ├── data_segments sample.ipynb        # Interactive demonstration on data slices
│   └── original_dataset_sample.ipynb     # Original prototype evaluated on full 24h data
├── scripts/                              # Execution Pipelines
│   ├── evaluate.py                       # Step 3: Statistical validation on test slices
│   ├── process_data.py                   # Step 1: Concept feature extraction
│   ├── run_pipeline.py                   # Step 4: End-to-end metacognitive streaming demo
│   └── train.py                          # Step 2: Model training and scaler fitting
└── src/                                  # Core Framework Modules
    ├── data/
    │   ├── dataset.py                    # PyTorch Dataset and O(1) windowing definitions
    │   ├── features.py                   # PSD and behavioral feature extraction
    │   └── loader.py                     # Data parser and global label registry
    ├── engine/
    │   ├── evaluator.py                  # Inference and metrics calculation engine
    │   └── trainer.py                    # Multi-task training loop
    ├── models/
    │   ├── losses.py                     # TemporalBehaviorContrastiveLoss, BinaryFocalLoss
    │   ├── metacognition.py              # Metacognitive Triage Layer
    │   └── networks.py                   # DualMechanismNet & Prototype Registry
    └── utils/
        └── common.py                     # Deterministic seed settings, logging utilities
```

---

## Quick Start

1. **Environment Setup**
   ```bash
   git clone https://github.com/qnzg2dj8xm-byte/DyMeta.git
   cd DyMeta
   pip install -r requirements.txt
   ```

2. **Run the pipeline**
   ```bash
   python -m scripts.process_data
   python -m scripts.train
   python -m scripts.evaluate
   python -m scripts.run_pipeline
   ```

   The repository includes truncated data slices for demonstration. For the full-data development workflow, please refer to `notebooks/original_dataset_sample.ipynb`.

---

## Academic Exchange & Contact

This project is maintained as an independent research endeavor. I welcome academic discussion, methodological feedback, and collaboration, especially on:

- dynamic neural-state inference,
- uncertainty-aware routing,
- concept bottleneck modeling,
- and metacognitively inspired closed-loop control.

**Issues & Pull Requests:** open for academic suggestions and code improvements.  
**License:** MIT License