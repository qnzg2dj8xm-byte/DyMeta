# DynaMeta: A Prototype-Level Metacognitive Framework for Multimodal Sleep Dynamics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

**DynaMeta** is an experimental, prototype-level deep learning framework exploring the application of metacognitive mechanisms to continuous biological time series. 

This project bridges computational modeling with neuroscience by addressing extreme class imbalance and state transition detection in 24-hour medial prefrontal cortex (mPFC) recordings. It introduces a **Dual-Mechanism Predictor (System 1)** combined with a **Heuristic-Driven Intervention Layer (System 2)**. While currently a prototype relying on rule-based metacognitive triggers rather than a fully learned policy, DynaMeta demonstrates a proof-of-concept for dynamically triaging continuous physiological streams.

---

## Dataset Provenance & Note

The core architecture is designed for and validated on a continuous **24-hour multimodal neural activity and natural behavior dataset** targeting the mPFC in freely moving mice. 

* **Data Source:** The original dataset utilized in this research is sourced from the open-access dataset provided by Han et al. (2025). If you use this codebase, please cite:
  > *Han, J., Zhou, F., Zhao, Z. et al. 24-hour simultaneous mPFC’s miniature 2-photon imaging, EEG-EMG, and video recording during natural behaviors. Sci Data 12, 1226 (2025).*
* **Repository Note:** To facilitate code demonstration, data provided under `data/raw/` consists of **truncated slices**. The evaluation on the complete 24-hour dataset is documented in `notebooks/original_dataset_sample.ipynb`.

---

## Project Objectives

This repository evaluates the model through two distinct academic lenses:
1. **Neuroscience:** Investigating how multimodal signals encode stable states and pre-transition dynamics in the mPFC.
2. **Artificial Intelligence:** Prototyping a "Metacognitive Triage" framework capable of self-assessing confidence, applying temporal smoothing as a proxy for delayed judgment, and conditionally invoking secondary reasoning.

---

## Core Architecture: The Four Modules

DynaMeta structures the pipeline into four decoupled modules:

### 1. State Identification (System 1 Baseline)
A multi-step predictor (via LSTM) that continuously decodes the underlying physiological state (Wake, NREM, REM, Microarousal) from latent representations.

### 2. Dynamic Pre-warning (Transition Head)
A specialized transition head extracts trend, variance, and energy dynamics ($\Delta$-concept) to predict impending phase transitions, operating independently from the primary classification confidence.

### 3. Heuristic-Driven Metacognitive Triage Layer
A rule-based policy layer evaluating cognitive dissonance to execute a proxy triage strategy:
* **Direct Report:** Output directly when Shannon entropy is low and the state aligns with learned prototypes.
* **Keep Watching (Proxy via Temporal Smoothing):** Applies temporal inertia filters (`temporal_filter`, `apply_cooldown`) to stabilize predictions when evidence fluctuates.
* **Invoke Sys2 (Threshold-Triggered):** Triggers a concept-anchor masking mechanism (secondary reasoning) when entropy or prototype distance exceeds predefined thresholds.

### 4. Concept Bottleneck (Interpretability)
Decisions are mapped to interpretable biological concepts (e.g., EEG Delta/Theta power, EMG variance, motor activities) to maintain transparency during Sys2 interventions.

---

## Current Limitations & Future Roadmap

DynaMeta is currently a **heuristic-triggered dual system prototype**. It successfully demonstrates state tracking, transition warning, and explainable correction. However, to evolve into a fully autonomous metacognitive agent, the following developments are planned for future iterations:

* **Unified Learned Meta-Controller:** Transitioning from threshold-based triggers to a fully learned meta-policy network that dynamically dictates the triage action.
* **Explicit Abstention Mechanism:** Replacing the current temporal smoothing proxy with a true "Wait/Delay" action, explicitly modeling the process of gathering more evidence.
* **Cost-Aware Reasoning:** Introducing explicit penalty modeling for invoking System 2, training the network to balance computational cost against predictive certainty.
* **Metacognitive Decision Logs:** Outputting unified logs explaining *why* the meta-controller chose to output, wait, or invoke deep reasoning at any given timestep.

---

## Implementation Details

To ensure computational efficiency and prevent common methodological errors, the codebase incorporates:
* **Memory-Efficient Windowing:** Uses zero-copy memory views (`numpy.lib.stride_tricks`) for $O(1)$ sequence generation, preventing memory exhaustion on 24h datasets.
* **Leakage Prevention:** Employs a global label registry and strictly serialized distribution scalers (`RobustScaler`) to avoid train-test leakage.
* **Loss Balancing:** Utilizes a weighted Temporal-Behavior Contrastive (TBC) Loss and Binary Focal Loss to mitigate extreme biological class imbalance.

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

1. **Environment Setup:**
   ```bash
   git clone [https://github.com/qnzg2dj8xm-byte/DyMeta.git](https://github.com/qnzg2dj8xm-byte/DyMeta.git)
   cd DyMeta
   pip install -r requirements.txt
   ```

2. **Execute the Pipeline:**
   Using the provided data slices in `data/raw/`, you can verify the pipeline sequentially:
   ```bash
   python -m scripts.process_data   # Extract interpretable concepts
   python -m scripts.train          # Train the dual-mechanism model
   python -m scripts.evaluate       # Output evaluation metrics
   python -m scripts.run_pipeline   # Launch the triage demonstration
   ```
   *Note: For interactive exploration and full dataset evaluation logic, please refer to the notebooks in the `notebooks/` directory.*

---

## Academic Exchange & Contact

This project is maintained as an independent research endeavor. I highly welcome academic discussions, constructive feedback, and collaborations—especially regarding the evolution of this prototype into a fully learned metacognitive policy.

* **Issues & Pull Requests:** Open for academic suggestions and code improvements.
* **License:** MIT License