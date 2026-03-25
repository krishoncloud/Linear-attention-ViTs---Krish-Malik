# Linear-attention-ViTs---Krish-Malik
# GSoC 2026 — ML4SCI

## Specific Task 2h: Linear attention vision transformers for end to end mass regression and classification

**Contributor:** Krish Malik  
**Organization:** ML4SCI (Machine Learning for Science)  
**Project:** [Linear Attention Vision Transformers for End-to-End Mass Regression and Classification](https://ml4sci.org/gsoc/2026/proposal_E2E5.html)

---

## Overview

This repository contains the prerequisite test task solution for the GSoC 2026 E2E project. The task involves building a Vision Transformer with **linear attention** for simultaneously classifying particle collision images (binary classification) and regressing the invariant mass of heavy particles, using multi-channel jet images from the CMS experiment.

The approach implements a complete pipeline: **self-supervised MAE pretraining** on unlabelled CMS data, followed by **supervised finetuning** for joint classification and mass regression.

---

## Architecture

**LinearViT** — a Vision Transformer with Katharopoulos linear attention:

- **Input:** 125×125 images, 8 detector channels (tracker pT, ECAL, HCAL, impact parameters, muon hits)
- **Patch Embedding:** Conv2d with kernel=stride=5 → 625 tokens, projected to 192-dim
- **Attention:** Katharopoulos linear attention with φ(x) = elu(x) + 1, achieving O(N·d²) complexity instead of O(N²·d)
- **Encoder:** 6 transformer blocks, 6 heads, pre-norm layout, MLP ratio 4
- **Heads:** Dual-head — linear classification head + MLP regression head from [CLS] token
- **Parameters:** ~2.87M

**MAE Pretraining:**
- 75% mask ratio (encoder sees 156 of 625 patches)
- Lightweight decoder: 2 layers, 96-dim, 3 heads
- 30 epochs on unlabelled CMS data (~490K samples)

---

## Results

### Final Comparison: Pretrained+Finetuned vs From-Scratch Baseline

| Metric | Pretrained | Baseline | Winner |
|---|---|---|---|
| Classification Accuracy | **85.20%** | 80.60% | Pretrained |
| Best Validation Loss | **22.72** | 24.53 | Pretrained |
| Epochs to Best | **17** | 20 | Pretrained |

### Pretrained + Finetuned Model (Detailed)

| Task | Metric | Value |
|---|---|---|
| Classification | Accuracy | 85.20% |
| Classification | Precision (macro) | 0.85 |
| Classification | Recall (macro) | 0.85 |
| Classification | F1-score (macro) | 0.85 |
| Regression | MAE | 22.32 GeV |
| Regression | R² | 0.6322 |
| Regression | Residual Std | 30.22 GeV |

### Key Findings

- **MAE pretraining provides a clear advantage:** +4.6% absolute accuracy gain over training from scratch, with faster convergence (best at epoch 17 vs 20).
- **Linear attention is viable for HEP:** Katharopoulos attention achieves competitive classification performance while scaling linearly in sequence length — critical for high-resolution detector images.
- **Joint training works:** The model simultaneously handles classification and mass regression through a shared encoder with dual task-specific heads.

---

## Plots

The notebook generates the following visualizations:

- **MAE Pretraining Loss Curve** — smooth convergence over 30 epochs
- **MAE Reconstruction Quality** — Original vs Masked (25% visible) vs Reconstructed jet images
- **Training Dynamics** — Pretrained vs Baseline validation loss and accuracy curves
- **Confusion Matrices** — for both pretrained and baseline models
- **Mass Regression Scatter** — Predicted vs True mass with R² and MAE
- **Residual Distribution** — centered near zero with std ~30 GeV

---

## Dataset

- **Unlabelled:** ~490K 8-channel 125×125 jet images from CMS Open Data (used for MAE pretraining)
- **Labelled:** 10K jet images with binary class labels + invariant mass (80/20 train/val split, stratified)

Data sourced from the [CMS Open Data Portal](https://opendata.cern.ch/) via CERNBox and NERSC.

---

## Repository Structure

```
├── LinearViT_GSoC_2h_v8.ipynb    # Complete notebook (runs end-to-end on Colab)
├── README.md
```

---

## How to Run

1. Open the notebook in **Google Colab** (T4 GPU runtime recommended)
2. Run the session setup cells to mount Google Drive and download datasets
3. Execute cells sequentially — the notebook handles:
   - Dataset download from CERNBox/NERSC
   - MAE pretraining (30 epochs, ~45 min on T4)
   - Finetuning with pretrained weights (20 epochs)
   - Baseline training from scratch (20 epochs)
   - Full evaluation and comparison

---

## References

- [1] C. Zheng, "The Linear Attention Resurrection in Vision Transformer," arXiv:2501.16182, 2025.
- [2] A. El-Nouby et al., "XCiT: Cross-Covariance Image Transformers," arXiv:2106.09681, 2021.
- [3] A. Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention," ICML 2020.
- [4] K. He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022.
- [5] D. Julson et al., "Continual Learning via Ensemble-Based Depth-Wise Masked Autoencoders for Data Quality Monitoring in HEP," arXiv:2603.02369, 2026.

---

## Contact

**Krish Malik** — [krishmalikus@gmail.com](mailto:krishmalikus@gmail.com) | [LinkedIn](https://www.linkedin.com/in/krish-malik-0933822b3/) | [GitHub](https://github.com/krishoncloud)
