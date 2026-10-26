> **Update (v2, Oct 2026):** see [`v2/`](v2/) for SleepQualityNet v2: Kaggle-ready notebook, channel-verified data pipeline, rule-derived N3 factor profile, and pre-specified RQ1–RQ7 analyses. The v1 results below were affected by an ISRUC channel-selection bug (83/100 subjects read from `E1-M2`); details in [`v2/docs/02_v1_audit_and_supervisor_comments.md`](v2/docs/02_v1_audit_and_supervisor_comments.md).

# SleepQualityNet

### Dual-Path CNN-Attention Framework for N3 Sleep Detection and EEG-Derived Delta-Intensity Estimation

SleepQualityNet is a **multi-task deep learning framework** for automated analysis of 30-second single-channel EEG epochs.

The framework jointly performs:

- 🧠 **N3 (Deep Sleep) Detection**
- 📈 **EEG-Derived Delta-Intensity Estimation**
- 🔍 **Explainable AI (XAI)**

> **Note:** The regression output is an EEG-derived delta-intensity index and is **not a clinically validated sleep-quality score**.

---

## 🧠 Framework

SleepQualityNet combines:

- Raw EEG representation learning
- CNN-based feature extraction
- Temporal attention
- Physiologically motivated EEG features
- Multi-task learning


## 📊 Dataset

The final verified cohort contains **95 participants**:

| Dataset | Participants |
|---|---:|
| ISRUC-Sleep-I | 17 |
| Sleep-EDF Expanded | 78 |
| **Total** | **95** |

Only verified EEG channels were used:

- **C4-A1** for ISRUC-Sleep
- **Fpz-Cz** for Sleep-EDF

Final dataset: **108,705 epochs**

---

## ⚙️ Training

- 30-second EEG epochs
- Sampling rate: **200 Hz**
- 5-fold **StratifiedGroupKFold**
- Subject-independent evaluation
- AdamW optimizer
- 60 epochs
- Batch size: 48
- Learning rate: 1e-4
- Global seed: 42

---

## 🔍 Explainability

The framework includes:

- SHAP feature attribution
- Gradient saliency
- Temporal attention visualization
- Delta-band perturbation analysis

---

## 📏 Evaluation

### Classification
Accuracy, Precision, Recall, F1-score, ROC-AUC

### Regression
RMSE, R², Pearson Correlation

### Stability
Prediction SD, Outlier Rate, Extreme Prediction Rate, IQR Ratio

---

## 💻 Environment

```text
Python       3.12.10
PyTorch      2.11.0
CUDA         12.8
NumPy        2.4.4
SciPy        1.17.1
Scikit-Learn 1.9.0
MNE          1.12.1
GPU          NVIDIA RTX 3050 Laptop GPU
```

---

## ⚠️ Limitations

The delta-intensity target is **algorithmically derived**, not clinically annotated.  
Slow-wave density is heuristic, and attention/SHAP outputs should not be interpreted as causal explanations.

---

## 👥 Authors

**MD Shamsul Alam Pranto**, **Sayed Mamun**, **Fahad Ibne Firoz**, **MD. Kamrul Islam**

**Research Project:** SleepQualityNet for Deep Sleep Detection and EEG-Derived Delta-Intensity Estimation using Verified Single-Channel EEG.
