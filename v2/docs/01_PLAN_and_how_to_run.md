# SleepQualityNet v2: plan, design and how to run on Kaggle

## Goal
From one EEG channel, (1) detect N3, (2) describe each detected N3 epoch with transparent factor scores and an overall **N3 EEG Rating**, (3) attach a **Signal Reliability Score**, and (4) summarise each night as an **Overnight N3 EEG Profile**. The whole design is evaluated so that every research question has an independent reference and a criterion fixed before results exist.

## Data (all public)
| Cohort | Role | Channel | Notes |
|---|---|---|---|
| ISRUC-Sleep subgroup I (100 participants, clinical) | training / CV | C4-A1 = C4-M1 (fallback C3) | Two scorers (`_1.txt`, `_2.txt`) give the inter-rater benchmark and ambiguity test. |
| Sleep-EDF Expanded, sleep-cassette (78 participants, healthy, ages 25–101) | training / CV | Fpz-Cz | Night 1 for training; **night 2 is evaluation-only** (test–retest). Age and sex come from `SC-subjects.xls`. |
| ISRUC-Sleep subgroup III (10 healthy participants) | external test only | C4-M1 | Never used for training, selection or thresholds. |

## Pipeline (code in `sqn2/`)
1. `data.py` / `download.py`: verified EEG channel, 0.3–35 Hz zero-phase filter on the whole recording, 100 Hz, every epoch kept with its index, v1 QC rules stored as flags, compact `.npz` cache.
2. `features.py`: Welch spectra (19 handcrafted detection features) and a zero-crossing slow-wave detector.
3. `models.py`: v1 CNN-attention backbone at 100 Hz, plus feature fusion and a dilated temporal CNN context (±15 epochs). No LSTM, no regression head.
4. `train.py`: 5-fold participant CV stratified by cohort; inner validation for early stopping, threshold and temperature; MC dropout (10 passes); Mahalanobis OOD distance; ablation variants on identical folds; features-only GBM; leave-one-cohort-out models.
5. `scores.py`: factor scores as montage-referenced percentiles of training-fold expert-N3 epochs; prior weights 0.35/0.30/0.20/0.15 (design parameters); exact additive decomposition; episodes; overnight profile.
6. `analysis.py`, `xai.py`, `report.py`, `figures.py`: RQ1–RQ7 against the criteria, tables, figures, methods-parameter table, GO/NO-GO table.

## Research questions, references and pre-specified criteria
| RQ | Independent reference | Criterion (fixed before running) |
|---|---|---|
| RQ1 detection | expert hypnogram; scorer 2; LOCO; ISRUC-III | pooled F1 ≥ 0.75 and PR-AUC ≥ 0.85; full support needs F1 ≥ 0.70 in each cohort and on ISRUC-III |
| RQ2 factors | homeostatic decline; expert N3 vs N2; scorer-disagreement continuum; age | every factor early > late (p < 0.05) and N3 > N2 |
| RQ3 overall rating | weight perturbation; within-cohort contrasts; split-half; night-to-night | 5th-percentile ρ ≥ 0.90 and early > late within each cohort |
| RQ4 components | identical folds, participant-level paired tests | all effects estimated with CI (any direction is a finding) |
| RQ5 XAI | band-stop perturbation, band-decomposed IG, SHAP surrogate, attention alignment | delta top in ≥ 70 % of participants for perturbation and IG, and SHAP top group delta with fidelity ≥ 0.70 |
| RQ6 reliability | prediction errors; inter-scorer disagreement | AUROC ≥ 0.75 and F1@80 % coverage > F1@100 % |
| RQ7 profiles | expert-hypnogram profile; night 2 | ICC(2,1) ≥ 0.75 for N3 minutes |

## Running on Kaggle (free tier)
1. Create a notebook and upload `SleepQualityNet_v2_kaggle.ipynb` (*File → Import Notebook*).
2. *Settings*: **Accelerator: GPU T4 ×2**, **Internet: ON** (needs a phone-verified account).
3. Set `QUICK_TEST = True` and *Run All*. It takes about 20–30 minutes and checks the downloads and every stage on a small subset.
4. Set `QUICK_TEST = False`, then **Save Version → Save & Run All (Commit)**. Expected time: downloads about 1–2 h (Sleep-EDF about 8 GB from PhysioNet; ISRUC .rar files), training about 2–4 h, analyses about 0.5 h. The 10-hour time guard stops new training runs early if needed.
5. When it finishes, download `sqn2_v2_outputs.zip` from *Output*. To resume or reuse, make a Dataset from the output (`sqn2_cache`, `runs`), attach it and set `PREVIOUS_OUTPUT_DIR`.

**If a download fails.** For ISRUC-III, make the Drive folder "Anyone with the link"; the official server is the fallback. If the ISRUC server is unreachable from Kaggle, run `export_compact.py` on your PC (it reads `D:\AI_Project\…`), upload the zip as a Kaggle Dataset, and set `RAW_*_DIR` or `PREVIOUS_OUTPUT_DIR`.

## Honest expectations
- RQ1, RQ6 and RQ7 are likely to meet their criteria: the v1 ranking quality was already high (ROC-AUC 0.978), and v2 fixes the threshold, the channel bug and the missing context.
- RQ2 and RQ3 depend on real physiology (homeostatic decline is well established) and on montage behaviour.
- RQ4 (attention) and RQ5 (agreement between methods) are genuinely uncertain. A null or mixed result is reported as such.
- Nothing here is clinical validation. Expert N3-intensity annotation (see `feasibility/expert_annotation_requirements.md`) is the step that would turn the rule-derived scores into validated ones.
