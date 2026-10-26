# SleepQualityNet v2: plan, design and how to run on Kaggle

## Goal
From one EEG channel, (1) detect N3, (2) describe each detected N3 epoch with transparent factor scores and an overall **N3 EEG Rating**, (3) attach a **Signal Reliability Score**, and (4) summarise each night as an **Overnight N3 EEG Profile**. The whole design is evaluated so that every research question has an independent reference and a criterion fixed before results exist.

## Data (all public; Sleep-EDF removed as outdated: 1989–94 recordings, R&K scoring, Fpz-Cz)
| Cohort | Role | Channel | Notes |
|---|---|---|---|
| ISRUC-Sleep subgroup I (100, sleep disorders) | training / CV | C4-A1 = C4-M1 (fallback C3) | Two scorers (`_1.txt`, `_2.txt`); age, sex and diagnosis from the ISRUC details sheet. |
| HMC, PhysioNet v1.1 (~150 clinical PSGs, 2018) | training / CV | C4-M1 | AASM v2.4 technician scoring; CC BY 4.0. Only aggregated demographics. |
| ISRUC-Sleep subgroup II (8 healthy, 2 sessions each) | external only | C4-M1 | Test–retest of the Overnight N3 EEG Profile. |
| ISRUC-Sleep subgroup III (10 healthy) | external only | C4-M1 | Two scorers. |
| Dreem DOD-H (25 healthy) / DOD-O (55 OSA) | external only | C3-M2 / C4-M1 | **5 expert scorers per recording.** Consensus = majority vote; also the graded expert-agreement reference. Zenodo, MIT licence. |

External cohorts are never used for training, model selection or thresholds. They are predicted by the ensemble of the 5 fold models.

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
| RQ1 detection | expert hypnogram; 2- and 5-scorer inter-rater benchmark; LOCO; 4 external cohorts | pooled F1 ≥ 0.75 and PR-AUC ≥ 0.85; full support needs F1 ≥ 0.70 in each training and external cohort |
| RQ2 factors | homeostatic decline; expert N3 vs N2; **graded expert agreement (fraction of 2–5 experts voting N3)**; age (ISRUC) | every factor early > late and N3 > N2 (p < 0.05); full support also needs the overall rating to rise with the fraction of experts voting N3 |
| RQ3 overall rating | weight perturbation; within-cohort contrasts; split-half; ISRUC-II test–retest | 5th-percentile ρ ≥ 0.90 and early > late within each cohort |
| RQ4 components | identical folds, participant-level paired tests | all effects estimated with CI (any direction is a finding) |
| RQ5 XAI | band-stop perturbation, band-decomposed IG, SHAP surrogate, attention alignment | delta top in ≥ 70 % of participants for perturbation and IG, and SHAP top group delta with fidelity ≥ 0.70 |
| RQ6 reliability | prediction errors (OOF and external); expert disagreement (ISRUC 2, DOD 5 scorers) | AUROC ≥ 0.75 and F1@80 % coverage > F1@100 % |
| RQ7 profiles | expert-hypnogram profile (OOF and external); ISRUC-II test–retest | ICC(2,1) ≥ 0.75 for N3 minutes (OOF) |

## Running on Kaggle (free tier)
1. Create a notebook and upload `SleepQualityNet_v2_kaggle.ipynb` (*File → Import Notebook*).
2. *Settings*: **Accelerator: GPU T4 ×2**, **Internet: ON** (needs a phone-verified account).
3. Set `QUICK_TEST = True` and *Run All*. It takes about 30 minutes and checks every download source and stage on a small subset.
4. **Commit 1 (data).** Set `QUICK_TEST = False` and `RUN_STAGES = ["data"]`, then *Save Version → Save & Run All (Commit)*. About 90 GB is streamed (HMC ~13 GB, ISRUC ~18 GB, DOD ~58 GB through range requests); only the compact cache (~2–3 GB) is kept. Then *Output → New Dataset* from `sqn2_cache`.
5. **Commit 2 (train + analyse).** Attach that dataset, set `PREVIOUS_OUTPUT_DIR = "/kaggle/input/<dataset>"` and `RUN_STAGES = ["train", "analyse"]`, then commit again. Expected time: training about 3–5 h, analyses about 0.5–1 h. The 10-hour time guard stops new training runs early if needed, and a re-run resumes.
6. Download `sqn2_v2_outputs.zip` from *Output*.

**If a download fails.**
- ISRUC-III: share the Drive folder as "Anyone with the link"; the official server is also tried.
- ISRUC server unreachable from Kaggle: run `export_compact.py` on your PC with your `D:\AI_Project\…` folders and upload the zip as a Dataset.
- DOD range streaming refused: set `DOD_ALLOW_FULL_ZIP_DOWNLOAD = True` (needs about 40 GB free disk).

## Honest expectations
- RQ1, RQ6 and RQ7 are likely to meet their criteria: the v1 ranking quality was already high (ROC-AUC 0.978), and v2 fixes the threshold, the channel bug and the missing context.
- RQ2 and RQ3 depend on real physiology (homeostatic decline is well established) and on montage behaviour.
- RQ4 (attention) and RQ5 (agreement between methods) are genuinely uncertain. A null or mixed result is reported as such.
- Nothing here is clinical validation. Expert N3-intensity annotation (see `feasibility/expert_annotation_requirements.md`) is the step that would turn the rule-derived scores into validated ones.
