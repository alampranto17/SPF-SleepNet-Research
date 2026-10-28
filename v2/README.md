# SleepQualityNet v2: single-channel N3 detection with a rule-derived, factor-decomposed N3 EEG profile

**v2 replaces the v1 notebook results.** The v1 loader read 83/100 ISRUC-I subjects from an EOG channel (`E1-M2`) and selected checkpoints on the test fold. See [`docs/02_v1_audit_and_supervisor_comments.md`](docs/02_v1_audit_and_supervisor_comments.md).

## What v2 produces
For each 30-s epoch of one central EEG channel (C4-M1/A1 or C3-M2/A2):
- **P(N3)** from a CNN-attention network with handcrafted-feature fusion and a dilated temporal-context block (±15 epochs);
- for detected N3 only: **Delta Intensity**, **Slow-Wave Morphology**, **Spectral Dominance** and **N3 Continuity** scores (0–100, montage-referenced percentiles), an exactly decomposable **Overall N3 EEG Rating** and a **Low / Moderate / High N3 slow-wave grade**;
- a cross-fitted **Signal Reliability Score** (0–100, low/medium/high, rejection flag);
- per night: N3 episodes and the **Overnight N3 EEG Profile**.

These are rule-derived research-prototype descriptors of the EEG signal, **not** clinical sleep-quality scores.

## Notebooks
| Notebook | Purpose | Hardware | Time |
|---|---|---|---|
| [`SleepQualityNet_v2_kaggle.ipynb`](SleepQualityNet_v2_kaggle.ipynb) | **Execution.** Data download → compact cache → participant-grouped training (primary model, ablations, 4 published baselines, GBM, leave-one-cohort-out) → pre-specified RQ1–RQ7 analyses → figures → archive and GitHub push. Numbered sections §0–§9 with a notebook map at the top. | Kaggle GPU T4 ×2 | 2–2.5 h when cache and runs are restored |
| [`SleepQualityNet_v2_results.ipynb`](SleepQualityNet_v2_results.ipynb) | **Results in manuscript order.** Reads the saved tables and figures (Kaggle output or `results/<run>/` here) and shows Table 1–3, the pre-specified verdicts and every RQ with its figures. No training, no EEG data, no hard-coded numbers. | CPU | ~1 min |
| [`DOD_alignment_diagnostic.ipynb`](DOD_alignment_diagnostic.ipynb) | Diagnostic for the DOD scorer-hypnogram alignment (documented in the Methods). | CPU | ~5 min |

Both main notebooks are generated from the package by `tools/build_notebook.py` and `tools/build_results_notebook.py`, so notebook and package code are always identical.

## Data
Training: ISRUC-Sleep I (100) and HMC (151). External: ISRUC-II (8 × 2 sessions), ISRUC-III (10), Dreem DOD-H (25) and DOD-O (55), 5 scorers each. Reference for the within-N3 grade: CAP Sleep Database (R&K S3/S4). All datasets are public and downloaded by the execution notebook from their original sources; **no EEG data is stored in this repository** (ISRUC licence: research use only). Full instructions: [`docs/01_PLAN_and_how_to_run.md`](docs/01_PLAN_and_how_to_run.md).

## Reproducibility
- Participant-grouped 5-fold CV; folds frozen at the first training session (`runs/folds.json` in the output archive).
- No test-fold selection: early stopping, temperature scaling and the decision threshold use an inner validation split.
- Fixed seeds; software and hardware recorded per run in `results/<run>/environment.json`.
- Pre-specified criteria in `sqn2/analysis.py::CRITERIA`; verdicts written mechanically to `results/<run>/research_question_results.csv`.
- Exclusions with reasons in `results/<run>/evaluation_exclusions.csv`; parameter values in `methods_parameters.csv`.
- Published baselines trained on identical folds, inputs, selection, threshold, loss and class weight; deviations from each original recipe in `sqn2/baselines.py::BASELINE_NOTES`.

## Repository layout
| Path | Content |
|---|---|
| `sqn2/` | package: `data`, `download`, `features`, `scores`, `models`, `baselines`, `train`, `stats`, `analysis`, `xai`, `report`, `figures`, `pipeline`, `kaggle_io`, `github_io` |
| `results/<run>/` | tables, figures and progress log of each Kaggle run (pushed automatically); `20261027-1633` is the first complete run |
| `tools/` | notebook builders and diagnostics |
| `docs/` | plan, v1 audit and supervisor-comment resolution, literature/novelty check, feasibility audit |
| `tests/make_synthetic.py` | synthetic EDF generator, **for code testing only** |
| `export_compact.py` | standalone local exporter (copy of `sqn2/data.py`) |

## Status
First complete real-data run: `results/20261027-1633`. The next run adds the DOD scorer-hypnogram alignment fix, the environment record and the additional figures (confusion matrices, ROC, calibration, per-participant F1, baseline forest plot, Bland–Altman, factor-by-stage, inter-rater).
