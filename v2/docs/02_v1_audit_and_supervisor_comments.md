# v1 audit and resolution of the supervisor's comments

Sources audited: `SPF_SleepNet_N3_research_final.ipynb` (107 cells, executed outputs) and the manuscript draft *"SleepQualityNet: A Dual-Output CNN-Attention Architecture…"* (17 pages, 23 inline comments by Kamrul).

---

## A. Critical problems found in the v1 code (they affect the reported numbers)

| # | Problem | Evidence in v1 | Consequence | v2 fix (file) |
|---|---|---|---|---|
| A1 | **Wrong EEG channel for 83/100 ISRUC-I subjects.** The loader searched only `C4-A1`, `F4-A1`, `C3-A2`, then silently fell back to `raw.ch_names[0]`. For subjects 11, 13, 14, 17, 19–21, 23, 25 and 27–100 no `C4-A1` channel was found, so `E1-M2` (EOG) was used (subject 40: `ROC`). ISRUC files label the same derivation either `C4-A1` or `C4-M1`; the v2 channel audit confirms the exact name per file. | Cell 17 output: `E1-M2: 82 subjects, C4-A1: 17, ROC: 1`. | **The manuscript's headline results (F1 0.7917 ± 0.0172, 100 participants, 83,523 epochs, "C4–A1") were computed mostly on an EOG channel and cannot be reported.** The later 95-participant notebook dropped those subjects instead of fixing the channel. ISRUC-III used the same loader, so the external result (F1 0.712) is also unverified. | `data.pick_channel` accepts the A1/M1 spellings of the same derivation and never falls back to a non-EEG channel (`ISRUC_EEG_CANDIDATES`). The channel audit is printed and saved as `channel_audit.csv`. |
| A2 | **Threshold fixed at 0.5 with pos_weight 10.6 and label smoothing.** | Precision 0.49–0.50 with ROC-AUC 0.978 and PR-AUC 0.905. | The F1 of 0.65 reflects a poor operating point, not poor discrimination. | Threshold maximising F1 is chosen on **inner-validation participants** only (`train.best_f1_threshold`). |
| A3 | **Model selection on the test fold.** The best EMA checkpoint was chosen by validation loss on the held-out fold. | `SleepQualityTrainer.train` saves on `val_loss` from `val_loader`, which is the CV test fold. | Optimistic bias in every CV number. | Inner validation = 15 % of training participants per cohort, used for early stopping, threshold and temperature (`train.make_folds`). |
| A4 | **Epoch order destroyed.** `create_epochs` deleted artifact / low-quality epochs without keeping their index. | Retention ranged from 22.7 % to 100 %, with no epoch index stored. | N3 continuity, episodes and whole-night profiles were impossible. Deleting low-SNR epochs also biases the evaluation toward easy epochs. | All epochs kept with `epoch_index`; v1 rules stored as flags (`flag_artifact`, `flag_lowq`, `flag_clip`). Primary metrics include all scored epochs; clean-only results are a sensitivity analysis. |
| A5 | **Circular quality target.** `q* = 0.6·delta + 0.4·SWA`, built from features the model receives. | Ridge regression R² = 0.9997, direct formula R² = 0.9787. | Quality R² is target reconstruction, not evidence of anything. | Regression head removed. Factor scores are transparent formulas (`scores.py`), validated only against **independent** references: homeostatic decline, two-scorer disagreement, night-to-night reproducibility and the expert hypnogram. |
| A6 | **"Slow-wave density" was a peak counter.** | Counts positive and negative peaks above half the threshold; no zero-crossing pairing or durations. | Not a slow-wave measure. | Zero-crossing full-wave detector giving negative/positive half-waves, peak-to-peak amplitude, durations and slopes (`features.detect_slow_waves`). |
| A7 | **"Fragmentation index" was a broadband zero-crossing rate.** | `extract_fragmentation_index`. | It does not measure N3 continuity. | Continuity is computed from true chronological neighbours (`scores.continuity_score`). |
| A8 | **Ablation pseudo-replication** (3 seeds on one split), which a later cell already flagged. | Cells 89, 92, 96. | — | Every variant uses identical 5 participant folds; participant-level Wilcoxon tests with Holm correction and bootstrap CIs (`analysis.rq4`). |
| A9 | Leakage in `EEGPreprocessingPipeline.prepare`: the split was "first 80 % of the concatenated array" and quality labels used the full dataset. | Cell 26. | Minor (superseded later). | Removed. |
| A10 | Speed: two forward passes per batch, Sleep-EDF up-sampled from 100 to 200 Hz, no mixed precision; about 54 min per run on the RTX 3050. | Cell 44 output. | The full protocol doesn't fit on Kaggle. | GPU-resident data, AMP, 100 Hz, chunked sequence batches: minutes per run. |

---

## B. Supervisor comments: point-by-point resolution

"Code" means fixed in v2 and quoted from `methods_parameters.csv`. "Text" means a manuscript wording change, with the action stated.

| # | Comment (Kamrul) | Type | Resolution |
|---|---|---|---|
| 1 | Replace the first abstract/intro paragraph with the suggested opening | Text | Adopt the suggested paragraph. Add a citation for the memory-consolidation and health-decline claim. |
| 2 | "Add a ref here" (high-amplitude delta oscillations) | Text | Cite the AASM Manual (Berry et al.) and a slow-wave physiology review (e.g. Achermann & Borbély). |
| 3 | The central argument is implicit | Text | v2 argument: *"Binary N3 detection alone hides large within-N3 variation; we show that a transparent, factor-decomposed N3 EEG profile can be produced from one channel and validated against references independent of its construction (homeostatic decline, inter-scorer disagreement, night-to-night reproducibility)."* |
| 4 | "here first" ("To the best of our knowledge") | Text | Use the phrase once, at the first novelty claim. |
| 5 | Inconsistent terms (deep sleep quality / quality index / quality score) | Code + Text | v2 uses fixed names throughout code and outputs: **N3 EEG Rating**, **Delta Intensity / Slow-Wave Morphology / Spectral Dominance / N3 Continuity Score**, **Signal Reliability Score**, **Overnight N3 EEG Profile**. "Quality" is dropped. |
| 6 | Dataset name inconsistent (ISRUC-Sleep-III vs Sub-group I) | Code + Text | v2: training cohorts are **ISRUC-Sleep subgroup I** and **Sleep-EDF Expanded (sleep-cassette)**; **ISRUC-Sleep subgroup III** is external only. Cohort labels in all outputs: `ISRUC-I`, `Sleep-EDF`, `ISRUC-III`. |
| 7 | Missing motivation for single-channel raw input | Text | Add cost, comfort, home/wearable deployability and fewer failure points. The single central or frontal derivation matches headband devices. |
| 8 | "To the best of our knowledge" appears twice | Text | Same as #4. |
| 9 | "Research gap" vs "challenge" wording | Text | Call all of them "gaps". |
| 10 | Mention in the intro how the (surrogate) target was created | Code + Text | The surrogate regression target no longer exists. State that the factor scores are **rule-derived** (formulas given), scaled to training-fold percentiles of expert-N3 epochs of the same montage, and that no expert quality labels exist. |
| 11 | Gaps and contributions are redundant; add an explainability gap | Text | RQs = questions (RQ1–RQ7). Contributions = (i) channel-verified heterogeneous cohort with external and leave-one-cohort-out testing, (ii) factor-decomposed N3 EEG profile, (iii) independent-reference validation, (iv) reliability score validated against errors and inter-scorer disagreement, (v) multi-method XAI with **explicit agreement metrics**. |
| 12 | Add quantitative results to the intro | Text | Fill from `research_question_results.csv` after the Kaggle run. |
| 13 | Add a "remainder of the paper" sentence | Text | Add one. |
| 14 | `u_i` and `τ` undefined | Code | The quality head (scaled sigmoid with learnable temperature) is removed. Every remaining quantity has an explicit definition in `scores.py`, `features.py` and `methods_parameters.csv`. |
| 15 | "Artifact screening" and "signal-quality screening" have no thresholds | Code | Exact rules: \|x\| > 300 µV; max z > 5; P(30–40 Hz) > 0.5·P(0.5–4 Hz); > 100 samples beyond ±4 SD per 30 s at 200 Hz; quality score ≤ 0.6 with the stated deductions. They are stored as flags, not used for deletion. See `methods_parameters.csv` rows "QC flag: …". |
| 16 | Rationale for the 5-IQR Winsorisation | Code | The log, 2–98 % clip, 5-IQR winsorise, z chain is replaced by **one** step: (x − median_train)/IQR_train, clipped at ±10. The fraction of clipped training values is reported per run (`feature_values_clipped_frac` in `rq4_variant_pooled_metrics.csv`) to show the clip only touches rare outliers. |
| 17 | ε appears in four equations with no value | Code | Values listed: band power +1e-6 µV², ratio/log denominators +1e-8 to 1e-12, IQR floor (IQR < 1e-6 → 1). |
| 18 | State the percentile-clipping bounds | Code | Percentile clipping removed (see #16). |
| 19 | "The notebook also configured three-seed ablation…" reads like an implementation note | Code + Text | v2 protocol: 5-fold participant CV stratified by cohort, inner validation, identical folds for all variants, fold membership saved in `runs/folds.json`. Describe exactly that. |
| 20 | `w+` has no formula | Code | `w+ = N_non-N3 / N_N3` over scored training-fold epochs (inner-validation participants excluded); `w+ = 1` in unweighted variants. Stored per run (`pos_weight`). |
| 21 | Cite the dataset | Text | ISRUC: Khalighi et al., *Comput. Methods Programs Biomed.* 2016. Sleep-EDF: Kemp et al., *IEEE TBME* 2000, and Goldberger et al., *Circulation* 2000 (PhysioNet). |
| 22 | Rationale for task weight 0.3, non-N3 weight 0.2, augmentation probabilities 0.40/0.25/0.25 | Code | Multi-task weights removed (no regression head). Augmentation reduced to amplitude scale U(0.9, 1.1) and Gaussian noise (SD 0.02) on every batch, to build invariance to inter-recording gain and noise. Time-shift removed because training chunks are contiguous sequences. Not tuned on test data. |
| 23 | "notebook" appears 3–4 times | Text | Replace with "implementation" or describe the procedure directly. |
| 24 | (a) How are feature-disabled / weighting-disabled variants built? (b) Which SHAP model? (c) Which gradient method? (d) Which band-removal filter? (e) What alignment metric? | Code | (a) Feature vector **removed** from the fusion input (parameter counts reported per variant); weighting disabled = `w+ = 1`. (b) TreeExplainer on a **Gradient-Boosting regression surrogate** of the network logit; fidelity R² on held-out epochs is reported. (c) **Integrated gradients** (zero baseline, 32 steps) plus gradient×input; band attribution by band-decomposed IG. (d) **4th-order Butterworth band-stop, zero-phase**, target epoch only, features recomputed (coherent intervention). (e) **Spearman correlation** between the 32 attention weights and per-segment slow-wave coverage, plus an attention **enrichment ratio**. All are in the `xai.py` docstring. |

---

## C. What must change in the manuscript before resubmission
1. Remove every number obtained with the v1 loader (sections Abstract, IV–V, Tables V–VI, Figs. 3–5) and replace it with v2 outputs.
2. Methods: rewrite from `methods_parameters.csv`. Delete the quality-target subsection (eqs. 15–19) and the quality head (eq. 6).
3. Results: report `research_question_results.csv` exactly, including any criterion not met.
4. Limitations: no expert or clinical N3-quality reference; factor scaling is montage-referenced; ISRUC-I is a clinical population (sleep disorders), Sleep-EDF SC is a healthy population.
