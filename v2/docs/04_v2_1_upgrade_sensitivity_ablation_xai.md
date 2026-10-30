# v2.1: sensitivity analyses, extended ablations and extended explanations

Added on 30 Oct 2026, **after** the first complete real-data run (`results/20261028-0834`). Everything here is reported separately from the pre-specified verdicts in `research_question_results.csv`. Those verdicts are not changed, and the criteria in `sqn2/analysis.py::CRITERIA` are untouched.

## 1. Why the "partially supported" verdicts stay as they are

| RQ | Verdict (run 0834) | What failed | Why it cannot be "fixed" after the fact |
|---|---|---|---|
| RQ1 | PARTIALLY_SUPPORTED | F1 ≥ 0.70 in every external cohort: DOD-H 0.704 passes, **DOD-O 0.580** fails | The criterion was fixed on 26 Oct, before any full result, and the repository history shows that. Lowering the bar, dropping DOD-O, or re-tuning the threshold on DOD after seeing it would be HARKing. Reviewers can see the commit dates. |
| RQ2b | NOT_SUPPORTED | G3: S4-vs-S3 AUROC on detected N3 = **0.743** (< 0.75). G1 and G2 are met. | Switching to the subset that passes (all expert S3/S4 epochs, AUROC 0.782) or to another grading variable after seeing the numbers is a post-hoc choice. |

What *is* legitimate, and what v2.1 adds:
1. **Explain the miss with evidence.** On DOD-O the five experts agree with each other at a mean F1 of only **0.603**. The 0.70 criterion is above the human ceiling for that cohort, and the model sits inside the expert-to-expert range (`rq1_interrater_benchmark.csv`). The new `sensitivity_rq1*.csv` tables show whether the DOD false positives are N2 epochs that some of the five experts called N3, and how F1 changes under each reference standard (consensus, individual experts, unanimous epochs, lenient, strict).
2. **Report the robustness of every conclusion** (sensitivity matrix, below).
3. **Pre-declare any model change and test it on data it was not designed on.** No change in v2.1 is tuned on DOD or CAP. A future improvement must be declared first, then judged by leave-one-cohort-out (training cohorts only) before it is ever evaluated on the external cohorts again.

## 2. Sensitivity analyses (`sqn2/sensitivity.py` → `sensitivity_*.csv`, `sensitivity_matrix.csv`)

| RQ | Before v2.1 | Added |
|---|---|---|
| RQ1 | QC-clean epochs (OOF only), fold-wise results, LOCO | decision threshold (0.5, ±0.10, calibrated, plus a label-optimised **upper bound**); reference standard; QC-clean epochs in every cohort; participants with ≥ 10 N3 epochs; false-positive structure (expert stage, expert vote fraction, slow-wave coverage) |
| RQ2 | per cohort | decline on **expert** N3 epochs (model-independent); first vs last third of the night; QC-clean epochs |
| RQ2b | two subsets | per-participant AUROC; **EXPLORATORY** cross-fitted High threshold on CAP participant halves; rating as an alternative grading variable (exploratory) |
| RQ3 | weights, LOFO, split-half, test-retest | collected into the matrix |
| RQ4 | none | seed noise floor, leave-one-component-out (section 3) |
| RQ5 | none | IG baseline and steps, participant bootstrap, per external cohort (section 4) |
| RQ6 | per external cohort | per cohort, confidence-only score without the OOD and QC components |
| RQ7 | none | episode definition (min 1/2/3 epochs × gap 0/2/4), per cohort, QC-clean |

`sensitivity_matrix.csv` has one row per RQ × analysis × setting. `holds` says whether the primary conclusion survives that setting.

## 3. Ablation study (`analysis.rq4_extended`, new variants in `train.VARIANTS`)

The pre-specified RQ4 adds each component to a model **without** temporal context. Only context had an effect (ΔF1 = +0.023), so the other components were never tested in the model that is actually reported. Added:

| variant | change from `full_v2` |
|---|---|
| `full_nofeat` | no handcrafted-feature fusion |
| `full_nopw` | no positive-class weighting |
| `full_noattn` | mean pooling instead of attention |
| `full_ctx_r3` | context ±3 epochs instead of ±15 |
| `full_deltafeat` | only the 10 delta/slow-wave features |
| `full_v2_seed1`, `full_v2_seed2` | training seed +1 / +2 |

* `rq4_extended_component_effects.csv`: participant-level paired Wilcoxon tests with 95 % CIs and their own Holm family. An effect counts only if p_holm < 0.05 **and** it exceeds the seed-noise floor (the largest seed-to-seed median difference of the primary model).
* `rq4_seed_variability.csv`: pooled metrics of the three seeds, SD and range.
* `rq4_variant_external_metrics.csv`: every ablation variant on every external cohort.

Cost on Kaggle T4 ×2 is about 1.6 min per fold per variant, so roughly 1 h for the 7 variants. Existing runs are restored from `sqn2-runs` and only the new variants are trained. The frozen folds are reused.

## 4. Explainability (`sqn2/xai_ext.py` → `rq5x_*`)

The pre-specified RQ5 explains only correctly detected N3 epochs from the training cohorts (120 per fold). Added:
* **Outcome-stratified explanations.** TP, FP, FN and TN-N2 epochs, out-of-fold and in **every external cohort** (5 fold models, attributions averaged). Same methods: band-stop perturbation with recomputed features, band-decomposed integrated gradients, and attention vs slow waves.
* **SHAP by outcome.** Signed group attribution shows whether the false positives are pushed to N3 by delta features. Every one of the 19 features is ranked by mean |SHAP| over folds.
* **Robustness.** IG zero baseline vs noise baseline; completeness error at 32 vs 64 steps; a participant-bootstrap CI for the RQ5 criterion; and the criterion re-applied per external cohort.
* **Figures.** `fig_rq5x_band_by_outcome.png`, `fig_rq5x_shap_features.png`, and `fig_rq5x_examples.png` (EEG with IG shading, attention and slow-wave coverage for a TP, a DOD false positive and a FN).

## 5. Other fixes
* **Results notebook run selection.** `GITHUB_RUN = "latest"` picked `20261028-1400`, a run from a container with no data access: no GPU, 16 participants, every download refused with 403. Its F1 of 0.98 is meaningless. "latest" now takes the newest run that has a GPU recorded, is not a quick test, and has complete training cohorts (ISRUC-I ≥ 95, HMC ≥ 145). Naming an invalid run explicitly prints a warning. **Report only `20261028-0834` (or a later valid run).**
* `analysis.rq2_bands`: the S4-vs-S3 AUROC for slow-wave morphology was never computed because the code asked for a non-existent column (`slow_wave_morphology` instead of `sw_morphology`). It now appears in `rq2_rk_grade_validation.csv`.

## 6. How to run
Upload the regenerated `SleepQualityNet_v2_kaggle.ipynb` and *Run All* with the `sqn2-cache` and `sqn2-runs` datasets attached, exactly as before. Only the 7 new variants train; then the full analysis, including v2.1, runs. Open `SleepQualityNet_v2_results.ipynb`; section 11 shows the v2.1 outputs.

## 7. Testing done before the push
* `sensitivity.run_all`: run end to end on a synthetic prediction table with the real column layout. All blocks ran without errors.
* `analysis.rq4_extended`: run on synthetic run files for every variant.
* Results notebook: executed end to end on the real `20261028-0834` tables (v2.1 sections report "not in this results folder"), and again with synthetic v2.1 tables (all sections render).
* Run selection: tested on the real `v2-kaggle` branch archive. It skips `1400` and selects `0834`.
* `xai_ext` and the new training variants need PyTorch, which could not be installed in this environment. They reuse the code paths of `xai.rq5` and `Trainer.run`, but **their first execution is the next Kaggle run**. Check `analysis_errors.txt` after it.
