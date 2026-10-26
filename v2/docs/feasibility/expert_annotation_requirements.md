# Expert annotation requirements (future validation of the factor scores)

There are no expert ratings of N3 slow-wave expression in ISRUC or Sleep-EDF, so the factor scores in v2 are validated only against physiological and human-disagreement references. A small annotation study would turn them into **expert-validated** scores, and is a natural task for an international collaborator's sleep lab.

| Item | Recommendation |
|---|---|
| Raters | ≥ 3 registered polysomnographic technologists or sleep physicians, independent of the authors (one researcher's rating is not ground truth) |
| Epochs | about 600 detected-N3 epochs (confirm with a sample-size calculation for the target ICC precision before starting) |
| Sampling | Stratified by cohort (ISRUC-I / Sleep-EDF / ISRUC-III), participant (≤ 6 per participant), time of night (early vs late) and model rating quintile |
| Presentation | 30-s single-channel trace with ±1 epoch context, fixed gain and montage label, randomised order, blinded to model scores, stage labels and participant identity |
| Scale | 5-point ordinal slow-wave expression: weak / mild / moderate / strong / very strong (anchored examples); optional 0–100 slider |
| Agreement | Weighted Cohen's κ (pairs), Fleiss' κ (≥ 3 raters, ordinal collapsed), ICC(2,k) for the slider |
| Consensus | Median rating; adjudication session for disagreements ≥ 2 points |
| Use | Correlation (Spearman, ICC) of each factor and of the overall rating with consensus; the weights may be refit **only** on a training subset, with the rest held out |
| Splits | Annotation participants fixed in advance; reuse the CV participant folds to prevent leakage |
