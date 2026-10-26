"""SleepQualityNet v2 — N3 detection + rule-derived N3 EEG factor scores.

Modules
-------
data      : dataset loaders (ISRUC-I, ISRUC-III, Sleep-EDF), per-epoch QC flags,
            compact per-recording cache (.npz), Google-Drive download helpers.
features  : vectorised spectral features, zero-crossing slow-wave detector.
scores    : rule-derived factor scores (0-100), overall N3 EEG Rating,
            episode / whole-night N3 profiles.
models    : SleepQualityNet v2 (CNN + optional attention, feature fusion,
            optional adjacent-epoch temporal context).
train     : GPU-resident trainer, participant-grouped CV, inner-validation
            threshold selection, MC-dropout inference.
analysis  : statistics and RQ1-RQ7 evaluations with pre-specified criteria.
"""
__version__ = "2.0.0"
