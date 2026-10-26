"""Data inventory, methods-parameter table and GO/NO-GO table, generated from the
loaded corpus and code constants (so the manuscript can quote exact values)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import data as D
from . import features as FT
from . import scores as SC


def data_inventory(corpus, out_dir):
    rec, ep = corpus.rec, corpus.ep
    notes = corpus.participant_meta.get("notes", {})
    pm = corpus.participant_meta.get("participants", {})
    rows = []
    for coh, g in rec.groupby("cohort"):
        e = ep[ep.cohort == coh]
        parts = g.participant.unique()
        has_age = sum(1 for p in parts if p in pm and pm[p].get("age") == pm[p].get("age"))
        nr = int(g.n_raters.max()) if "n_raters" in g else 1
        rows += [
            (coh, "participants", "AVAILABLE", f"{len(parts)}", "cache meta.participant"),
            (coh, "recordings", "AVAILABLE", f"{len(g)}", "cache meta.recording"),
            (coh, "nights", "AVAILABLE", ",".join(map(str, sorted(g.night.unique()))), "meta.night"),
            (coh, "continuous recording / epoch order", "AVAILABLE",
             "all epochs of the analysis window kept with epoch_index", "npz epoch_index"),
            (coh, "recording start/end", "PARTIALLY_AVAILABLE",
             "crop offset (s) relative to file start; clock time not retained", "meta.crop_start_s"),
            (coh, "EEG channel", "AVAILABLE",
             ", ".join(f"{k}:{v}" for k, v in g.channel_raw.value_counts().items()), "meta.channel_raw"),
            (coh, "sampling frequency", "AVAILABLE",
             f"native {sorted(g.native_fs.unique())} Hz -> analysed at {D.TARGET_FS} Hz", "meta.native_fs"),
            (coh, "sleep-stage labels (scorer 1)", "AVAILABLE",
             f"{int((e.label >= 0).sum()):,} scored epochs", "npz labels"),
            (coh, "independent expert scorers", "AVAILABLE" if nr >= 2 else "NOT_AVAILABLE",
             f"{nr} scorer(s); reference = {g.reference.iloc[0] if 'reference' in g else ''}",
             "npz labels_multi"),
            (coh, "arousal labels", "NOT_AVAILABLE", "not loaded by this pipeline", ""),
            (coh, "respiratory-event labels", "NOT_AVAILABLE", "not loaded by this pipeline", ""),
            (coh, "movement-event labels", "NOT_VERIFIED", "not loaded; unscored epochs = -1", ""),
            (coh, "sleep efficiency / TST / WASO", "PARTIALLY_AVAILABLE",
             "derivable from the expert hypnogram only (no sleep/wake model)", "labels"),
            (coh, "subjective sleep-quality scores", "NOT_AVAILABLE", "", ""),
            (coh, "clinical diagnoses", "NOT_VERIFIED", "not loaded", ""),
            (coh, "age / sex", "AVAILABLE" if has_age else "NOT_AVAILABLE",
             f"{has_age}/{len(parts)} participants; {notes.get(coh, '')}", "participant_metadata.json"),
            (coh, "medication / health outcomes", "NOT_AVAILABLE", "", ""),
        ]
    df = pd.DataFrame(rows, columns=["dataset", "field", "status", "detail", "source"])
    df.to_csv(os.path.join(out_dir, "data_inventory.csv"), index=False)
    return df


def methods_parameters(cfg, out_dir):
    rows = [
        ("Analysis sampling rate", f"{D.TARGET_FS} Hz", "polyphase resampling (scipy.resample_poly)"),
        ("Model band-pass", "0.3-35 Hz, 4th-order Butterworth, zero-phase, whole recording", ""),
        ("QC band-pass", "0.5-45 Hz (or 0.95*Nyquist), 4th-order Butterworth, zero-phase", ""),
        ("Epoch length", f"{D.EPOCH_SEC} s ({D.EPOCH_SAMPLES} samples)", ""),
        ("EEG derivation (all cohorts)", "C4-A1 = C4-M1 (priority), then C3-A2 = C3-M2; no non-EEG "
         "fallback", "A1/M1 are the same mastoid reference"),
        ("HMC analysis window", "first to last scored epoch (AASM v2.4 technician scoring)", ""),
        ("DOD reference label", "majority vote of 5 scorers; ties -> scorer with highest mean agreement; "
         "record 5d8e93a5 (no consensus) excluded", "Guillot et al. 2020"),
        ("Stage mapping", "ISRUC code 4 (old S4) -> N3; unscored/movement -> -1 (excluded from loss/metrics)", ""),
        ("QC flag: amplitude", "max |x| > 300 uV", "v1 rule, stored as flag_artifact"),
        ("QC flag: z-score", "max |x - mean| / SD > 5", "v1 rule"),
        ("QC flag: HF vs delta", "P(30-40 Hz) > 0.5 * P(0.5-4 Hz)", "v1 rule"),
        ("QC flag: clipping", "> 100 samples beyond mean +/- 4 SD per 30 s at 200 Hz (scaled by fs)", "v1 rule"),
        ("QC flag: low quality", "score <= 0.6; score = 1 - 0.3[P30-40/P0.5-15 > 0.3] - 0.3[SNR < 3 dB] "
         "- 0.4[RMS < 1 uV]; SNR = 10 log10(P0.5-4 / P30-45)", "v1 rule"),
        ("QC usage", "flags are NOT used to delete epochs; primary metrics include all scored epochs; "
         "clean-only metrics reported as sensitivity analysis", "fixes order loss"),
        ("Raw-input scaling", "divide each recording by the median per-epoch SD of that recording", ""),
        ("Welch PSD", "4-s Hann windows, 50 % overlap (0.25 Hz resolution)", ""),
        ("Bands", ", ".join(f"{k} {v[0]}-{v[1]} Hz" for k, v in FT.BANDS.items()), ""),
        ("Slow-wave detector", "0.3-4 Hz 2nd-order zero-phase band-pass; negative half-wave 0.25-1.0 s; "
         "full wave 0.5-2.0 s; AASM wave p2p >= 75 uV", "zero-crossing full-wave"),
        ("Handcrafted detection features", f"{len(FT.DETECTION_FEATURES)}: " + ", ".join(FT.DETECTION_FEATURES), ""),
        ("Feature standardisation", "(x - median_train) / IQR_train; clipped to +/-10 IQR "
         "(bounds the influence of residual artifacts; fraction clipped is reported)", "single step; "
         "replaces v1 log/2-98 % clip/5-IQR winsor/z chain"),
        ("Numerical epsilons", "PSD band power +1e-6 uV^2; ratio/log denominators +1e-8..1e-12; "
         "IQR floor 1e-6 -> 1.0", "all values explicit in code"),
        ("Positive-class weight", "w+ = N_non-N3 / N_N3 over scored training-fold epochs (inner-validation "
         "participants excluded); w+ = 1 in unweighted variants", ""),
        ("Loss", "binary cross-entropy (no label smoothing, no regression head)", ""),
        ("Optimiser", f"AdamW lr={cfg['lr']} wd={cfg['wd']}; linear warm-up {cfg['warmup_steps']} steps, cosine decay", ""),
        ("Batch", f"{cfg['chunks_per_batch']} chunks x {cfg['chunk_len']} consecutive epochs", "identical for all variants"),
        ("Augmentation", "per-chunk amplitude scale U(0.9,1.1) and Gaussian noise SD 0.02 (in recording-"
         "scaled units), every batch", "gain/noise invariance; not tuned on test data"),
        ("Early stopping", f"inner-validation PR-AUC, patience {cfg['patience']}, max {cfg['max_passes']} passes", ""),
        ("Inner validation", "15 % of training participants per cohort", "no outer-test data used for selection"),
        ("Decision threshold", "maximises F1 on inner-validation participants", ""),
        ("Calibration", "temperature scaling on inner validation (reported, not used for thresholding)", ""),
        ("MC dropout", f"{cfg['mc_passes']} stochastic passes", "uncertainty for RQ6"),
        ("OOD distance", "Mahalanobis distance of 64-d embedding to nearest class mean "
         "(shared covariance from training embeddings)", ""),
        ("Factor scaling", "percentile of training-fold expert-N3 epochs, same montage family", "0-100"),
        ("Overall rating weights", ", ".join(f"{k}={v}" for k, v in SC.PRIOR_WEIGHTS.items()),
         "design parameters, not clinically established"),
        ("Continuity window", f"+/-{SC.CONTINUITY_HALF_WINDOW} epochs", ""),
        ("N3 episode", "runs of predicted N3 merged across gaps <= 2 epochs; >= 2 N3 epochs", ""),
        ("Cross-validation", "5 folds by participant (ISRUC-I + HMC), stratified by cohort", ""),
        ("Within-N3 grade", "slow-wave coverage (0.5-2 Hz full waves, p2p >= 75 uV) per 30-s epoch: "
                            "Low < 20 %, Moderate 20-50 %, High > 50 % (R&K stage 3 / 4 anchors); "
                            "secondary: N3 EEG Rating 0-33 / 33-67 / 67-100; detected N3 epochs only"),
        ("Grade reference", "CAP Sleep Database (PhysioNet, 108 PSGs, R&K scoring with S3 and S4 "
                            "separate, C4-A1/C3-A2): external only, never used for training or thresholds"),
        ("Same-protocol baselines", "DeepSleepNet, TinySleepNet, AttnSleep, L-SeqSleepNet re-implemented at "
                                    "100 Hz; raw EEG only; identical folds, inner-validation selection, "
                                    "threshold, temperature, loss and positive-class weight as the proposed model"),
        ("External cohorts", "ISRUC-II (test-retest), ISRUC-III, DOD-H, DOD-O: never used for training, "
         "selection or thresholds; predicted by the 5-fold model ensemble", ""),
        ("Seed", f"{cfg['seed']} (per-fold seed derived deterministically)", ""),
    ]
    df = pd.DataFrame(rows, columns=["parameter", "value", "note"])
    df.to_csv(os.path.join(out_dir, "methods_parameters.csv"), index=False)
    return df


def go_no_go(summary_df, out_dir):
    st = dict(zip(summary_df.research_question, summary_df.status)) if summary_df is not None else {}

    def g(rq, full="GO", part="GO_WITH_LIMITATIONS"):
        s = st.get(rq)
        return full if s in ("SUPPORTED", "ANSWERED") else part if s and "PARTIAL" in s else "HOLD"
    rows = [
        ("Binary N3 detection", g("RQ1")),
        ("Delta Intensity Score", g("RQ2", "GO_WITH_LIMITATIONS")),
        ("Slow-Wave Morphology Score", g("RQ2", "GO_WITH_LIMITATIONS")),
        ("Spectral Dominance Score", g("RQ2", "GO_WITH_LIMITATIONS")),
        ("N3 Continuity Score", g("RQ2", "GO_WITH_LIMITATIONS")),
        ("Reliability Score", g("RQ6")),
        ("Overall N3 EEG Rating", g("RQ3", "GO_WITH_LIMITATIONS")),
        ("Episode-level rating", g("RQ7", "GO_WITH_LIMITATIONS")),
        ("Whole-night N3 profile", g("RQ7")),
        ("Within-N3 grade (Low/Moderate/High)", g("RQ2b", "GO_WITH_LIMITATIONS")),
        ("Clinical sleep-quality claim", "NO_GO"),
    ]
    df = pd.DataFrame(rows, columns=["phase", "decision"])
    df["basis"] = "mechanical mapping from research_question_results.csv; scores remain rule-derived " \
                  "research-prototype descriptors without expert or clinical validation"
    df.to_csv(os.path.join(out_dir, "go_no_go_decisions.csv"), index=False)
    return df
