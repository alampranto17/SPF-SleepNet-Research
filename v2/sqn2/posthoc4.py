"""v2.5 post hoc checks requested by the peer review (no change to any prespecified verdict).

M1  DOD alignment sensitivity.  The DOD hypnograms were shifted to the signal with an offset
    chosen by maximising label agreement (a label-based step).  This module re-scores the
    SAME saved primary-model predictions against the DOD labels shifted by a range of
    offsets around the chosen one, and against the unshifted (lag 0) hypnogram.  It shows
    how much the DOD result depends on the offset.  It does NOT provide a label-free
    alignment: a label-free offset needs the EDF start time and the hypnogram start time
    from the original metadata, which the cache does not store.

M7  Delta-feature control.  The detector receives 19 handcrafted features, 10 of which are
    the delta-band group that also defines the delta-intensity factor.  The variant
    `full_nodeltafeat` receives the 9 remaining features.  If detection holds without the
    delta group, the detector's accuracy does not rely on the features that define a factor.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import analysis as A
from . import features as FT
from . import train as T
from .data import consensus_labels, shift_hypnograms

DOD_COHORTS = ("DOD-H", "DOD-O")
REL_SHIFTS = list(range(-30, 31, 3))      # epochs relative to the label-chosen offset
NODELTA = "full_nodeltafeat"
FULL = T.PRIMARY_VARIANT


def _lag_of(corpus, rec):
    if "scorer_lag_epochs" not in corpus.rec.columns:
        raise RuntimeError("scorer_lag_epochs missing from the corpus recording table")
    v = corpus.rec.loc[rec, "scorer_lag_epochs"]
    return int(v) if pd.notna(v) else 0


def _labels_shifted(corpus, rec, delta):
    """Consensus labels when the scorer hypnograms are moved by `delta` epochs from the
    offset used in the cache (delta = 0 reproduces the cached labels)."""
    multi = corpus.labels_multi[rec]
    n = multi.shape[0]
    cols = [multi[:, j] for j in range(multi.shape[1])]
    return consensus_labels(shift_hypnograms(cols, delta, n))


def dod_alignment_sensitivity(corpus, run_dir, out_dir, log=print):
    oof, ext = A.load_oof(corpus, run_dir, FULL)
    if ext is None or not len(ext):
        raise RuntimeError("no external predictions for the primary model")
    dod = ext[ext.cohort.isin(DOD_COHORTS)].copy()
    pos = corpus.ep.groupby("rec").cumcount().to_numpy()
    dod["pos"] = pos[dod.idx.to_numpy()]

    # Sanity check: rebuilding the labels from the cached multi-rater matrix must
    # reproduce the labels used in training; otherwise the shifts would be wrong.
    mism = 0
    for rec in dod.rec.unique():
        lab = _labels_shifted(corpus, rec, 0)
        ref = corpus.ep.label.to_numpy()[corpus.ep.rec.to_numpy() == rec]
        mism += int((lab != ref).sum())
    log(f"[posthoc4] M1 consistency: label rebuild mismatches = {mism} (expected 0)")

    lag = {r: _lag_of(corpus, r) for r in dod.rec.unique()}
    scenarios = [(f"shift{d:+d}", {r: d for r in lag}) for d in REL_SHIFTS]
    scenarios.append(("lag0_unshifted", {r: -lag[r] for r in lag}))   # raw hypnogram

    frames = []
    for name, delta_of in scenarios:
        parts = []
        for rec, g in dod.groupby("rec"):
            lab = _labels_shifted(corpus, rec, delta_of[rec])
            y_lab = lab[g.pos.to_numpy()]
            keep = y_lab >= 0
            h = g[keep].copy()
            h["y"] = (y_lab[keep] == 3).astype(int)
            parts.append(h)
        f = pd.concat(parts, ignore_index=True)
        f["scenario"] = name
        frames.append(f)
    allf = pd.concat(frames, ignore_index=True)

    rows = []
    for (name, coh), g in allf.groupby(["scenario", "cohort"]):
        m = A.pooled_metrics(g)
        rows.append({"scenario": name, "cohort": coh, "n_epochs": len(g),
                     "n_participants": m["n_participants"], "positive_rate": float(g.y.mean()),
                     **{k: m[k] for k in ("f1", "precision", "recall", "pr_auc", "roc_auc", "kappa")}})
    res = pd.DataFrame(rows)
    res["rel_shift_epochs"] = res.scenario.map(
        lambda s: np.nan if s == "lag0_unshifted" else int(s.replace("shift", "")))
    res.to_csv(os.path.join(out_dir, "posthoc4_m1_alignment_sensitivity.csv"), index=False)

    lags = (corpus.rec.loc[list(lag)]
            .reset_index(drop=True)[["recording", "cohort"]].assign(lag_epochs=list(lag.values())))
    if "scorer_vs_h5_agreement" in corpus.rec.columns:
        lags["agreement"] = corpus.rec.loc[list(lag), "scorer_vs_h5_agreement"].to_numpy()
    lags.to_csv(os.path.join(out_dir, "posthoc4_m1_label_offsets.csv"), index=False)
    log("[posthoc4] M1 written: alignment sensitivity and label offsets")
    return res


def _paired_auc_ci(m, n_boot, rng):
    """Participant-level paired bootstrap of the AUC difference (full - nodelta)."""
    groups = {p: g for p, g in m.groupby("participant")}
    parts = list(groups)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(parts), len(parts), replace=True)
        g = pd.concat([groups[parts[i]] for i in pick], ignore_index=True)
        if g.y.nunique() < 2:
            continue
        out.append(roc_auc_score(g.y, g.p_full) - roc_auc_score(g.y, g.p_nd))
    if not out:
        return np.nan, np.nan
    return tuple(np.percentile(out, [2.5, 97.5]))


def nodelta_comparison(corpus, run_dir, out_dir, n_boot=1000, log=print):
    from .train import feature_columns
    full_oof, full_ext = A.load_oof(corpus, run_dir, FULL)
    nd_oof, nd_ext = A.load_oof(corpus, run_dir, NODELTA)
    if nd_oof is None:
        raise RuntimeError(f"{NODELTA} has no saved runs; train it first")
    rng = np.random.default_rng(0)
    rows = []
    for split, a, b in (("OOF", full_oof, nd_oof), ("external", full_ext, nd_ext)):
        a = A.scored(a)[["idx", "y", "prob", "pred", "participant", "cohort"]]
        b = A.scored(b)[["idx", "prob", "pred"]]
        m = a.merge(b, on="idx", suffixes=("", "_nd"))
        log(f"[posthoc4] M7 {split}: {len(m)} epochs paired (full {len(a)}, nodelta {len(b)})")
        for coh, g in m.groupby("cohort"):
            if g.y.nunique() < 2:
                continue
            g = g.rename(columns={"prob": "p_full", "prob_nd": "p_nd"})
            mf = A.pooled_metrics(g.assign(prob=g.p_full))
            mn = A.pooled_metrics(g.assign(prob=g.p_nd, pred=g.pred_nd))
            lo, hi = _paired_auc_ci(g, n_boot, rng)
            rows.append({"split": split, "cohort": coh, "n_epochs": len(g),
                         "n_participants": mf["n_participants"],
                         "roc_auc_full": mf["roc_auc"], "roc_auc_nodelta": mn["roc_auc"],
                         "roc_auc_diff": mf["roc_auc"] - mn["roc_auc"],
                         "roc_auc_diff_ci_lo": lo, "roc_auc_diff_ci_hi": hi,
                         "f1_full": mf["f1"], "f1_nodelta": mn["f1"],
                         "pr_auc_full": mf["pr_auc"], "pr_auc_nodelta": mn["pr_auc"]})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "posthoc4_m7_nodelta_vs_full.csv"), index=False)

    nf = [FT.DETECTION_FEATURES[i] for i in feature_columns({"feat_subset": "nondelta"})]
    params = pd.DataFrame([
        {"variant": FULL, "n_handcrafted": len(FT.DETECTION_FEATURES)},
        {"variant": NODELTA, "n_handcrafted": len(nf), "features": ";".join(nf)}])
    params.to_csv(os.path.join(out_dir, "posthoc4_m7_feature_sets.csv"), index=False)
    log("[posthoc4] M7 written: nodelta vs full comparison")
    return res


def run_all(corpus, run_dir, out_dir, log=print):
    for name, fn in (("M1", dod_alignment_sensitivity), ("M7", nodelta_comparison)):
        try:
            fn(corpus, run_dir, out_dir, log=log)
        except Exception as exc:          # one failing check must not hide the other
            import traceback
            log(f"[posthoc4] {name} FAILED: {type(exc).__name__}: {exc}")
            log(traceback.format_exc()[-1500:])
