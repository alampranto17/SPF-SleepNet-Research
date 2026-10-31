"""v2.2 post hoc analyses requested by the pre-submission review (31 Oct 2026).

Reported separately from the prespecified verdicts; nothing here changes a criterion.

1. Within-N3 comparators.  Is the detector's own continuous output (N3 logit,
   calibrated probability) as informative as the rule-derived scores?  Every
   candidate is evaluated with the same three references used for the factor
   scores: graded expert agreement, the night-time decline and expert R&K S4 vs S3
   (CAP).  A signal-only composite (rating without the detector-derived continuity
   factor) is added.
2. Calibration.  Expected calibration error (ECE) of the temperature-scaled
   probability with and without positive-class weighting (full_v2 vs full_nopw),
   and of a cross-fitted Platt scaling (slope + intercept) of the full model.
3. ICC confidence intervals (McGraw & Wong, absolute agreement, single measure)
   for the overnight-profile agreement and the ISRUC-II test-retest analyses.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import analysis as A
from . import scores as SC
from . import stats as ST
from .train import PRIMARY_VARIANT

CANDIDATES = ["rating_prior", "signal_composite", "delta_intensity", "sw_morphology",
              "spectral_dominance", "sw_time_frac", "n3_logit", "p_n3_calibrated"]


# ---------------------------------------------------------------------------
def ece(y, p, bins=10):
    """Expected calibration error with equal-width bins (as in fig_rq1_calibration)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def icc21_ci(x, y, alpha=0.05):
    """ICC(2,1) with the McGraw & Wong (1996) confidence interval."""
    X = np.column_stack([x, y]).astype(float)
    X = X[~np.isnan(X).any(1)]
    n, k = X.shape
    if n < 3:
        return np.nan, np.nan, np.nan, n
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    icc = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    if not np.isfinite(icc) or icc >= 1:
        return float(icc), np.nan, np.nan, n
    a = k * icc / (n * (1 - icc))
    b = 1 + k * icc * (n - 1) / (n * (1 - icc))
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    fl = stats.f.ppf(1 - alpha / 2, n - 1, v)
    fu = stats.f.ppf(1 - alpha / 2, v, n - 1)
    lo = n * (msr - fl * mse) / (fl * (k * msc + (k * n - k - n) * mse) + n * msr)
    hi = n * (fu * msr - mse) / (k * msc + (k * n - k - n) * mse + n * fu * msr)
    return float(icc), float(lo), float(hi), n


# ---------------------------------------------------------------------------
def _frame(corpus, run_dir):
    oof, ext = A.load_oof(corpus, run_dir, PRIMARY_VARIANT)
    S = A.compute_fold_scores(corpus, oof, run_dir)
    parts = [oof.join(S).assign(split="OOF")]
    if ext is not None and len(ext):
        Sx = A.compute_external_scores(corpus, ext)
        parts.append(ext.join(Sx).assign(split="external"))
    d = pd.concat(parts, ignore_index=True)
    w = SC.PRIOR_WEIGHTS
    sig = ["delta_intensity", "sw_morphology", "spectral_dominance"]
    d["signal_composite"] = sum(d[f] * w[f] for f in sig) / sum(w[f] for f in sig)
    d["n3_logit"] = d["logit"] if "logit" in d else np.nan
    d["p_n3_calibrated"] = d["prob_cal"] if "prob_cal" in d else d["prob"]
    d["sw_time_frac"] = corpus.feat.loc[d.idx.to_numpy(), "sw_time_frac"].to_numpy()
    d["half"] = A._half(d)
    return d


def within_n3_comparators(d):
    rows = []
    sets = {"all multi-rater": d}
    for c in d.cohort.unique():
        sets[c] = d[d.cohort == c]
    for name, g in sets.items():
        for detected_only in (False, True):
            gg = g[g.pred == 1] if detected_only else g
            for col in CANDIDATES:
                if col not in gg or gg[col].isna().all():
                    continue
                r = A.graded_agreement(gg, col)
                if r.get("n", 0) >= 5:
                    rows.append({"reference": "graded expert agreement (median within-participant Spearman)",
                                 "set": name, "epochs": "detected N3 only" if detected_only else "all N2/N3-voted",
                                 "candidate": col, "value": r["median_diff"], "n_participants": r["n"],
                                 "p": r["p"]})
    det = d[(d.pred == 1) & d.is_primary]
    for split, g in det.groupby("split"):
        for col in CANDIDATES:
            if col not in g or g[col].isna().all():
                continue
            pm = A._early_late(g, col)
            if pm is None or len(pm) < 5:
                continue
            e, l = pm["median"]["early"], pm["median"]["late"]
            w = ST.wilcoxon_paired(e, l, "greater")
            rows.append({"reference": "night-time decline (share of participants early > late)",
                         "set": split, "epochs": "detected N3", "candidate": col,
                         "value": float((e > l).mean()), "n_participants": int(len(pm)), "p": w["p"]})
    if "rk_code" in d:
        cap = d[d.rk_code.isin([3, 4])]
        for subset, g in (("all S3/S4", cap), ("S3/S4 and detected N3", cap[cap.pred == 1])):
            y4 = (g.rk_code == 4).astype(int).to_numpy()
            if len(np.unique(y4)) < 2:
                continue
            for col in CANDIDATES:
                if col not in g or g[col].isna().all():
                    continue
                m = g[col].notna().to_numpy()
                rows.append({"reference": "expert R&K S4 vs S3, AUROC (CAP)", "set": subset, "epochs": subset,
                             "candidate": col, "value": float(roc_auc_score(y4[m], g[col].to_numpy()[m])),
                             "n_participants": int(g.participant.nunique()), "p": np.nan})
    return pd.DataFrame(rows)


def calibration(corpus, run_dir):
    rows = []
    for variant in (PRIMARY_VARIANT, "full_nopw"):
        oof, ext = A.load_oof(corpus, run_dir, variant)
        if oof is None:
            continue
        sets = [("OOF pooled", A.scored(oof))]
        if ext is not None and len(ext):
            sets += [(f"external {c}", g) for c, g in A.scored(ext, primary_only=False).groupby("cohort")]
        for name, g in sets:
            p = g.prob_cal if "prob_cal" in g else g.prob
            rows.append({"variant": variant, "method": "temperature scaling (inner split)", "set": name,
                         "ece": ece(g.y, p), "brier": float(np.mean((p - g.y) ** 2)), "n": int(len(g))})
        if variant != PRIMARY_VARIANT or "logit" not in oof:
            continue
        # cross-fitted Platt scaling: each fold calibrated by a model fitted on the other folds' OOF
        o = A.scored(oof).copy()
        o["p_platt"] = np.nan
        for f in o.fold.unique():
            tr, te = o.fold != f, o.fold == f
            lr = LogisticRegression().fit(o.loc[tr, ["logit"]].to_numpy(), o.loc[tr, "y"].to_numpy())
            o.loc[te, "p_platt"] = lr.predict_proba(o.loc[te, ["logit"]].to_numpy())[:, 1]
        rows.append({"variant": variant, "method": "Platt scaling (cross-fitted on OOF)", "set": "OOF pooled",
                     "ece": ece(o.y, o.p_platt), "brier": float(np.mean((o.p_platt - o.y) ** 2)), "n": int(len(o))})
        if ext is not None and len(ext) and "logit" in ext:
            lr = LogisticRegression().fit(o[["logit"]].to_numpy(), o.y.to_numpy())
            for c, g in A.scored(ext, primary_only=False).groupby("cohort"):
                p = lr.predict_proba(g[["logit"]].to_numpy())[:, 1]
                rows.append({"variant": variant, "method": "Platt scaling (fitted on all OOF)", "set": f"external {c}",
                             "ece": ece(g.y, p), "brier": float(np.mean((p - g.y) ** 2)), "n": int(len(g))})
    return pd.DataFrame(rows)


def icc_intervals(out_dir):
    """CIs for rq7_profile_agreement (needs rq7_overnight_n3_eeg_profiles.csv)."""
    p = os.path.join(out_dir, "rq7_overnight_n3_eeg_profiles.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    prof = pd.read_csv(p)            # all recordings (ISRUC-II: both sessions), as in rq7_profile_agreement
    rows = []
    metrics = ["n3_minutes", "longest_episode_min", "n_episodes", "median_rating"]
    for (split, coh), g in [(("OOF", "all"), prof[prof.split == "OOF"])] + \
            [(("external", c), g) for c, g in prof[prof.split != "OOF"].groupby("cohort")]:
        for m in metrics:
            a, b = f"pred_{m}", f"expert_{m}"
            if a in g and b in g:
                icc, lo, hi, n = icc21_ci(g[a], g[b])
                rows.append({"analysis": "predicted vs expert", "set": f"{split}:{coh}", "metric": m,
                             "icc21": icc, "ci95_low": lo, "ci95_high": hi, "n": n})
    allp = pd.read_csv(p)
    tr = allp[allp.cohort == "ISRUC-II"].sort_values("rec")
    for col in ["pred_median_rating", "pred_n3_minutes", "expert_n3_minutes", "pred_median_delta_intensity",
                "pred_median_sw_morphology", "pred_median_spectral_dominance"]:
        if col not in tr:
            continue
        pairs = [g[col].to_numpy() for _, g in tr.groupby("participant") if len(g) == 2]
        if len(pairs) >= 3:
            X = np.array(pairs, float)
            icc, lo, hi, n = icc21_ci(X[:, 0], X[:, 1])
            rows.append({"analysis": "test-retest (ISRUC-II)", "set": "ISRUC-II", "metric": col,
                         "icc21": icc, "ci95_low": lo, "ci95_high": hi, "n": n})
    return pd.DataFrame(rows)


def run_all(corpus, run_dir, out_dir, log=print):
    log("[posthoc] within-N3 comparators (logit, calibrated probability, signal-only composite)")
    d = _frame(corpus, run_dir)
    within_n3_comparators(d).to_csv(os.path.join(out_dir, "posthoc_within_n3_comparators.csv"), index=False)
    log("[posthoc] calibration with / without class weighting, Platt scaling")
    calibration(corpus, run_dir).to_csv(os.path.join(out_dir, "posthoc_calibration.csv"), index=False)
    log("[posthoc] ICC confidence intervals")
    icc_intervals(out_dir).to_csv(os.path.join(out_dir, "posthoc_icc_ci.csv"), index=False)
    log("[posthoc] done")
