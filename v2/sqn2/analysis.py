"""RQ1-RQ7 analyses for SleepQualityNet v2.

Every analysis reads out-of-fold (OOF) predictions produced by train.Trainer:
a participant contributes predictions only from the fold model that never saw
them.  Success criteria are fixed in ``CRITERIA`` *before* any result exists
and are evaluated mechanically in ``rq_summary``; a criterion that is not met
is reported as such -- results are never re-tuned against the test folds.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

from . import scores as SC
from . import stats as ST
from .train import binary_metrics, load_run, VARIANTS, PRIMARY_VARIANT

CRITERIA = {
    "RQ1": "Pooled OOF N3 F1 >= 0.75 AND PR-AUC >= 0.85 (primary recordings, all scored "
           "epochs); full support additionally requires F1 >= 0.70 in every training cohort "
           "and on ISRUC-III.",
    "RQ2": "Each signal factor (delta intensity, slow-wave morphology, spectral dominance) "
           "and continuity is higher in early- than late-night detected N3 (participant-level "
           "one-sided Wilcoxon p < 0.05) and higher for expert N3 than expert N2 epochs.",
    "RQ3": "Rank stability under reasonable weight perturbation (Dirichlet, alpha = 20 x prior "
           "weights): 5th-percentile Spearman rho >= 0.90 vs the prior-weight rating, AND the "
           "early > late-night contrast is significant (p < 0.05) within each training cohort.",
    "RQ4": "Answered when every component effect is estimated on identical folds with a "
           "participant-level paired test and 95% CI (direction is a finding, not a criterion).",
    "RQ5": "Consistent physiological evidence if delta is the top band for BOTH band-stop "
           "perturbation and integrated-gradient band attribution in >= 70% of participants, "
           "and the SHAP-surrogate top feature group is delta-related with surrogate "
           "fidelity R^2 >= 0.70.",
    "RQ6": "Cross-fitted reliability score detects misclassified epochs with AUROC >= 0.75 "
           "AND F1 at 80% coverage exceeds F1 at 100% coverage.",
    "RQ7": "Predicted vs expert-hypnogram N3 minutes per night: ICC(2,1) >= 0.75 "
           "(secondary: episode count, longest episode).",
}


# ---------------------------------------------------------------------------
# Loading OOF predictions
# ---------------------------------------------------------------------------
def load_oof(corpus, run_dir, variant):
    files = sorted(glob.glob(os.path.join(run_dir, f"{variant}__fold[0-9]*.npz")))
    if not files:
        return None, None
    rows, ext = [], []
    for f in files:
        r = load_run(f)
        meta = json.loads(r["meta"])
        test_recs = set(meta["test_recs"])
        idx = r["idx"]
        rec = corpus.ep.rec.to_numpy()[idx]
        is_test = np.array([x in test_recs for x in rec])
        df = pd.DataFrame({"idx": idx, "prob": r["prob"], "fold": meta["fold"],
                           "threshold": float(r["threshold"])})
        for k in ("prob_cal", "mc_mean", "mc_entropy", "mc_mi", "mc_std", "maha", "logit"):
            if k in r:
                df[k] = r[k]
        rows.append(df[is_test])
        ext.append(df[~is_test])
    oof = pd.concat(rows, ignore_index=True)
    oof = oof.join(corpus.ep, on="idx")
    oof["pred"] = (oof.prob >= oof.threshold).astype(int)
    oof["y"] = (oof.label == 3).astype(int)
    exdf = pd.concat(ext, ignore_index=True) if ext else pd.DataFrame()
    if len(exdf):
        num = exdf.drop(columns=["fold"]).groupby("idx").mean()
        num = num.join(corpus.ep, on=num.index.name)
        num = num.reset_index().rename(columns={"index": "idx"})
        num["pred"] = (num.prob >= num.threshold).astype(int)
        num["y"] = (num.label == 3).astype(int)
        exdf = num
    return oof, exdf


def scored(df, primary_only=True):
    m = df.label >= 0
    if primary_only:
        m &= df.is_primary
    return df[m]


# ---------------------------------------------------------------------------
# RQ1
# ---------------------------------------------------------------------------
def participant_bootstrap(df, n_boot=1000, seed=0, metrics=("f1", "pr_auc", "roc_auc", "kappa")):
    """95 % CI by resampling PARTICIPANTS with replacement (epochs stay clustered)."""
    from sklearn.metrics import average_precision_score, cohen_kappa_score
    rng = np.random.default_rng(seed)
    groups = [(g.y.to_numpy(), g.prob.to_numpy(), g.pred.to_numpy())
              for _, g in df.groupby("participant")]
    vals = {m: [] for m in metrics}
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        y = np.concatenate([groups[i][0] for i in pick])
        if y.min() == y.max():
            continue
        p = np.concatenate([groups[i][1] for i in pick])
        pr = np.concatenate([groups[i][2] for i in pick])
        mt = {"f1": f1_score(y, pr, zero_division=0)}
        if "pr_auc" in metrics:
            mt["pr_auc"] = average_precision_score(y, p)
        if "roc_auc" in metrics:
            mt["roc_auc"] = roc_auc_score(y, p)
        if "kappa" in metrics:
            mt["kappa"] = cohen_kappa_score(y, pr)
        for m in metrics:
            vals[m].append(mt[m])
    return {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)
            for m, v in vals.items()}


def pooled_metrics(df):
    """Metrics with fold-specific thresholds already applied in df.pred."""
    y, p = df.y.to_numpy(), df.prob.to_numpy()
    m = binary_metrics(y, p, 0.5)
    from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, matthews_corrcoef
    pr = df.pred.to_numpy()
    m.update({"f1": f1_score(y, pr, zero_division=0),
              "precision": precision_score(y, pr, zero_division=0),
              "recall": recall_score(y, pr, zero_division=0),
              "kappa": cohen_kappa_score(y, pr) if len(np.unique(y)) > 1 else np.nan,
              "mcc": matthews_corrcoef(y, pr) if len(np.unique(y)) > 1 else np.nan})
    m.pop("threshold", None)
    m["n_participants"] = int(df.participant.nunique())
    return m


def rq1(corpus, run_dir, out_dir, n_boot=1000):
    oof, ext = load_oof(corpus, run_dir, PRIMARY_VARIANT)
    rows = []
    s = scored(oof)
    for name, g in [("pooled_all_cohorts", s)] + [(f"cohort={c}", g) for c, g in s.groupby("cohort")]:
        m = pooled_metrics(g)
        ci = participant_bootstrap(g, n_boot)
        m.update({f"{k}_ci95": v for k, v in ci.items()})
        m["subset"] = name
        rows.append(m)
    clean = s[~s.flag_any_v1.astype(bool)]
    m = pooled_metrics(clean)
    m["subset"] = "pooled_clean_epochs_only(v1 QC flags)"
    rows.append(m)
    for f, g in s.groupby("fold"):
        m = pooled_metrics(g)
        m["subset"] = f"fold={f}"
        rows.append(m)
    if ext is not None and len(ext):
        e = scored(ext, primary_only=False)
        m = pooled_metrics(e)
        m.update({f"{k}_ci95": v for k, v in participant_bootstrap(e, n_boot).items()})
        m["subset"] = "external_ISRUC-III(5-fold ensemble)"
        rows.append(m)
    # leave-one-cohort-out
    for f in sorted(glob.glob(os.path.join(run_dir, f"{PRIMARY_VARIANT}__foldLOCO_*.npz"))):
        r = load_run(f)
        df = pd.DataFrame({"idx": r["idx"], "prob": r["prob"]}).join(corpus.ep, on="idx")
        df["pred"] = (df.prob >= float(r["threshold"])).astype(int)
        df["y"] = (df.label == 3).astype(int)
        for c, g in scored(df, primary_only=False).groupby("cohort"):
            m = pooled_metrics(g)
            m["subset"] = f"{json.loads(r['meta'])['fold']} -> test {c}"
            rows.append(m)
    res = pd.DataFrame(rows)
    # inter-rater (ISRUC scorer 2) benchmark
    ir = []
    both = s[(s.label2 >= 0)]
    for c, g in both.groupby("cohort"):
        y1, y2 = (g.label == 3).astype(int), (g.label2 == 3).astype(int)
        from sklearn.metrics import cohen_kappa_score
        ir.append({"cohort": c, "n_epochs": len(g), "n_participants": g.participant.nunique(),
                   "scorer2_vs_scorer1_f1": f1_score(y1, y2), "scorer2_vs_scorer1_kappa":
                   cohen_kappa_score(y1, y2), "model_vs_scorer1_f1": f1_score(y1, g.pred),
                   "model_vs_scorer1_kappa": cohen_kappa_score(y1, g.pred),
                   "model_vs_scorer2_f1": f1_score(y2, g.pred),
                   "model_vs_scorer2_kappa": cohen_kappa_score(y2, g.pred)})
    if ext is not None and len(ext):
        e = scored(ext, primary_only=False)
        e = e[e.label2 >= 0]
        if len(e):
            from sklearn.metrics import cohen_kappa_score
            y1, y2 = (e.label == 3).astype(int), (e.label2 == 3).astype(int)
            ir.append({"cohort": "ISRUC-III(external)", "n_epochs": len(e),
                       "n_participants": e.participant.nunique(),
                       "scorer2_vs_scorer1_f1": f1_score(y1, y2),
                       "scorer2_vs_scorer1_kappa": cohen_kappa_score(y1, y2),
                       "model_vs_scorer1_f1": f1_score(y1, e.pred),
                       "model_vs_scorer1_kappa": cohen_kappa_score(y1, e.pred),
                       "model_vs_scorer2_f1": f1_score(y2, e.pred),
                       "model_vs_scorer2_kappa": cohen_kappa_score(y2, e.pred)})
    ir = pd.DataFrame(ir)
    res.to_csv(os.path.join(out_dir, "rq1_detection_metrics.csv"), index=False)
    ir.to_csv(os.path.join(out_dir, "rq1_interrater_benchmark.csv"), index=False)
    oof.to_parquet(os.path.join(out_dir, f"oof_predictions_{PRIMARY_VARIANT}.parquet"))
    if ext is not None and len(ext):
        ext.to_parquet(os.path.join(out_dir, "external_isruc3_predictions.parquet"))
    return res, ir, oof, ext


# ---------------------------------------------------------------------------
# RQ2 / RQ3 factor scores
# ---------------------------------------------------------------------------
def compute_fold_scores(corpus, oof, run_dir, variant=PRIMARY_VARIANT):
    """Signal factor scores for every OOF epoch with fold-specific references."""
    feat = corpus.feat
    parts = []
    for f in sorted(oof.fold.unique()):
        r = load_run(os.path.join(run_dir, f"{variant}__fold{f}.npz"))
        meta = json.loads(r["meta"])
        tr = corpus.ep.index[corpus.ep.rec.isin(meta["train_recs"]) & (corpus.ep.label == 3)]
        ref_df = feat.loc[tr].copy()
        ref_df["montage"] = corpus.ep.loc[tr, "montage"].to_numpy()
        ref = SC.FactorReference().fit(ref_df)
        g = oof[oof.fold == f]
        X = feat.loc[g.idx.to_numpy()].copy()
        X.index = g.index
        X["montage"] = g.montage.to_numpy()
        sc = SC.factor_scores(X, ref)
        parts.append(sc)
    S = pd.concat(parts).loc[oof.index]
    S["n3_continuity"] = SC.continuity_score(oof.pred.to_numpy(), oof.rec.to_numpy(),
                                             oof.epoch_idx.to_numpy())
    S["n3_continuity_expert"] = SC.continuity_score(oof.y.to_numpy(), oof.rec.to_numpy(),
                                                    oof.epoch_idx.to_numpy())
    S["rating_prior"] = SC.overall_rating(S, SC.PRIOR_WEIGHTS)
    S["rating_equal"] = SC.overall_rating(S, SC.EQUAL_WEIGHTS)
    return S


def _half(df):
    n = df.groupby("rec").epoch_idx.transform("max") + 1
    return np.where(df.epoch_idx < n / 2, "early", "late")


def rq2(oof, S, corpus, out_dir):
    d = oof.join(S)
    d["half"] = _half(d)
    facs = SC.FACTORS + ["rating_prior"]
    rows = []
    det = d[(d.pred == 1) & d.is_primary]
    # T2 homeostatic decline within participant (detected N3)
    for fac in facs:
        pm = det.groupby(["participant", "half"])[fac].agg(["median", "size"]).unstack()
        ok = (pm["size"]["early"] >= 10) & (pm["size"]["late"] >= 10) if "late" in pm["size"] else []
        pm = pm[ok]
        w = ST.wilcoxon_paired(pm["median"]["early"], pm["median"]["late"], "greater")
        ci = ST.boot_ci_median_diff(pm["median"]["early"] - pm["median"]["late"])
        rows.append({"test": "early>late (detected N3)", "factor": fac, **w,
                     "ci95_low": ci[0], "ci95_high": ci[1], "cohort": "all"})
        for c in det.cohort.unique():
            sub = det[det.cohort == c]
            pm = sub.groupby(["participant", "half"])[fac].agg(["median", "size"]).unstack()
            if "late" not in pm["size"]:
                continue
            pm = pm[(pm["size"]["early"] >= 10) & (pm["size"]["late"] >= 10)]
            w = ST.wilcoxon_paired(pm["median"]["early"], pm["median"]["late"], "greater")
            rows.append({"test": "early>late (detected N3)", "factor": fac, **w, "cohort": c})
    # T1 known groups: expert N3 vs expert N2 (continuity uses expert sequence)
    ex = d[d.is_primary & d.label.isin([2, 3])].copy()
    ex["n3_continuity"] = ex["n3_continuity_expert"]
    for fac in SC.FACTORS:
        pm = ex.groupby(["participant", "label"])[fac].median().unstack()
        if 3 not in pm or 2 not in pm:
            continue
        w = ST.wilcoxon_paired(pm[3], pm[2], "greater")
        rows.append({"test": "expertN3>expertN2", "factor": fac, **w, "cohort": "all"})
    # T4 expert-ambiguity continuum (ISRUC scorer 1 vs scorer 2): epochs where
    # the two experts disagree (N2 vs N3) should lie BETWEEN consensus N2 and
    # consensus N3 -- an independent human reference for the factor scores.
    amb = d[d.is_primary & d.label.isin([2, 3]) & d.label2.isin([2, 3])].copy()
    if len(amb):
        amb["grp"] = np.select([(amb.label == 2) & (amb.label2 == 2),
                                (amb.label == 3) & (amb.label2 == 3)],
                               ["consensus_N2", "consensus_N3"], "disagree_N2_N3")
        for fac in SC.FACTORS[:3] + ["rating_prior"]:
            pm = amb.groupby(["participant", "grp"])[fac].median().unstack()
            if not {"consensus_N2", "consensus_N3", "disagree_N2_N3"} <= set(pm.columns):
                continue
            w1 = ST.wilcoxon_paired(pm["disagree_N2_N3"], pm["consensus_N2"], "greater")
            w2 = ST.wilcoxon_paired(pm["consensus_N3"], pm["disagree_N2_N3"], "greater")
            rows.append({"test": "ambiguous>consensusN2 (scorer2)", "factor": fac, **w1, "cohort": "ISRUC"})
            rows.append({"test": "consensusN3>ambiguous (scorer2)", "factor": fac, **w2, "cohort": "ISRUC"})
    # T3 age (only if metadata exists)
    pmeta = corpus.participant_meta.get("participants", {})
    if pmeta:
        pr = det.groupby(["participant", "cohort"])[facs].median().reset_index()
        pr["age"] = pr.participant.map(lambda p: pmeta.get(p, {}).get("age", np.nan))
        for c, g in [("all", pr)] + list(pr.groupby("cohort")):
            for fac in facs:
                rho, p = ST.spearman(g[fac], g["age"])
                rows.append({"test": "spearman_vs_age (expect <0)", "factor": fac,
                             "n": int(g["age"].notna().sum()), "median_diff": rho, "p": p,
                             "cohort": c})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "rq2_factor_validity.csv"), index=False)
    desc = det.groupby("cohort")[facs].describe().T
    desc.to_csv(os.path.join(out_dir, "rq2_factor_descriptives_detected_n3.csv"))
    return res


def rq3(oof, S, out_dir, n_draws=1000, seed=0):
    d = oof.join(S)
    det = d[(d.pred == 1) & d.is_primary].copy()
    rng = np.random.default_rng(seed)
    F = det[SC.FACTORS].to_numpy()
    prior = np.array([SC.PRIOR_WEIGHTS[k] for k in SC.FACTORS])
    base = F @ prior
    out = {}
    for name, alpha in [("reasonable(alpha=20*prior)", 20 * prior), ("extreme(alpha=1)", np.ones(4))]:
        W = rng.dirichlet(alpha, n_draws)
        R = F @ W.T
        from scipy.stats import rankdata
        rb = rankdata(base)
        rhos = [np.corrcoef(rb, rankdata(R[:, j]))[0, 1] for j in range(n_draws)]
        out[name] = {"median_rho": float(np.median(rhos)), "p05_rho": float(np.percentile(rhos, 5)),
                     "min_rho": float(np.min(rhos))}
    # alternative weightings
    pca_w = SC.pca_weights(det)
    det["rating_pca"] = SC.overall_rating(det, pca_w)
    alt = {"prior_vs_equal_rho": ST.spearman(det.rating_prior, det.rating_equal)[0],
           "prior_vs_pca_rho": ST.spearman(det.rating_prior, det.rating_pca)[0],
           "pca_weights_pooled": pca_w}
    # factor redundancy
    red = det[SC.FACTORS].corr(method="spearman")
    red.to_csv(os.path.join(out_dir, "rq3_factor_redundancy_spearman.csv"))
    # missing-factor behaviour
    miss = {}
    for k in SC.FACTORS:
        w = {kk: v for kk, v in SC.PRIOR_WEIGHTS.items() if kk != k}
        w[k] = 0.0
        miss[f"without_{k}"] = ST.spearman(det.rating_prior, SC.overall_rating(det, w))[0]
    # split-half (odd/even epochs) participant reliability
    det = det.sort_values(["rec", "epoch_idx"])
    det["parity"] = det.groupby("rec").cumcount() % 2
    sh = det[det.is_primary].groupby(["participant", "parity"]).rating_prior.median().unstack()
    sh = sh.dropna()
    r_sh = ST.spearman(sh[0], sh[1])[0]
    sb = 2 * r_sh / (1 + r_sh) if r_sh == r_sh else np.nan
    # night-to-night (Sleep-EDF night 1 vs night 2; inference only for night 2)
    dn = d[(d.pred == 1) & (d.cohort == "Sleep-EDF")]
    nn = dn.groupby(["participant", "is_primary"]).rating_prior.median().unstack()
    nn = nn.dropna() if True in nn and False in nn else pd.DataFrame()
    n2n = {"n_participants": int(len(nn)),
           "icc21": ST.icc(nn[True], nn[False]) if len(nn) else np.nan,
           "icc31": ST.icc(nn[True], nn[False], "3,1") if len(nn) else np.nan}
    # participant-level bootstrap CI of the cohort mean rating
    pr = det.groupby(["participant", "cohort"]).rating_prior.median().reset_index()
    cis = {}
    for c, g in pr.groupby("cohort"):
        v = g.rating_prior.to_numpy()
        bs = [np.mean(rng.choice(v, len(v))) for _ in range(2000)]
        cis[c] = {"mean_participant_median": float(v.mean()),
                  "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res = {"weight_sensitivity": out, "alternative_weightings": alt,
           "missing_factor_rho": miss, "split_half_spearman_brown": sb,
           "night_to_night": n2n, "cohort_rating_ci": cis,
           "learned_latent_rating": "NOT IMPLEMENTED: no independent target exists; a "
                                    "learned latent rating would be circular/unidentifiable."}
    with open(os.path.join(out_dir, "rq3_overall_rating_stability.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


# ---------------------------------------------------------------------------
# RQ4 component analysis
# ---------------------------------------------------------------------------
COMPARISONS = [
    ("handcrafted feature fusion", "cnn_feat", "cnn"),
    ("positive-class weighting", "cnn_pw", "cnn"),
    ("feature fusion | weighting", "cnn_feat_pw", "cnn_pw"),
    ("temporal attention (vs mean pooling)", "cnn_feat_pw", "cnn_feat_pw_noattn"),
    ("adjacent-epoch context", "full_v2", "cnn_feat_pw"),
    ("CNN backbone over features-only GBM", "cnn_feat_pw", "feat_gbm"),
]


def _participant_metrics(df):
    rows = []
    for p, g in df.groupby("participant"):
        y = g.y.to_numpy()
        r = {"participant": p, "cohort": g.cohort.iloc[0],
             "f1": f1_score(y, g.pred, zero_division=0) if y.sum() else np.nan,
             "n3_min_pred": g.pred.sum() / 2, "n3_min_true": y.sum() / 2}
        try:
            from sklearn.metrics import average_precision_score
            r["pr_auc"] = average_precision_score(y, g.prob) if 0 < y.sum() < len(y) else np.nan
        except Exception:
            r["pr_auc"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("participant")


def rq4(corpus, run_dir, out_dir):
    pooled, per_part, params = {}, {}, {}
    for v in list(VARIANTS) + ["feat_gbm"]:
        oof, _ = load_oof(corpus, run_dir, v)
        if oof is None:
            continue
        s = scored(oof)
        pooled[v] = pooled_metrics(s)
        per_part[v] = _participant_metrics(s)
        f0 = sorted(glob.glob(os.path.join(run_dir, f"{v}__fold[0-9]*.npz")))[0]
        r = load_run(f0)
        params[v] = int(r.get("n_params", 0)) if "n_params" in r else 0
        pooled[v]["n_params"] = params[v]
        pooled[v]["n3_minutes_icc21"] = ST.icc(per_part[v].n3_min_pred, per_part[v].n3_min_true)
        if "feat_clip_frac" in r:
            pooled[v]["feature_values_clipped_frac"] = float(r["feat_clip_frac"])
    tab = pd.DataFrame(pooled).T
    tab.to_csv(os.path.join(out_dir, "rq4_variant_pooled_metrics.csv"))
    rows = []
    for name, a, b in COMPARISONS:
        if a not in per_part or b not in per_part:
            continue
        for metric in ("f1", "pr_auc"):
            j = per_part[a][[metric]].join(per_part[b][[metric]], lsuffix="_a", rsuffix="_b").dropna()
            w = ST.wilcoxon_paired(j[f"{metric}_a"], j[f"{metric}_b"])
            ci = ST.boot_ci_median_diff(j[f"{metric}_a"] - j[f"{metric}_b"])
            rows.append({"component": name, "with": a, "without": b, "metric": metric,
                         "pooled_with": pooled[a].get(metric if metric != "f1" else "f1"),
                         "pooled_without": pooled[b].get(metric),
                         **w, "ci95_low": ci[0], "ci95_high": ci[1],
                         "params_with": params.get(a), "params_without": params.get(b)})
    comp = pd.DataFrame(rows)
    if len(comp):
        comp["p_holm"] = ST.holm(comp.p.to_numpy())
    comp.to_csv(os.path.join(out_dir, "rq4_component_effects.csv"), index=False)
    return tab, comp


# ---------------------------------------------------------------------------
# RQ6 reliability
# ---------------------------------------------------------------------------
RELIABILITY_FEATURES = ["confidence", "mc_entropy", "mc_mi", "maha", "qc_score_v1",
                        "qc_hf_ratio", "qc_n_clipped", "flag_any_v1"]


def rq6(oof, out_dir):
    d = scored(oof).copy()
    p = d.prob_cal if "prob_cal" in d else d.prob
    d["confidence"] = np.maximum(p, 1 - p)
    d["error"] = (d.pred != d.y).astype(int)
    comp = []
    for k in RELIABILITY_FEATURES:
        x = d[k].astype(float).to_numpy()
        sign = -1 if k in ("confidence", "qc_score_v1") else 1   # higher = more error-prone
        auc = roc_auc_score(d.error, sign * x) if d.error.nunique() > 1 else np.nan
        comp.append({"component": k, "group": ("model_confidence" if k in ("confidence", "mc_entropy")
                                               else "epistemic/OOD" if k in ("mc_mi", "maha")
                                               else "signal_quality"),
                     "error_detection_auroc": auc})
    comp = pd.DataFrame(comp)
    # cross-fitted combined score (fit on other folds' participants only)
    X = d[RELIABILITY_FEATURES].astype(float).to_numpy()
    X = np.column_stack([X, np.log1p(np.clip(d.maha.to_numpy(), 0, None))])
    rel = np.zeros(len(d))
    for f in d.fold.unique():
        tr, te = d.fold.to_numpy() != f, d.fold.to_numpy() == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        yc = 1 - d.error.to_numpy()[tr]
        if np.unique(yc).size < 2:          # degenerate fold (no errors): constant score
            rel[te] = float(yc.mean())
            continue
        lr = LogisticRegression(max_iter=2000, class_weight=None)
        lr.fit((X[tr] - mu) / sd, yc)
        rel[te] = lr.predict_proba((X[te] - mu) / sd)[:, 1]
    d["reliability"] = 100 * rel
    d["reliability_cat"] = pd.cut(d.reliability, [-1, 80, 95, 101], labels=["low", "medium", "high"])
    auc = roc_auc_score(d.error, -d.reliability)
    # risk-coverage
    rc = []
    order = np.argsort(-d.reliability.to_numpy())
    for cov in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
        keep = order[: int(round(cov * len(d)))]
        g = d.iloc[keep]
        rc.append({"coverage": cov, "f1": f1_score(g.y, g.pred, zero_division=0),
                   "accuracy": float((g.y == g.pred).mean()), "n": len(g)})
    rc = pd.DataFrame(rc)
    cat = d.groupby("reliability_cat", observed=False).agg(
        n=("error", "size"), error_rate=("error", "mean")).reset_index()
    comp.loc[len(comp)] = {"component": "COMBINED (cross-fitted logistic)", "group": "combined",
                           "error_detection_auroc": auc}
    # independent reference: do low-reliability epochs coincide with epochs on
    # which two human experts disagree about N3 (ISRUC scorer 1 vs scorer 2)?
    both = d[d.label2 >= 0]
    if len(both) and both.label.ne(both.label2).any():
        dis = ((both.label == 3) != (both.label2 == 3)).astype(int)
        if dis.nunique() > 1:
            comp.loc[len(comp)] = {"component": "COMBINED vs expert N3 disagreement",
                                   "group": "independent_reference",
                                   "error_detection_auroc": roc_auc_score(dis, -both.reliability)}
            comp.loc[len(comp)] = {"component": "mc_entropy vs expert N3 disagreement",
                                   "group": "independent_reference",
                                   "error_detection_auroc": roc_auc_score(dis, both.mc_entropy)}
    comp.to_csv(os.path.join(out_dir, "rq6_reliability_components.csv"), index=False)
    rc.to_csv(os.path.join(out_dir, "rq6_risk_coverage.csv"), index=False)
    cat.to_csv(os.path.join(out_dir, "rq6_reliability_categories.csv"), index=False)
    return comp, rc, cat, d[["idx", "reliability", "reliability_cat"]]


# ---------------------------------------------------------------------------
# RQ7 episodes and Overnight N3 EEG Profile
# ---------------------------------------------------------------------------
def rq7(oof, S, rel_df, out_dir):
    d = oof.join(S)
    if rel_df is not None:
        d = d.merge(rel_df, on="idx", how="left")
    prows, erows = [], []
    for rec, g in d.groupby("rec"):
        g = g.sort_values("epoch_idx")
        rated = np.where(g.pred == 1, g.rating_prior, np.nan)
        pp, eps = SC.night_profile(g.pred.to_numpy(), rated, g.n3_continuity.to_numpy(),
                                   g.get("reliability", pd.Series(np.nan, index=g.index)).to_numpy(),
                                   rel_high=95)
        # expert-hypnogram profile; rating with continuity from expert sequence
        Sx = g[SC.FACTORS].copy()
        Sx["n3_continuity"] = g.n3_continuity_expert
        rx = np.where(g.y == 1, SC.overall_rating(Sx), np.nan)
        pe, _ = SC.night_profile(g.y.to_numpy(), rx, g.n3_continuity_expert.to_numpy())
        base = {"rec": rec, "participant": g.participant.iloc[0], "cohort": g.cohort.iloc[0],
                "is_primary": bool(g.is_primary.iloc[0])}
        for k in SC.FACTORS[:3]:
            pp[f"median_{k}"] = float(g[k][g.pred == 1].median()) if (g.pred == 1).any() else np.nan
            pe[f"median_{k}"] = float(g[k][g.y == 1].median()) if (g.y == 1).any() else np.nan
        prows.append({**base, **{f"pred_{k}": v for k, v in pp.items()},
                      **{f"expert_{k}": v for k, v in pe.items()}})
        for (s, e, n3, intr) in eps:
            seg = g.iloc[s:e]
            r = np.where(seg.pred == 1, seg.rating_prior, np.nan)
            erows.append({**base, "start_epoch": int(seg.epoch_idx.iloc[0]),
                          "duration_min": n3 * 0.5, "interruptions": intr,
                          "median_rating": float(np.nanmedian(r)),
                          "rating_iqr": float(np.nanpercentile(r, 75) - np.nanpercentile(r, 25)),
                          **{f"median_{k}": float(seg[k][seg.pred == 1].median())
                             for k in SC.FACTORS}})
    prof, epi = pd.DataFrame(prows), pd.DataFrame(erows)
    prof.to_csv(os.path.join(out_dir, "rq7_overnight_n3_eeg_profiles.csv"), index=False)
    epi.to_csv(os.path.join(out_dir, "rq7_n3_episodes.csv"), index=False)
    pr = prof[prof.is_primary]
    agree = []
    for k in ("n3_minutes", "n_episodes", "longest_episode_min", "median_episode_min",
              "n3_first_half_fraction", "median_rating"):
        a, b = pr[f"pred_{k}"], pr[f"expert_{k}"]
        agree.append({"metric": k, "icc21": ST.icc(a, b), "icc31": ST.icc(a, b, "3,1"),
                      "spearman": ST.spearman(a, b)[0], **ST.bland_altman(a, b)})
    agree = pd.DataFrame(agree)
    # night-to-night stability of predicted profile (Sleep-EDF)
    # night-to-night reproducibility (Sleep-EDF night 1 vs night 2, both nights in
    # the participant's test fold; night 2 never used for training).  Tests whether
    # factor ratings are more reproducible than conventional N3 minutes.
    sedf = prof[prof.cohort == "Sleep-EDF"]
    for m in ("pred_n3_minutes", "expert_n3_minutes", "pred_median_rating",
              "pred_median_delta_intensity", "pred_median_sw_morphology",
              "pred_median_spectral_dominance", "pred_n_episodes", "pred_longest_episode_min"):
        if m not in sedf:
            continue
        piv = sedf.pivot_table(index="participant", columns="is_primary", values=m)
        if True in piv and False in piv:
            piv = piv.dropna()
            agree.loc[len(agree)] = {"metric": f"night1_vs_night2::{m}",
                                     "icc21": ST.icc(piv[True], piv[False]),
                                     "icc31": ST.icc(piv[True], piv[False], "3,1"),
                                     "spearman": ST.spearman(piv[True], piv[False])[0],
                                     **ST.bland_altman(piv[True], piv[False])}
    agree.to_csv(os.path.join(out_dir, "rq7_profile_agreement.csv"), index=False)
    return prof, epi, agree


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _status(ok_full, ok_partial=None):
    if ok_full:
        return "SUPPORTED"
    if ok_partial:
        return "PARTIALLY_SUPPORTED"
    return "NOT_SUPPORTED"


def rq_summary(out_dir, r1=None, r2=None, r3=None, r4=None, r5=None, r6=None, r7=None):
    rows = []
    if r1 is not None:
        t = r1.set_index("subset")
        po = t.loc["pooled_all_cohorts"]
        coh = t[t.index.str.startswith("cohort=")]
        ext = t.loc[t.index.str.startswith("external")]
        base_ok = po.f1 >= 0.75 and po.pr_auc >= 0.85
        full = base_ok and (coh.f1 >= 0.70).all() and (len(ext) == 0 or (ext.f1 >= 0.70).all())
        obs = (f"pooled F1={po.f1:.3f} {tuple(round(x, 3) for x in po.f1_ci95)}, "
               f"PR-AUC={po.pr_auc:.3f}; " + "; ".join(f"{i}: F1={r.f1:.3f}" for i, r in coh.iterrows())
               + ("; " + "; ".join(f"ISRUC-III F1={r.f1:.3f}" for _, r in ext.iterrows()) if len(ext) else ""))
        rows.append(("RQ1", CRITERIA["RQ1"], obs, _status(full, base_ok)))
    if r2 is not None:
        t = r2[(r2.test == "early>late (detected N3)") & (r2.cohort == "all") &
               r2.factor.isin(SC.FACTORS)]
        k = r2[(r2.test == "expertN3>expertN2")]
        ok_t = (t.p < 0.05).sum()
        ok_k = (k.p < 0.05).sum()
        obs = "; ".join(f"{r.factor}: early-late={r.median_diff:.1f} (p={r.p:.2g})" for _, r in t.iterrows())
        full = ok_t == 4 and ok_k == 4
        rows.append(("RQ2", CRITERIA["RQ2"], obs + f"; N3>N2 significant for {ok_k}/4",
                     _status(full, ok_t >= 2)))
    if r3 is not None and r2 is not None:
        ws = r3["weight_sensitivity"]["reasonable(alpha=20*prior)"]["p05_rho"]
        cc = r2[(r2.factor == "rating_prior") & (r2.test == "early>late (detected N3)") &
                (r2.cohort != "all")]
        coh_ok = len(cc) > 0 and (cc.p < 0.05).all()
        obs = (f"p05 rho={ws:.3f}; night-to-night ICC(2,1)={r3['night_to_night']['icc21']:.3f}; "
               f"split-half SB={r3['split_half_spearman_brown']:.3f}; cohort early>late p: "
               + ", ".join(f"{r.cohort}={r.p:.2g}" for _, r in cc.iterrows()))
        rows.append(("RQ3", CRITERIA["RQ3"], obs, _status(ws >= 0.90 and coh_ok, ws >= 0.90 or coh_ok)))
    if r4 is not None:
        comp = r4[1]
        f1c = comp[comp.metric == "f1"]
        obs = "; ".join(f"{r.component}: dF1={r.median_diff:+.3f} [{r.ci95_low:+.3f},{r.ci95_high:+.3f}] "
                        f"p_holm={r.p_holm:.2g}" for _, r in f1c.iterrows())
        rows.append(("RQ4", CRITERIA["RQ4"], obs,
                     "ANSWERED" if len(f1c) == len(COMPARISONS) else "PARTIALLY_ANSWERED"))
    if r5 is not None:
        rows.append(("RQ5", CRITERIA["RQ5"], r5["observed"], r5["status"]))
    if r6 is not None:
        comp, rc = r6[0], r6[1]
        auc = float(comp[comp.component.str.startswith("COMBINED")].error_detection_auroc.iloc[0])
        f100 = float(rc[rc.coverage == 1.0].f1.iloc[0])
        f80 = float(rc[rc.coverage == 0.8].f1.iloc[0])
        rows.append(("RQ6", CRITERIA["RQ6"], f"AUROC={auc:.3f}; F1@100%={f100:.3f}, F1@80%={f80:.3f}",
                     _status(auc >= 0.75 and f80 > f100, auc >= 0.75 or f80 > f100)))
    if r7 is not None:
        a = r7[2].set_index("metric")
        icc_m = a.loc["n3_minutes", "icc21"]
        obs = "; ".join(f"{m}: ICC={a.loc[m, 'icc21']:.3f}" for m in a.index)
        rows.append(("RQ7", CRITERIA["RQ7"], obs,
                     _status(icc_m >= 0.75, icc_m >= 0.60)))
    df = pd.DataFrame(rows, columns=["research_question", "prespecified_criterion",
                                     "observed", "status"])
    df.to_csv(os.path.join(out_dir, "research_question_results.csv"), index=False)
    return df
