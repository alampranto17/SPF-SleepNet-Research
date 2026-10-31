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
from .train import binary_metrics, load_run, VARIANTS, PRIMARY_VARIANT, ABLATION_VARIANTS, BASELINE_VARIANTS

CRITERIA = {
    "RQ1": "Pooled OOF N3 F1 >= 0.75 AND PR-AUC >= 0.85 (ISRUC-I + HMC, all scored epochs); "
           "full support additionally requires F1 >= 0.70 in every training cohort and in "
           "every external cohort (ISRUC-II, ISRUC-III, DOD-H, DOD-O).",
    "RQ2": "Each signal factor (delta intensity, slow-wave morphology, spectral dominance) "
           "and continuity is higher in early- than late-night detected N3 (participant-level "
           "one-sided Wilcoxon p < 0.05) and higher for expert N3 than expert N2 epochs; full "
           "support additionally requires the overall rating to increase with the fraction of "
           "experts scoring N3 (graded agreement, multi-rater cohorts, p < 0.05).",
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
    # added 26 Oct 2026, before any full-cohort result existed
    "RQ2b": "Within-N3 slow-wave grade (Low <20 %, Moderate 20-50 %, High >50 % slow-wave "
            "coverage; R&K anchors) is valid if (G3) slow-wave coverage separates expert R&K S4 "
            "from S3 on CAP with AUROC >= 0.75, AND (G1) the High share is larger early than late "
            "in the night (p < 0.05), AND (G2) the grade rises with the fraction of experts "
            "voting N3 (p < 0.05). Partial support: G3 met alone.",
    "RQ1b": "Same-protocol comparison with DeepSleepNet, TinySleepNet, AttnSleep and "
            "L-SeqSleepNet: answered when every model is trained on identical folds and the "
            "participant-level paired difference (F1, PR-AUC) is reported with 95% CI and Holm "
            "correction (direction is a finding, not a criterion).",
}


# ---------------------------------------------------------------------------
# Loading OOF predictions
# ---------------------------------------------------------------------------
RUN_ALIGNMENT = []   # one row per checked run file (written to run_alignment_check.csv)


def check_run_alignment(corpus, f, r, meta):
    """A saved run stores global epoch indices into the corpus.  They are only valid if the
    corpus still contains exactly the recordings (and epoch counts) of the training session.
    Returns True if every predicted epoch maps to a recording of the run's split and every
    such recording is covered completely."""
    want = list(dict.fromkeys(list(meta.get("test_recs", [])) + list(meta.get("external_recs", []))))
    idx = np.asarray(r["idx"])
    reason = ""
    if not want:
        reason = "split has no recordings"
    elif idx.size and (idx.min() < 0 or idx.max() >= len(corpus.ep)):
        reason = "epoch index outside the corpus"
    else:
        missing = [x for x in want if x not in corpus.rec.index]
        if missing:
            reason = f"{len(missing)} split recordings not in the cache (e.g. {missing[0]})"
        else:
            expected = corpus.idx_of(want)
            if len(expected) != len(idx) or not np.array_equal(np.sort(expected), np.sort(idx)):
                got = pd.Series(corpus.ep.rec.to_numpy()[idx]).value_counts()
                bad = [x for x in want if got.get(x, 0) != int(corpus.rec.loc[x, "n"])]
                reason = (f"epoch indices do not match the current corpus "
                          f"({len(bad)} recordings differ, e.g. {bad[0] if bad else '?'})")
    RUN_ALIGNMENT.append({"run": os.path.basename(f), "n_epochs": int(idx.size),
                          "aligned": not reason, "reason": reason})
    if reason:
        print(f"WARNING: {os.path.basename(f)} skipped - {reason}. Its predictions cannot be "
              f"mapped to the current cache; retrain it or restore the matching cache.")
    return not reason


def write_run_alignment(out_dir):
    df = pd.DataFrame(RUN_ALIGNMENT).drop_duplicates("run") if RUN_ALIGNMENT else \
        pd.DataFrame(columns=["run", "n_epochs", "aligned", "reason"])
    df.to_csv(os.path.join(out_dir, "run_alignment_check.csv"), index=False)
    return df


def load_oof(corpus, run_dir, variant):
    files = sorted(glob.glob(os.path.join(run_dir, f"{variant}__fold[0-9]*.npz")))
    if not files:
        return None, None
    rows, ext = [], []
    for f in files:
        r = load_run(f)
        meta = json.loads(r["meta"])
        if not check_run_alignment(corpus, f, r, meta):
            continue
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
    if not rows:
        return None, None
    oof = pd.concat(rows, ignore_index=True)
    oof = oof.join(corpus.ep, on="idx")
    oof = _drop_invalid(oof, variant, "OOF")
    oof["pred"] = (oof.prob >= oof.threshold).astype(int)
    oof["y"] = (oof.label == 3).astype(int)
    exdf = pd.concat(ext, ignore_index=True) if ext else pd.DataFrame()
    if len(exdf):
        num = exdf.drop(columns=["fold"]).groupby("idx").mean()
        num = num.join(corpus.ep, on=num.index.name)
        num = num.reset_index().rename(columns={"index": "idx"})
        num = _drop_invalid(num, variant, "external")
        num["pred"] = (num.prob >= num.threshold).astype(int)
        num["y"] = (num.label == 3).astype(int)
        exdf = num
    return oof, exdf


EXCLUSIONS = []      # filled by load_oof, written by rq1 / write_exclusions


def _drop_invalid(df, variant, split):
    """Remove (i) recordings whose cached signal contained non-finite samples (float16
    overflow; model-independent data-quality rule) and (ii) any remaining non-finite
    prediction.  Every exclusion is recorded for the Methods section."""
    bad = pd.Series(False, index=df.index)
    if "signal_valid" in df:
        inv = ~df.signal_valid.fillna(True).astype(bool)
        for (coh, rec), g in df[inv].groupby(["cohort", "rec"]):
            EXCLUSIONS.append({"variant": variant, "split": split, "cohort": coh, "rec": rec,
                               "n_epochs": len(g), "reason": "non-finite signal in cache (float16 overflow)"})
        bad |= inv
    if "label_valid" in df:
        lv = ~df.label_valid.fillna(True).astype(bool) & ~bad
        for (coh, rec), g in df[lv].groupby(["cohort", "rec"]):
            EXCLUSIONS.append({"variant": variant, "split": split, "cohort": coh, "rec": rec,
                               "n_epochs": len(g),
                               "reason": "scorer hypnograms not aligned to the signal (agreement < 0.6)"})
        bad |= lv
    nf = ~np.isfinite(df.prob.to_numpy(dtype=float))
    nf_only = nf & ~bad.to_numpy()
    if nf_only.any():
        for (coh, rec), g in df[nf_only].groupby(["cohort", "rec"]):
            EXCLUSIONS.append({"variant": variant, "split": split, "cohort": coh, "rec": rec,
                               "n_epochs": len(g), "reason": "non-finite model output"})
    bad |= nf
    return df[~bad.to_numpy()].copy()


def write_exclusions(out_dir):
    ex = pd.DataFrame(EXCLUSIONS).drop_duplicates() if EXCLUSIONS else \
        pd.DataFrame(columns=["variant", "split", "cohort", "rec", "n_epochs", "reason"])
    ex.to_csv(os.path.join(out_dir, "evaluation_exclusions.csv"), index=False)
    return ex


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


def interrater_rows(df, corpus, label):
    """Model vs individual experts vs expert-expert agreement (N3 vs non-N3),
    for every cohort that has >= 2 individual raters."""
    from itertools import combinations
    from sklearn.metrics import cohen_kappa_score
    rows = []
    for coh, g in df.groupby("cohort"):
        mats, preds = [], []
        for rec, gr in g.groupby("rec"):
            lm = corpus.labels_multi.get(rec)
            if lm is None:
                continue
            mats.append(lm[gr.epoch_idx.to_numpy()])
            preds.append(gr.pred.to_numpy())
        if not mats:
            continue
        M, P = np.concatenate(mats), np.concatenate(preds)
        raters = [j for j in range(M.shape[1]) if (M[:, j] >= 0).any()]
        if len(raters) < 2:
            continue
        rr_f1, rr_k, mr_f1, mr_k = [], [], [], []
        for a, b in combinations(raters, 2):
            m = (M[:, a] >= 0) & (M[:, b] >= 0)
            ya, yb = (M[m, a] == 3).astype(int), (M[m, b] == 3).astype(int)
            rr_f1.append(f1_score(ya, yb, zero_division=0))
            rr_k.append(cohen_kappa_score(ya, yb))
        for a in raters:
            m = M[:, a] >= 0
            ya = (M[m, a] == 3).astype(int)
            mr_f1.append(f1_score(ya, P[m], zero_division=0))
            mr_k.append(cohen_kappa_score(ya, P[m]))
        rows.append({"cohort": coh, "set": label, "n_raters": len(raters), "n_epochs": len(M),
                     "n_participants": g.participant.nunique(),
                     "expert_vs_expert_f1_mean": float(np.mean(rr_f1)),
                     "expert_vs_expert_f1_range": (float(np.min(rr_f1)), float(np.max(rr_f1))),
                     "expert_vs_expert_kappa_mean": float(np.mean(rr_k)),
                     "model_vs_expert_f1_mean": float(np.mean(mr_f1)),
                     "model_vs_expert_f1_range": (float(np.min(mr_f1)), float(np.max(mr_f1))),
                     "model_vs_expert_kappa_mean": float(np.mean(mr_k)),
                     "model_within_expert_range": bool(np.mean(mr_f1) >= np.min(rr_f1))})
    return rows


def rq1(corpus, run_dir, out_dir, n_boot=1000):
    oof, ext = load_oof(corpus, run_dir, PRIMARY_VARIANT)
    rows = []
    s = scored(oof)
    for name, g in [("pooled_all_cohorts", s)] + [(f"cohort={c}", g) for c, g in s.groupby("cohort")]:
        m = pooled_metrics(g)
        m.update({f"{k}_ci95": v for k, v in participant_bootstrap(g, n_boot).items()})
        m["subset"] = name
        rows.append(m)
    m = pooled_metrics(s[~s.flag_any_v1.astype(bool)])
    m["subset"] = "pooled_clean_epochs_only(v1 QC flags)"
    rows.append(m)
    for f, g in s.groupby("fold"):
        m = pooled_metrics(g)
        m["subset"] = f"fold={f}"
        rows.append(m)
    if ext is not None and len(ext):
        for c, g in scored(ext, primary_only=False).groupby("cohort"):
            m = pooled_metrics(g)
            m.update({f"{k}_ci95": v for k, v in participant_bootstrap(g, n_boot).items()})
            m["subset"] = f"external={c} (5-fold ensemble)"
            rows.append(m)
    for f in sorted(glob.glob(os.path.join(run_dir, f"{PRIMARY_VARIANT}__foldLOCO_*.npz"))):
        r = load_run(f)
        if not check_run_alignment(corpus, f, r, json.loads(r["meta"])):
            continue
        df = pd.DataFrame({"idx": r["idx"], "prob": r["prob"]}).join(corpus.ep, on="idx")
        df = _drop_invalid(df, PRIMARY_VARIANT + ":LOCO", "LOCO")
        df["pred"] = (df.prob >= float(r["threshold"])).astype(int)
        df["y"] = (df.label == 3).astype(int)
        for c, g in scored(df, primary_only=False).groupby("cohort"):
            m = pooled_metrics(g)
            m["subset"] = f"{json.loads(r['meta'])['fold']} -> test {c}"
            rows.append(m)
    res = pd.DataFrame(rows)
    ir = interrater_rows(scored(oof), corpus, "out-of-fold")
    if ext is not None and len(ext):
        ir += interrater_rows(scored(ext, primary_only=False), corpus, "external")
    ir = pd.DataFrame(ir)
    res.to_csv(os.path.join(out_dir, "rq1_detection_metrics.csv"), index=False)
    write_exclusions(out_dir)
    ir.to_csv(os.path.join(out_dir, "rq1_interrater_benchmark.csv"), index=False)
    oof.to_parquet(os.path.join(out_dir, f"oof_predictions_{PRIMARY_VARIANT}.parquet"))
    if ext is not None and len(ext):
        ext.to_parquet(os.path.join(out_dir, "external_predictions.parquet"))
    return res, ir, oof, ext


# ---------------------------------------------------------------------------
# RQ2 / RQ3 factor scores
# ---------------------------------------------------------------------------
def _finish_scores(S, df):
    S["n3_continuity"] = SC.continuity_score(df.pred.to_numpy(), df.rec.to_numpy(),
                                             df.epoch_idx.to_numpy())
    S["n3_continuity_expert"] = SC.continuity_score(df.y.to_numpy(), df.rec.to_numpy(),
                                                    df.epoch_idx.to_numpy())
    S["rating_prior"] = SC.overall_rating(S, SC.PRIOR_WEIGHTS)
    S["rating_equal"] = SC.overall_rating(S, SC.EQUAL_WEIGHTS)
    return S


def _reference(corpus, recs):
    feat = corpus.feat
    tr = corpus.ep.index[corpus.ep.rec.isin(recs) & (corpus.ep.label == 3)]
    ref_df = feat.loc[tr].copy()
    ref_df["montage"] = corpus.ep.loc[tr, "montage"].to_numpy()
    return SC.FactorReference().fit(ref_df)


def _score_rows(corpus, df, ref):
    X = corpus.feat.loc[df.idx.to_numpy()].copy()
    X.index = df.index
    X["montage"] = df.montage.to_numpy()
    return SC.factor_scores(X, ref)


def compute_fold_scores(corpus, oof, run_dir, variant=PRIMARY_VARIANT):
    """Signal factor scores for OOF epochs; references re-fitted per fold."""
    parts = []
    for f in sorted(oof.fold.unique()):
        meta = json.loads(load_run(os.path.join(run_dir, f"{variant}__fold{f}.npz"))["meta"])
        g = oof[oof.fold == f]
        parts.append(_score_rows(corpus, g, _reference(corpus, meta["train_recs"])))
    return _finish_scores(pd.concat(parts).loc[oof.index], oof)


def compute_external_scores(corpus, ext):
    """External cohorts: reference = all training-cohort expert-N3 epochs."""
    from .train import TRAIN_COHORTS
    recs = list(corpus.rec[corpus.rec.cohort.isin(TRAIN_COHORTS) & corpus.rec.is_primary].index)
    return _finish_scores(_score_rows(corpus, ext, _reference(corpus, recs)), ext)


def _half(df):
    n = df.groupby("rec").epoch_idx.transform("max") + 1
    return np.where(df.epoch_idx < n / 2, "early", "late")


def _early_late(det, fac):
    pm = det.groupby(["participant", "half"])[fac].agg(["median", "size"]).unstack()
    if "late" not in pm["size"] or "early" not in pm["size"]:
        return None
    return pm[(pm["size"]["early"] >= 10) & (pm["size"]["late"] >= 10)]


def graded_agreement(d, fac):
    """Per participant Spearman between a score and the FRACTION of experts who
    scored the epoch N3 (epochs where every rater said N2 or N3, >= 2 raters);
    one-sided Wilcoxon of the participant rhos against 0."""
    from scipy import stats
    g = d[d.all_votes_n2n3 & (d.n_raters_epoch >= 2)]
    rhos = []
    for _, gp in g.groupby("participant"):
        if gp.n3_vote_frac.nunique() >= 2 and len(gp) >= 10:
            rhos.append(ST.spearman(gp[fac], gp.n3_vote_frac)[0])
    rhos = np.array([r for r in rhos if r == r])
    if len(rhos) < 5:
        return {"n": int(len(rhos)), "median_diff": float(np.median(rhos)) if len(rhos) else np.nan,
                "p": np.nan, "W": np.nan}
    t = stats.wilcoxon(rhos, alternative="greater")
    return {"n": int(len(rhos)), "median_diff": float(np.median(rhos)), "p": float(t.pvalue),
            "W": float(t.statistic)}


def rq2(oof, S, corpus, out_dir, ext=None, Sx=None):
    d = oof.join(S)
    d["half"] = _half(d)
    facs = SC.FACTORS + ["rating_prior"]
    rows = []
    det = d[(d.pred == 1) & d.is_primary]
    # T2 homeostatic decline within participant (detected N3), training cohorts (OOF)
    for fac in facs:
        pm = _early_late(det, fac)
        if pm is not None:
            w = ST.wilcoxon_paired(pm["median"]["early"], pm["median"]["late"], "greater")
            ci = ST.boot_ci_median_diff(pm["median"]["early"] - pm["median"]["late"])
            rows.append({"test": "early>late (detected N3)", "factor": fac, **w,
                         "ci95_low": ci[0], "ci95_high": ci[1], "cohort": "all"})
        for c in det.cohort.unique():
            pm = _early_late(det[det.cohort == c], fac)
            if pm is not None:
                w = ST.wilcoxon_paired(pm["median"]["early"], pm["median"]["late"], "greater")
                rows.append({"test": "early>late (detected N3)", "factor": fac, **w, "cohort": c})
    # T1 known groups: expert N3 vs expert N2 (continuity from expert sequence)
    ex = d[d.is_primary & d.label.isin([2, 3])].copy()
    ex["n3_continuity"] = ex["n3_continuity_expert"]
    for fac in SC.FACTORS:
        pm = ex.groupby(["participant", "label"])[fac].median().unstack()
        if 3 in pm and 2 in pm:
            rows.append({"test": "expertN3>expertN2", "factor": fac,
                         **ST.wilcoxon_paired(pm[3], pm[2], "greater"), "cohort": "all"})
    # external cohorts (secondary): homeostatic decline replicated?
    allx = d
    if ext is not None and Sx is not None and len(ext):
        dx = ext.join(Sx)
        dx["half"] = _half(dx)
        detx = dx[dx.pred == 1]
        for c in detx.cohort.unique():
            for fac in facs:
                pm = _early_late(detx[detx.cohort == c], fac)
                if pm is not None:
                    w = ST.wilcoxon_paired(pm["median"]["early"], pm["median"]["late"], "greater")
                    rows.append({"test": "early>late (detected N3)", "factor": fac, **w,
                                 "cohort": f"external:{c}"})
        allx = pd.concat([d, dx], ignore_index=True)
    # T4 graded expert agreement (ISRUC 2 scorers, DOD 5 scorers)
    for fac in SC.FACTORS[:3] + ["rating_prior"]:
        rows.append({"test": "graded agreement: score ~ fraction of experts voting N3",
                     "factor": fac, **graded_agreement(allx, fac), "cohort": "all multi-rater"})
        for c in allx[allx.n_raters_epoch >= 2].cohort.unique():
            rows.append({"test": "graded agreement: score ~ fraction of experts voting N3",
                         "factor": fac, **graded_agreement(allx[allx.cohort == c], fac), "cohort": c})
    # T3 age (only where per-participant age exists)
    pmeta = corpus.participant_meta.get("participants", {})
    if pmeta:
        detall = allx[(allx.pred == 1)]
        pr = detall.groupby(["participant", "cohort"])[facs].median().reset_index()
        pr["age"] = pr.participant.map(lambda p: pmeta.get(p, {}).get("age", np.nan))
        pr = pr[pr.age.notna()]
        for c, g in [("all with age", pr)] + list(pr.groupby("cohort")):
            for fac in facs:
                rho, p = ST.spearman(g[fac], g["age"])
                rows.append({"test": "spearman_vs_age (expect <0)", "factor": fac,
                             "n": int(len(g)), "median_diff": rho, "p": p, "cohort": c})
    # table for the graded-agreement figure
    g = allx[allx.all_votes_n2n3 & (allx.n_raters_epoch >= 2)]
    if len(g):
        g = g.assign(vote=g.n3_vote_frac.round(2))
        g.groupby(["cohort", "vote"])[SC.FACTORS[:3] + ["rating_prior"]].agg(["mean", "count"]) \
         .to_csv(os.path.join(out_dir, "rq2_scores_by_expert_vote_fraction.csv"))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "rq2_factor_validity.csv"), index=False)
    det.groupby("cohort")[facs].describe().T.to_csv(
        os.path.join(out_dir, "rq2_factor_descriptives_detected_n3.csv"))
    return res


def test_retest(d, value_col="rating_prior", min_rated=10):
    """First vs second recording of the same participant (ISRUC-II sessions)."""
    det = d[d.pred == 1]
    g = det.groupby(["participant", "rec"])[value_col].agg(["median", "size"]).reset_index()
    g = g[g["size"] >= min_rated].sort_values(["participant", "rec"])
    pairs = g.groupby("participant")["median"].apply(list)
    pairs = pairs[pairs.apply(len) >= 2]
    if not len(pairs):
        return {"n_participants": 0, "icc21": np.nan, "icc31": np.nan}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return {"n_participants": int(len(pairs)), "icc21": ST.icc(a, b), "icc31": ST.icc(a, b, "3,1")}


def rq3(oof, S, out_dir, ext=None, Sx=None, n_draws=1000, seed=0):
    d = oof.join(S)
    det = d[(d.pred == 1) & d.is_primary].copy()
    rng = np.random.default_rng(seed)
    F = det[SC.FACTORS].to_numpy()
    prior = np.array([SC.PRIOR_WEIGHTS[k] for k in SC.FACTORS])
    base = F @ prior
    out = {}
    from scipy.stats import rankdata
    rb = rankdata(base)
    for name, alpha in [("reasonable(alpha=20*prior)", 20 * prior), ("extreme(alpha=1)", np.ones(4))]:
        W = rng.dirichlet(alpha, n_draws)
        R = F @ W.T
        rhos = [np.corrcoef(rb, rankdata(R[:, j]))[0, 1] for j in range(n_draws)]
        out[name] = {"median_rho": float(np.median(rhos)), "p05_rho": float(np.percentile(rhos, 5)),
                     "min_rho": float(np.min(rhos))}
    pca_w = SC.pca_weights(det)
    det["rating_pca"] = SC.overall_rating(det, pca_w)
    alt = {"prior_vs_equal_rho": ST.spearman(det.rating_prior, det.rating_equal)[0],
           "prior_vs_pca_rho": ST.spearman(det.rating_prior, det.rating_pca)[0],
           "pca_weights_pooled": pca_w}
    det[SC.FACTORS].corr(method="spearman").to_csv(
        os.path.join(out_dir, "rq3_factor_redundancy_spearman.csv"))
    miss = {}
    for k in SC.FACTORS:
        w = {kk: v for kk, v in SC.PRIOR_WEIGHTS.items() if kk != k}
        w[k] = 0.0
        miss[f"without_{k}"] = ST.spearman(det.rating_prior, SC.overall_rating(det, w))[0]
    det = det.sort_values(["rec", "epoch_idx"])
    det["parity"] = det.groupby("rec").cumcount() % 2
    sh = det.groupby(["participant", "parity"]).rating_prior.median().unstack().dropna()
    r_sh = ST.spearman(sh[0], sh[1])[0] if len(sh) > 3 else np.nan
    sb = 2 * r_sh / (1 + r_sh) if r_sh == r_sh else np.nan
    tr = {"note": "first vs second recording of the same participant (ISRUC-II sessions)"}
    if ext is not None and Sx is not None and len(ext):
        dx = ext.join(Sx)
        for col in ("rating_prior", "delta_intensity", "sw_morphology", "spectral_dominance"):
            tr[col] = test_retest(dx, col)
    pr = det.groupby(["participant", "cohort"]).rating_prior.median().reset_index()
    cis = {}
    for c, g in pr.groupby("cohort"):
        v = g.rating_prior.to_numpy()
        bs = [np.mean(rng.choice(v, len(v))) for _ in range(2000)]
        cis[c] = {"mean_participant_median": float(v.mean()),
                  "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res = {"weight_sensitivity": out, "alternative_weightings": alt,
           "missing_factor_rho": miss, "split_half_spearman_brown": sb,
           "test_retest": tr, "cohort_rating_ci": cis,
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
    for v in ABLATION_VARIANTS + ["feat_gbm"]:
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


# Added in v2.1, after the first full run (20261028-0834).  Reported in their own
# table with their own Holm family; they do NOT enter the pre-specified RQ4 verdict.
EXTENDED_COMPARISONS = [
    ("feature fusion | full model", "full_v2", "full_nofeat"),
    ("positive-class weighting | full model", "full_v2", "full_nopw"),
    ("temporal attention | full model", "full_v2", "full_noattn"),
    ("context receptive field +/-15 vs +/-3 epochs", "full_v2", "full_ctx_r3"),
    ("all 19 features vs delta-group features only", "full_v2", "full_deltafeat"),
]


def rq4_extended(corpus, run_dir, out_dir):
    """Leave-one-component-out ablations from the full model, training-seed
    variability, and every ablation variant on the external cohorts."""
    from .train import SEED_VARIANTS
    names = ABLATION_VARIANTS + ["feat_gbm"]
    per_part, oofs, exts = {}, {}, {}
    for v in names:
        oof, ext = load_oof(corpus, run_dir, v)
        if oof is None:
            continue
        oofs[v], exts[v] = scored(oof), ext
        per_part[v] = _participant_metrics(oofs[v])
    # seed variability of the primary model
    seed_rows, seed_vs = [], [v for v in SEED_VARIANTS if v in oofs]
    for v in seed_vs:
        m = pooled_metrics(oofs[v])
        seed_rows.append({"variant": v, "f1": m["f1"], "pr_auc": m["pr_auc"], "kappa": m["kappa"]})
    noise = {}
    if len(seed_vs) >= 2:
        sd = pd.DataFrame(seed_rows)
        seed_rows.append({"variant": "SD across seeds", "f1": sd.f1.std(ddof=1),
                          "pr_auc": sd.pr_auc.std(ddof=1), "kappa": sd.kappa.std(ddof=1)})
        seed_rows.append({"variant": "range across seeds", "f1": sd.f1.max() - sd.f1.min(),
                          "pr_auc": sd.pr_auc.max() - sd.pr_auc.min(),
                          "kappa": sd.kappa.max() - sd.kappa.min()})
        # participant-level seed-to-seed |median difference|: the noise floor for RQ4
        for metric in ("f1", "pr_auc"):
            diffs = []
            for a, b in [(seed_vs[i], seed_vs[j]) for i in range(len(seed_vs))
                         for j in range(i + 1, len(seed_vs))]:
                j = per_part[a][[metric]].join(per_part[b][[metric]], lsuffix="_a", rsuffix="_b").dropna()
                diffs.append(abs(float(np.median(j[f"{metric}_a"] - j[f"{metric}_b"]))))
            noise[metric] = max(diffs)
    pd.DataFrame(seed_rows).to_csv(os.path.join(out_dir, "rq4_seed_variability.csv"), index=False)
    # leave-one-component-out comparisons, own Holm family
    rows = []
    for name, a, b in EXTENDED_COMPARISONS:
        if a not in per_part or b not in per_part:
            continue
        for metric in ("f1", "pr_auc"):
            j = per_part[a][[metric]].join(per_part[b][[metric]], lsuffix="_a", rsuffix="_b").dropna()
            w = ST.wilcoxon_paired(j[f"{metric}_a"], j[f"{metric}_b"])
            ci = ST.boot_ci_median_diff(j[f"{metric}_a"] - j[f"{metric}_b"])
            rows.append({"component": name, "with": a, "without": b, "metric": metric,
                         **w, "ci95_low": ci[0], "ci95_high": ci[1],
                         "seed_noise_floor": noise.get(metric, np.nan)})
    comp = pd.DataFrame(rows)
    if len(comp):
        comp["p_holm"] = ST.holm(comp.p.to_numpy())
        # an effect is interpretable only if it is significant AND larger than the
        # largest seed-to-seed difference of the primary model
        comp["exceeds_seed_noise"] = comp.median_diff.abs() > comp.seed_noise_floor
    comp.to_csv(os.path.join(out_dir, "rq4_extended_component_effects.csv"), index=False)
    # every variant on every external cohort (5-fold ensembles)
    xr = []
    for v, ext in exts.items():
        if ext is None or not len(ext):
            continue
        for c, g in scored(ext, primary_only=False).groupby("cohort"):
            m = pooled_metrics(g)
            xr.append({"variant": v, "cohort": c, **{k: m.get(k) for k in
                                                     ("f1", "pr_auc", "roc_auc", "precision", "recall",
                                                      "kappa", "n_participants")}})
    pd.DataFrame(xr).to_csv(os.path.join(out_dir, "rq4_variant_external_metrics.csv"), index=False)
    return comp


# ---------------------------------------------------------------------------
# RQ1b: same-protocol comparison with published architectures
# ---------------------------------------------------------------------------
def sota_comparison(corpus, run_dir, out_dir, n_boot=1000):
    """SleepQualityNet v2 vs DeepSleepNet / TinySleepNet / AttnSleep / L-SeqSleepNet,
    all re-trained on identical folds with identical selection, threshold and
    calibration.  Participant-level paired Wilcoxon (Holm) and bootstrap CIs."""
    models = [PRIMARY_VARIANT] + BASELINE_VARIANTS
    rows, per_part, ext_rows = [], {}, []
    for v in models:
        oof, ext = load_oof(corpus, run_dir, v)
        if oof is None:
            continue
        n_folds = oof.fold.nunique()
        s = scored(oof)
        r = load_run(sorted(glob.glob(os.path.join(run_dir, f"{v}__fold[0-9]*.npz")))[0])
        m = pooled_metrics(s)
        m.update({f"{k}_ci95": val for k, val in participant_bootstrap(s, n_boot).items()})
        m.update(model=v, set="cross-validation (training cohorts)", n_folds=n_folds,
                 n_params=int(r.get("n_params", 0)),
                 train_minutes_per_fold=float(r.get("seconds", np.nan)) / 60)
        rows.append(m)
        per_part[v] = _participant_metrics(s)
        if ext is not None and len(ext):
            for c, g in scored(ext, primary_only=False).groupby("cohort"):
                mm = pooled_metrics(g)
                mm.update(model=v, set=f"external={c}", n_folds=n_folds,
                          n_params=int(r.get("n_params", 0)))
                ext_rows.append(mm)
    tab = pd.DataFrame(rows + ext_rows)
    if len(tab):
        lead = ["model", "set", "f1", "pr_auc", "roc_auc", "kappa", "precision", "recall",
                "n_participants", "n_params"]
        tab = tab[[c for c in lead if c in tab] + [c for c in tab if c not in lead]]
    tab.to_csv(os.path.join(out_dir, "rq1_sota_comparison.csv"), index=False)
    tests = []
    for b in BASELINE_VARIANTS:
        if b not in per_part or PRIMARY_VARIANT not in per_part:
            continue
        for metric in ("f1", "pr_auc"):
            j = per_part[PRIMARY_VARIANT][[metric]].join(per_part[b][[metric]], lsuffix="_a",
                                                         rsuffix="_b").dropna()
            w = ST.wilcoxon_paired(j[f"{metric}_a"], j[f"{metric}_b"])
            ci = ST.boot_ci_median_diff(j[f"{metric}_a"] - j[f"{metric}_b"])
            tests.append({"comparison": f"{PRIMARY_VARIANT} vs {b}", "metric": metric,
                          "median_participant_diff": float(np.median(j[f"{metric}_a"] - j[f"{metric}_b"]))
                          if len(j) else np.nan,
                          **w, "ci95_low": ci[0], "ci95_high": ci[1]})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_holm"] = ST.holm(tests.p.to_numpy())
    tests.to_csv(os.path.join(out_dir, "rq1_sota_paired_tests.csv"), index=False)
    return tab, tests


# ---------------------------------------------------------------------------
# RQ2b: within-N3 grade (Low / Moderate / High)
# ---------------------------------------------------------------------------
# Two grades for every DETECTED N3 epoch; neither creates a new label:
#  * "N3 slow-wave grade" (primary): fraction of the epoch covered by slow waves
#    (0.5-2 Hz full waves, peak-to-peak >= 75 uV) with the Rechtschaffen & Kales
#    anchors: < 20 % Low (below the AASM N3 amplitude criterion), 20-50 % Moderate
#    (R&K stage 3), > 50 % High (R&K stage 4).  Thresholds come from the scoring
#    manuals, not from our data or weights.
#  * "N3 EEG Rating band" (secondary): overall N3 EEG Rating 0-33 / 33-67 / 67-100.
# Validation references, all independent of the grade construction:
#  G1 within-night (homeostatic) decline of the High share;
#  G2 graded expert agreement (fraction of 2-5 experts voting N3);
#  G3 EXPERT R&K S3 vs S4 on the CAP Sleep Database (visual scoring by the CAP experts);
#  G4 night-to-night stability (ISRUC-II).
GRADE_NAMES = ("Low", "Moderate", "High")
SW_GRADE_EDGES = (0.20, 0.50)            # R&K / AASM anchors (fraction of the 30-s epoch)
BAND_EDGES = (100 / 3, 200 / 3)          # rating scale tertiles, fixed a priori


def _grade(v, edges):
    v = np.asarray(v, dtype=float)
    out = np.where(v > edges[1], 2, np.where(v >= edges[0], 1, 0)).astype(float)
    out[np.isnan(v)] = np.nan
    return out


def n3_band(r):
    return _grade(r, BAND_EDGES)


def n3_sw_grade(sw_time_frac):
    return _grade(sw_time_frac, SW_GRADE_EDGES)


def rq2_bands(oof, S, out_dir, ext=None, Sx=None, rating="rating_prior", corpus=None):
    from sklearn.metrics import cohen_kappa_score, roc_auc_score as _auc
    parts = [oof.join(S).assign(split="OOF")]
    if ext is not None and Sx is not None and len(ext):
        parts.append(ext.join(Sx).assign(split="external"))
    d = pd.concat(parts, ignore_index=True)
    d["half"] = _half(d)
    d["rating_band"] = n3_band(d[rating])
    if corpus is not None and "sw_time_frac" in corpus.feat:
        d["sw_time_frac"] = corpus.feat.loc[d.idx.to_numpy(), "sw_time_frac"].to_numpy()
        d["sw_grade"] = n3_sw_grade(d.sw_time_frac)
    grades = [g for g in ("sw_grade", "rating_band") if g in d]
    det = d[d.pred == 1]
    rows = []
    for gcol in grades:
        dd = det[det[gcol].notna()]
        (dd.groupby(["split", "cohort"])[gcol].value_counts(normalize=True).unstack(fill_value=0)
           .rename(columns=dict(enumerate(GRADE_NAMES)))
           .to_csv(os.path.join(out_dir, f"rq2_{gcol}_distribution.csv")))
        # G1 High share early > late, per participant
        for name, g in [("all", dd)] + list(dd.groupby("cohort")):
            sh = g.assign(high=(g[gcol] == 2).astype(float)).groupby(["participant", "half"]).high \
                  .agg(["mean", "size"]).unstack()
            if "early" not in sh["size"] or "late" not in sh["size"]:
                continue
            sh = sh[(sh["size"]["early"] >= 10) & (sh["size"]["late"] >= 10)]
            if len(sh) >= 3:
                rows.append({"grade": gcol, "test": "G1 High share early > late", "cohort": name,
                             **ST.wilcoxon_paired(sh["mean"]["early"], sh["mean"]["late"], "greater")})
        # G2 graded expert agreement
        g = dd[dd.all_votes_n2n3 & (dd.n_raters_epoch >= 2)]
        for name, gg in [("all multi-rater", g)] + list(g.groupby("cohort")):
            if gg.n3_vote_frac.nunique() >= 2 and len(gg) >= 30:
                rho, p = ST.spearman(gg[gcol], gg.n3_vote_frac)
                rows.append({"grade": gcol, "test": "G2 grade ~ fraction of experts voting N3 (Spearman)",
                             "cohort": name, "n": int(len(gg)), "median_diff": rho, "p": p})
        if len(g):
            (g.assign(vote=g.n3_vote_frac.round(2)).groupby(["cohort", "vote"])[gcol]
              .value_counts(normalize=True).unstack(fill_value=0)
              .rename(columns=dict(enumerate(GRADE_NAMES)))
              .to_csv(os.path.join(out_dir, f"rq2_{gcol}_by_expert_vote_fraction.csv")))
        # G4 ISRUC-II test-retest of the High share
        tr = dd[dd.cohort == "ISRUC-II"]
        if len(tr):
            g2 = tr.assign(high=(tr[gcol] == 2).astype(float)).groupby(["participant", "rec"]).high \
                   .agg(["mean", "size"]).reset_index()
            g2 = g2[g2["size"] >= 10].sort_values(["participant", "rec"])
            pairs = g2.groupby("participant")["mean"].apply(list)
            pairs = pairs[pairs.apply(len) >= 2]
            if len(pairs) >= 3:
                rows.append({"grade": gcol, "test": "G4 High share test-retest ICC(2,1)",
                             "cohort": "ISRUC-II", "n": int(len(pairs)),
                             "median_diff": ST.icc(np.array([q[0] for q in pairs]),
                                                   np.array([q[1] for q in pairs])), "p": np.nan})
    # G3 expert R&K S3 vs S4 (CAP)
    rk_rows = []
    if "rk_code" in d:
        cap = d[d.rk_code.isin([3, 4])]
        for subset, g in (("expert S3/S4 epochs", cap), ("expert S3/S4 AND detected N3", cap[cap.pred == 1])):
            if g.rk_code.nunique() < 2 or len(g) < 30:
                continue
            y4 = (g.rk_code == 4).astype(int).to_numpy()
            r = {"subset": subset, "n_epochs": int(len(g)), "n_participants": int(g.participant.nunique()),
                 "share_S4": float(y4.mean())}
            for col in ("sw_time_frac", rating, "delta_intensity", "sw_morphology",
                        "spectral_dominance"):
                if col in g and g[col].notna().any():
                    m = g[col].notna().to_numpy()
                    r[f"auroc_S4_vs_S3_{col}"] = float(_auc(y4[m], g[col].to_numpy()[m]))
            if "sw_grade" in g:
                m = g.sw_grade.notna().to_numpy()
                hi = (g.sw_grade.to_numpy()[m] == 2).astype(int)
                r["kappa_High_vs_S4"] = float(cohen_kappa_score(y4[m], hi))
                r["sens_High_for_S4"] = float(hi[y4[m] == 1].mean()) if (y4[m] == 1).any() else np.nan
                r["spec_notHigh_for_S3"] = float(1 - hi[y4[m] == 0].mean()) if (y4[m] == 0).any() else np.nan
                ct = pd.crosstab(g.rk_code.map({3: "expert S3", 4: "expert S4"}),
                                 g.sw_grade.map(dict(enumerate(GRADE_NAMES))))
                ct.to_csv(os.path.join(out_dir, f"rq2_rk_crosstab_{'detected' if 'detected' in subset else 'all'}.csv"))
            # participant-level: per participant AUROC, Wilcoxon > 0.5
            au = []
            for _, gp in g.groupby("participant"):
                yy = (gp.rk_code == 4).astype(int)
                if yy.nunique() == 2 and gp.sw_time_frac.notna().all() if "sw_time_frac" in gp else False:
                    au.append(_auc(yy, gp.sw_time_frac))
            if len(au) >= 5:
                from scipy import stats as _st
                r["participant_auroc_median"] = float(np.median(au))
                r["participant_auroc_p_gt_0.5"] = float(_st.wilcoxon(np.array(au) - 0.5,
                                                                     alternative="greater").pvalue)
            rk_rows.append(r)
            if "sw_grade" in g:
                rows.append({"grade": "sw_grade", "test": f"G3 expert R&K S4 vs S3 AUROC [{subset}]",
                             "cohort": "CAP", "n": int(len(g)),
                             "median_diff": r.get("auroc_S4_vs_S3_sw_time_frac", np.nan),
                             "p": r.get("participant_auroc_p_gt_0.5", np.nan)})
    pd.DataFrame(rk_rows).to_csv(os.path.join(out_dir, "rq2_rk_grade_validation.csv"), index=False)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out_dir, "rq2_n3_grade_validity.csv"), index=False)
    cols = ["idx", "rec", "epoch_idx", "participant", "cohort", "split", "pred", "label"] + \
        [c for c in ("rk_code", "sw_time_frac", "sw_grade", rating, "rating_band") if c in d]
    d.loc[d.pred == 1, cols].to_parquet(os.path.join(out_dir, "n3_grades_detected_epochs.parquet"))
    return res


# ---------------------------------------------------------------------------
# RQ6 reliability
# ---------------------------------------------------------------------------
RELIABILITY_FEATURES = ["confidence", "mc_entropy", "mc_mi", "maha", "qc_score_v1",
                        "qc_hf_ratio", "qc_n_clipped", "flag_any_v1"]


def _rel_matrix(d):
    p = d.prob_cal if "prob_cal" in d else d.prob
    d = d.assign(confidence=np.maximum(p, 1 - p))
    X = d[RELIABILITY_FEATURES].astype(float).to_numpy()
    return d, np.column_stack([X, np.log1p(np.clip(d.maha.to_numpy(), 0, None))])


def _disagreement_rows(d, label):
    """Independent reference: do low-reliability epochs coincide with epochs on
    which human experts disagree about N3 (ISRUC 2 scorers, DOD 5 scorers)?"""
    out = []
    m = d[(d.n_raters_epoch >= 2) & d.n3_vote_frac.notna()]
    for coh, g in [("all", m)] + list(m.groupby("cohort")):
        dis = ((g.n3_vote_frac > 0) & (g.n3_vote_frac < 1)).astype(int)
        if dis.nunique() < 2:
            continue
        out.append({"component": f"COMBINED vs expert N3 disagreement [{label}:{coh}]",
                    "group": "independent_reference",
                    "error_detection_auroc": roc_auc_score(dis, -g.reliability)})
        out.append({"component": f"mc_entropy vs expert N3 disagreement [{label}:{coh}]",
                    "group": "independent_reference",
                    "error_detection_auroc": roc_auc_score(dis, g.mc_entropy)})
    return out


def rq6(oof, out_dir, ext=None):
    d = scored(oof).copy()
    d, X = _rel_matrix(d)
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
    rel = np.zeros(len(d))
    yc_all = 1 - d.error.to_numpy()
    for f in d.fold.unique():
        tr, te = d.fold.to_numpy() != f, d.fold.to_numpy() == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        if np.unique(yc_all[tr]).size < 2:
            rel[te] = float(yc_all[tr].mean())
            continue
        lr = LogisticRegression(max_iter=2000).fit((X[tr] - mu) / sd, yc_all[tr])
        rel[te] = lr.predict_proba((X[te] - mu) / sd)[:, 1]
    d["reliability"] = 100 * rel
    d["reliability_cat"] = pd.cut(d.reliability, [-1, 80, 95, 101], labels=["low", "medium", "high"])
    auc = roc_auc_score(d.error, -d.reliability) if d.error.nunique() > 1 else np.nan
    rc = []
    order = np.argsort(-d.reliability.to_numpy())
    for cov in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
        g = d.iloc[order[: int(round(cov * len(d)))]]
        rc.append({"coverage": cov, "f1": f1_score(g.y, g.pred, zero_division=0),
                   "accuracy": float((g.y == g.pred).mean()), "n": len(g)})
    rc = pd.DataFrame(rc)
    cat = d.groupby("reliability_cat", observed=False).agg(
        n=("error", "size"), error_rate=("error", "mean")).reset_index()
    comp.loc[len(comp)] = {"component": "COMBINED (cross-fitted logistic)", "group": "combined",
                           "error_detection_auroc": auc}
    for r in _disagreement_rows(d, "OOF"):
        comp.loc[len(comp)] = r
    # external cohorts: model fitted on ALL OOF epochs, applied unchanged
    rel_ext = None
    if ext is not None and len(ext):
        e = scored(ext, primary_only=False).copy()
        e, Xe = _rel_matrix(e)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        if np.unique(yc_all).size > 1:
            lr = LogisticRegression(max_iter=2000).fit((X - mu) / sd, yc_all)
            e["reliability"] = 100 * lr.predict_proba((Xe - mu) / sd)[:, 1]
        else:
            e["reliability"] = 100 * float(yc_all.mean())
        e["error"] = (e.pred != e.y).astype(int)
        for c, g in e.groupby("cohort"):
            if g.error.nunique() > 1:
                comp.loc[len(comp)] = {"component": f"COMBINED [external:{c}]", "group": "external",
                                       "error_detection_auroc": roc_auc_score(g.error, -g.reliability)}
        for r in _disagreement_rows(e, "external"):
            comp.loc[len(comp)] = r
        e["reliability_cat"] = pd.cut(e.reliability, [-1, 80, 95, 101], labels=["low", "medium", "high"])
        rel_ext = e[["idx", "reliability", "reliability_cat"]]
    comp.to_csv(os.path.join(out_dir, "rq6_reliability_components.csv"), index=False)
    rc.to_csv(os.path.join(out_dir, "rq6_risk_coverage.csv"), index=False)
    cat.to_csv(os.path.join(out_dir, "rq6_reliability_categories.csv"), index=False)
    return comp, rc, cat, d[["idx", "reliability", "reliability_cat"]], rel_ext


# ---------------------------------------------------------------------------
# RQ7 episodes and Overnight N3 EEG Profile
# ---------------------------------------------------------------------------
def _profiles(d, split):
    prows, erows = [], []
    for rec, g in d.groupby("rec"):
        g = g.sort_values("epoch_idx")
        rated = np.where(g.pred == 1, g.rating_prior, np.nan)
        pp, eps = SC.night_profile(g.pred.to_numpy(), rated, g.n3_continuity.to_numpy(),
                                   g.get("reliability", pd.Series(np.nan, index=g.index)).to_numpy(),
                                   rel_high=95)
        Sx = g[SC.FACTORS].copy()
        Sx["n3_continuity"] = g.n3_continuity_expert
        rx = np.where(g.y == 1, SC.overall_rating(Sx), np.nan)
        pe, _ = SC.night_profile(g.y.to_numpy(), rx, g.n3_continuity_expert.to_numpy())
        base = {"rec": rec, "participant": g.participant.iloc[0], "cohort": g.cohort.iloc[0],
                "is_primary": bool(g.is_primary.iloc[0]), "split": split}
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
    return prows, erows


PROFILE_METRICS = ("n3_minutes", "n_episodes", "longest_episode_min", "median_episode_min",
                   "n3_first_half_fraction", "median_rating")


def _agreement(pr, label):
    rows = []
    for k in PROFILE_METRICS:
        a, b = pr[f"pred_{k}"], pr[f"expert_{k}"]
        rows.append({"set": label, "metric": k, "icc21": ST.icc(a, b), "icc31": ST.icc(a, b, "3,1"),
                     "spearman": ST.spearman(a, b)[0], **ST.bland_altman(a, b)})
    return rows


def rq7(oof, S, rel_df, out_dir, ext=None, Sx=None, rel_ext=None):
    d = oof.join(S)
    if rel_df is not None:
        d = d.merge(rel_df, on="idx", how="left")
    prows, erows = _profiles(d, "OOF")
    if ext is not None and Sx is not None and len(ext):
        dx = ext.join(Sx)
        if rel_ext is not None:
            dx = dx.merge(rel_ext, on="idx", how="left")
        p2, e2 = _profiles(dx, "external")
        prows += p2
        erows += e2
    prof, epi = pd.DataFrame(prows), pd.DataFrame(erows)
    prof.to_csv(os.path.join(out_dir, "rq7_overnight_n3_eeg_profiles.csv"), index=False)
    epi.to_csv(os.path.join(out_dir, "rq7_n3_episodes.csv"), index=False)
    agree = _agreement(prof[(prof.split == "OOF") & prof.is_primary], "OOF (criterion)")
    for c, g in prof[prof.split == "external"].groupby("cohort"):
        agree += _agreement(g, f"external:{c}")
    agree = pd.DataFrame(agree)
    # test-retest (ISRUC-II: two recordings per participant, never trained on).
    # Tests whether factor ratings are more reproducible than conventional N3 minutes.
    multi = prof.groupby("participant").rec.transform("nunique") >= 2
    tp = prof[multi].sort_values(["participant", "rec"])
    for m in ("pred_n3_minutes", "expert_n3_minutes", "pred_median_rating",
              "pred_median_delta_intensity", "pred_median_sw_morphology",
              "pred_median_spectral_dominance", "pred_n_episodes", "pred_longest_episode_min"):
        if m not in tp or not len(tp):
            continue
        firsts = tp.groupby("participant")[m].nth(0).to_numpy()
        seconds = tp.groupby("participant")[m].nth(1).to_numpy()
        agree.loc[len(agree)] = {"set": "test-retest (ISRUC-II sessions)", "metric": m,
                                 "icc21": ST.icc(firsts, seconds), "icc31": ST.icc(firsts, seconds, "3,1"),
                                 "spearman": ST.spearman(firsts, seconds)[0],
                                 **ST.bland_altman(firsts, seconds)}
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


def rq_summary(out_dir, r1=None, r2=None, r3=None, r4=None, r5=None, r6=None, r7=None,
               r2b=None, r1b=None):
    rows = []
    if r1 is not None:
        t = r1.set_index("subset")
        po = t.loc["pooled_all_cohorts"]
        coh = t[t.index.str.startswith("cohort=")]
        ext = t.loc[t.index.str.startswith("external=")]
        from .train import EXTERNAL_COHORTS as _EXT
        ext = ext[[i.split("=")[1].split(" ")[0] in _EXT for i in ext.index]]   # pre-specified only
        base_ok = po.f1 >= 0.75 and po.pr_auc >= 0.85
        full = base_ok and (coh.f1 >= 0.70).all() and (len(ext) == 0 or (ext.f1 >= 0.70).all())
        obs = (f"pooled F1={po.f1:.3f} {tuple(round(x, 3) for x in po.f1_ci95)}, "
               f"PR-AUC={po.pr_auc:.3f}; " + "; ".join(f"{i}: F1={r.f1:.3f}" for i, r in coh.iterrows())
               + ("; " + "; ".join(f"{i.split('=')[1].split(' ')[0]}: F1={r.f1:.3f}"
                                   for i, r in ext.iterrows()) if len(ext) else ""))
        rows.append(("RQ1", CRITERIA["RQ1"], obs, _status(full, base_ok)))
    if r2 is not None:
        t = r2[(r2.test == "early>late (detected N3)") & (r2.cohort == "all") &
               r2.factor.isin(SC.FACTORS)]
        k = r2[(r2.test == "expertN3>expertN2")]
        ok_t = (t.p < 0.05).sum()
        ok_k = (k.p < 0.05).sum()
        ga = r2[(r2.test.str.startswith("graded agreement")) & (r2.factor == "rating_prior") &
                (r2.cohort == "all multi-rater")]
        ga_ok = len(ga) > 0 and bool((ga.p < 0.05).all())
        obs = "; ".join(f"{r.factor}: early-late={r.median_diff:.1f} (p={r.p:.2g})" for _, r in t.iterrows())
        obs += f"; N3>N2 significant for {ok_k}/4"
        if len(ga):
            obs += f"; graded agreement (overall rating) median rho={ga.median_diff.iloc[0]:.2f}, p={ga.p.iloc[0]:.2g}"
        full = ok_t == 4 and ok_k == 4 and ga_ok
        rows.append(("RQ2", CRITERIA["RQ2"], obs, _status(full, ok_t >= 2 or ga_ok)))
    if r3 is not None and r2 is not None:
        ws = r3["weight_sensitivity"]["reasonable(alpha=20*prior)"]["p05_rho"]
        cc = r2[(r2.factor == "rating_prior") & (r2.test == "early>late (detected N3)") &
                (r2.cohort != "all") & ~r2.cohort.astype(str).str.startswith("external:")]
        coh_ok = len(cc) > 0 and (cc.p < 0.05).all()
        trr = r3.get("test_retest", {}).get("rating_prior", {})
        obs = (f"p05 rho={ws:.3f}; test-retest ICC(2,1)={trr.get('icc21', float('nan')):.3f} "
               f"(n={trr.get('n_participants', 0)}); "
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
        auc = float(comp[comp.component == "COMBINED (cross-fitted logistic)"].error_detection_auroc.iloc[0])
        f100 = float(rc[rc.coverage == 1.0].f1.iloc[0])
        f80 = float(rc[rc.coverage == 0.8].f1.iloc[0])
        rows.append(("RQ6", CRITERIA["RQ6"], f"AUROC={auc:.3f}; F1@100%={f100:.3f}, F1@80%={f80:.3f}",
                     _status(auc >= 0.75 and f80 > f100, auc >= 0.75 or f80 > f100)))
    if r7 is not None:
        ag = r7[2]
        a = ag[ag.set == "OOF (criterion)"].set_index("metric")
        icc_m = a.loc["n3_minutes", "icc21"]
        obs = "; ".join(f"{r.set}/{r.metric}: ICC={r.icc21:.3f}" for _, r in ag.iterrows()
                        if r.metric in ("n3_minutes", "median_rating", "longest_episode_min")
                        or r.set.startswith("test-retest"))
        rows.append(("RQ7", CRITERIA["RQ7"], obs,
                     _status(icc_m >= 0.75, icc_m >= 0.60)))
    if r1b is not None:
        tab, tests = r1b
        f1t = tests[tests.metric == "f1"] if len(tests) else tests
        obs = "; ".join(f"{r.comparison}: dF1={r.median_participant_diff:+.3f} "
                        f"[{r.ci95_low:+.3f},{r.ci95_high:+.3f}] p_holm={r.p_holm:.2g}"
                        for _, r in f1t.iterrows()) if len(f1t) else "baselines not trained yet"
        from .train import BASELINE_VARIANTS as _BV
        rows.append(("RQ1b", CRITERIA["RQ1b"], obs,
                     "ANSWERED" if len(f1t) == len(_BV) else
                     ("PARTIALLY_ANSWERED" if len(f1t) else "NOT_RUN")))
    if r2b is not None and len(r2b):
        sw = r2b[r2b.grade == "sw_grade"]
        g3 = sw[sw.test.str.startswith("G3") & sw.test.str.contains("detected")]
        if not len(g3):
            g3 = sw[sw.test.str.startswith("G3")]
        g1 = sw[(sw.test.str.startswith("G1")) & (sw.cohort == "all")]
        g2 = sw[(sw.test.str.startswith("G2")) & (sw.cohort == "all multi-rater")]
        ok3 = bool(len(g3)) and float(g3.median_diff.iloc[0]) >= 0.75
        ok1 = bool(len(g1)) and float(g1.p.iloc[0]) < 0.05
        ok2 = bool(len(g2)) and float(g2.p.iloc[0]) < 0.05 and float(g2.median_diff.iloc[0]) > 0
        obs = (f"CAP S4-vs-S3 AUROC={float(g3.median_diff.iloc[0]):.3f}; " if len(g3) else "CAP not available; ") + \
              (f"High early>late p={float(g1.p.iloc[0]):.2g}; " if len(g1) else "") + \
              (f"grade~expert-vote rho={float(g2.median_diff.iloc[0]):.3f} p={float(g2.p.iloc[0]):.2g}"
               if len(g2) else "")
        rows.append(("RQ2b", CRITERIA["RQ2b"], obs, _status(ok3 and ok1 and ok2, ok3)))
    df = pd.DataFrame(rows, columns=["research_question", "prespecified_criterion",
                                     "observed", "status"])
    df.to_csv(os.path.join(out_dir, "research_question_results.csv"), index=False)
    return df
