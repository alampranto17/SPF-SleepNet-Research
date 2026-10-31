"""Sensitivity analyses for every research question (added in v2.1).

None of these analyses changes a pre-specified verdict.  Each one re-computes a
primary quantity under one alternative, defensible analysis choice and records
whether the conclusion of the primary analysis still holds.  Everything is
written to ``sensitivity_*.csv`` and summarised in ``sensitivity_matrix.csv``
(one row per RQ x analysis x setting, with ``holds`` = True / False / NaN for
descriptive rows).

  RQ1  decision threshold; reference standard (consensus, individual experts,
       unanimous epochs, lenient >=1 expert); QC-clean epochs in every cohort;
       participants with too few N3 epochs for a stable F1; error structure of
       false positives (expert stage, expert vote fraction).
  RQ2  homeostatic decline on EXPERT N3 epochs (model-independent); first vs last
       third of the night instead of halves; QC-clean epochs; per cohort.
  RQ2b G3 on QC-clean epochs; exploratory cross-fitted High-threshold calibration
       on CAP participant halves (never a verdict, hypothesis-generating only).
  RQ3  weight perturbation, alternative weightings, leave-one-factor-out.
  RQ4  leave-one-component-out from the full model against the seed-noise floor
       (computed in analysis.rq4_extended).
  RQ5  explanation robustness (computed in xai_ext.rq5_extended).
  RQ6  error-detection AUROC per cohort; reliability score without the
       Mahalanobis (OOD) component.
  RQ7  episode definition (minimum episode length 1/2/3 epochs, gap tolerance
       0/2/4 epochs); per training cohort; QC-clean epochs.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score

from . import scores as SC
from . import stats as ST

MIN_N3_EPOCHS = 10        # participants with fewer expert-N3 epochs have an unstable F1


class _Matrix:
    def __init__(self):
        self.rows = []

    def add(self, rq, analysis, setting, value, primary=np.nan, holds=np.nan, note=""):
        self.rows.append({"research_question": rq, "analysis": analysis, "setting": setting,
                          "value": value, "primary_value": primary, "holds": holds, "note": note})

    def frame(self):
        return pd.DataFrame(self.rows)


def _f1(y, pred):
    return float(f1_score(y, pred, zero_division=0)) if len(y) else np.nan


def _sets(oof, ext):
    """(name, frame, is_external) for every evaluation set of RQ1."""
    s = oof[(oof.label >= 0) & oof.is_primary]
    out = [("pooled OOF", s, False)] + [(f"OOF {c}", g, False) for c, g in s.groupby("cohort")]
    if ext is not None and len(ext):
        e = ext[ext.label >= 0]
        out += [(f"external {c}", g, True) for c, g in e.groupby("cohort")]
    return out


# ---------------------------------------------------------------------------
# RQ1
# ---------------------------------------------------------------------------
def rq1_sensitivity(oof, ext, corpus, out_dir, M):
    rows = []
    for name, g, _ in _sets(oof, ext):
        y, p, thr = g.y.to_numpy(), g.prob.to_numpy(), g.threshold.to_numpy()
        base = _f1(y, g.pred.to_numpy())
        settings = {
            "inner-validation threshold (primary)": g.pred.to_numpy(),
            "fixed threshold 0.5": (p >= 0.5).astype(int),
            "inner threshold - 0.10": (p >= np.clip(thr - 0.10, 0, 1)).astype(int),
            "inner threshold + 0.10": (p >= np.clip(thr + 0.10, 0, 1)).astype(int),
        }
        if "prob_cal" in g:
            settings["temperature-scaled probability >= 0.5"] = (g.prob_cal.to_numpy() >= 0.5).astype(int)
        for k, pr in settings.items():
            rows.append({"set": name, "analysis": "decision threshold", "setting": k, "f1": _f1(y, pr)})
        # upper bound only: threshold chosen on the evaluated labels themselves
        if y.any() and not y.all():
            grid = np.linspace(0.05, 0.95, 91)
            best = max(grid, key=lambda t: _f1(y, (p >= t).astype(int)))
            rows.append({"set": name, "analysis": "decision threshold",
                         "setting": f"label-optimised threshold {best:.2f} (UPPER BOUND, not an estimate)",
                         "f1": _f1(y, (p >= best).astype(int))})
        # QC-clean epochs (v1 artefact / low-quality / clipping flags)
        if "flag_any_v1" in g:
            c = g[~g.flag_any_v1.astype(bool)]
            rows.append({"set": name, "analysis": "QC", "setting": "QC-clean epochs only",
                         "f1": _f1(c.y.to_numpy(), c.pred.to_numpy()), "n_epochs": len(c)})
        # participant-level: stable-F1 participants only
        pp = []
        for _, gp in g.groupby("participant"):
            if gp.y.sum() >= MIN_N3_EPOCHS:
                pp.append(_f1(gp.y.to_numpy(), gp.pred.to_numpy()))
        if pp:
            rows.append({"set": name, "analysis": "participants",
                         "setting": f"median participant F1 (>= {MIN_N3_EPOCHS} expert-N3 epochs)",
                         "f1": float(np.median(pp)), "n_participants": len(pp)})
            rows.append({"set": name, "analysis": "participants",
                         "setting": "share of participants with F1 >= 0.70", "f1": float(np.mean(np.array(pp) >= 0.70)),
                         "n_participants": len(pp)})
        M.add("RQ1", "decision threshold", f"{name}: fixed 0.5",
              _f1(y, (p >= 0.5).astype(int)), base,
              holds=(_f1(y, (p >= 0.5).astype(int)) >= 0.70) == (base >= 0.70))
    # reference standard (multi-rater cohorts)
    ref_rows = []
    for name, g, _ in _sets(oof, ext):
        if name == "pooled OOF":
            continue
        g = g[(g.n_raters_epoch >= 2) & g.n3_vote_frac.notna()]
        if not len(g):
            continue
        pr = g.pred.to_numpy()
        base = _f1(g.y.to_numpy(), pr)
        una = g[(g.n3_vote_frac == 0) | (g.n3_vote_frac == 1)]
        lenient = (g.n3_vote_frac > 0).astype(int).to_numpy()
        strict = (g.n3_vote_frac == 1).astype(int).to_numpy()
        ind = []
        for rec, gr in g.groupby("rec"):
            lm = corpus.labels_multi.get(rec)
            if lm is not None:
                ind.append((lm[gr.epoch_idx.to_numpy()], gr.pred.to_numpy()))
        per_rater = []
        if ind:
            L = np.concatenate([a for a, _ in ind])
            P = np.concatenate([b for _, b in ind])
            for j in range(L.shape[1]):
                m = L[:, j] >= 0
                if m.any():
                    per_rater.append(_f1((L[m, j] == 3).astype(int), P[m]))
        for k, val, n in (("consensus label (primary)", base, len(g)),
                          ("unanimous epochs only (all experts agree)",
                           _f1(una.y.to_numpy(), una.pred.to_numpy()), len(una)),
                          ("lenient: N3 if >= 1 expert scored N3", _f1(lenient, pr), len(g)),
                          ("strict: N3 only if every expert scored N3", _f1(strict, pr), len(g)),
                          ("individual experts, mean", float(np.mean(per_rater)) if per_rater else np.nan, len(g)),
                          ("individual experts, best", float(np.max(per_rater)) if per_rater else np.nan, len(g))):
            ref_rows.append({"set": name, "analysis": "reference standard", "setting": k, "f1": val,
                             "n_epochs": n, "share_of_epochs_unanimous": len(una) / max(len(g), 1)})
            if k != "consensus label (primary)":
                M.add("RQ1", "reference standard", f"{name}: {k}", val, base,
                      holds=bool(val >= 0.70) if val == val else np.nan,
                      note="criterion F1 >= 0.70 re-applied under this reference")
    # error structure of false positives
    err = []
    for name, g, _ in _sets(oof, ext):
        fp = g[(g.pred == 1) & (g.y == 0)]
        if not len(fp):
            continue
        r = {"set": name, "n_false_positive": len(fp),
             "fp_share_of_predicted_n3": len(fp) / max(int((g.pred == 1).sum()), 1)}
        for st, lab in ((2, "N2"), (1, "N1"), (4, "REM"), (0, "Wake")):
            r[f"fp_expert_{lab}"] = float((fp.label == st).mean())
        mv = fp[fp.n_raters_epoch >= 2]
        if len(mv):
            r["fp_with_>=1_expert_N3_vote"] = float((mv.n3_vote_frac > 0).mean())
            r["fp_median_expert_N3_vote_fraction"] = float(mv.n3_vote_frac.median())
        if "sw_time_frac" in g:
            r["fp_median_sw_time_frac"] = float(fp.sw_time_frac.median())
            r["tp_median_sw_time_frac"] = float(g[(g.pred == 1) & (g.y == 1)].sw_time_frac.median())
        err.append(r)
    pd.DataFrame(rows + ref_rows).to_csv(os.path.join(out_dir, "sensitivity_rq1.csv"), index=False)
    pd.DataFrame(err).to_csv(os.path.join(out_dir, "sensitivity_rq1_false_positive_structure.csv"),
                             index=False)
    for r in rows:
        if r["analysis"] in ("QC", "participants") and "share" not in r["setting"]:
            M.add("RQ1", r["analysis"], f"{r['set']}: {r['setting']}", r["f1"], np.nan,
                  holds=bool(r["f1"] >= 0.70) if r["f1"] == r["f1"] else np.nan)


# ---------------------------------------------------------------------------
# RQ2
# ---------------------------------------------------------------------------
def _part(df, mode):
    n = df.groupby("rec").epoch_idx.transform("max") + 1
    if mode == "halves":
        return np.where(df.epoch_idx < n / 2, "early", "late")
    return np.where(df.epoch_idx < n / 3, "early", np.where(df.epoch_idx >= 2 * n / 3, "late", "mid"))


def _decline(d, fac, sel, mode):
    g = d[sel].copy()
    g["part"] = _part(g, mode)
    g = g[g.part != "mid"]
    pm = g.groupby(["participant", "part"])[fac].agg(["median", "size"]).unstack()
    if "early" not in pm["size"] or "late" not in pm["size"]:
        return None
    pm = pm[(pm["size"]["early"] >= 10) & (pm["size"]["late"] >= 10)]
    if len(pm) < 3:
        return None
    return ST.wilcoxon_paired(pm["median"]["early"], pm["median"]["late"], "greater")


def rq2_sensitivity(oof, S, ext, Sx, out_dir, M):
    d = oof.join(S)
    d = d[d.is_primary]
    parts = [("OOF", d)]
    if ext is not None and Sx is not None and len(ext):
        parts.append(("external", ext.join(Sx)))
    rows = []
    for split, dd in parts:
        for fac in SC.FACTORS + ["rating_prior"]:
            col = fac
            if fac == "n3_continuity":
                expert_col = "n3_continuity_expert"
            else:
                expert_col = fac
            settings = [("detected N3, halves (primary)", dd.pred == 1, "halves", col),
                        ("EXPERT N3 epochs (model-independent), halves", dd.y == 1, "halves", expert_col),
                        ("detected N3, first vs last third", dd.pred == 1, "thirds", col)]
            if "flag_any_v1" in dd:
                settings.append(("detected N3, QC-clean epochs, halves",
                                 (dd.pred == 1) & ~dd.flag_any_v1.astype(bool), "halves", col))
            if fac == "rating_prior":
                rated = dd.assign(rating_expert=SC.overall_rating(
                    dd[SC.FACTORS[:3]].assign(n3_continuity=dd.n3_continuity_expert)))
                settings[1] = ("EXPERT N3 epochs (model-independent), halves", dd.y == 1, "halves",
                               "rating_expert")
                dd = rated
            prim = None
            for k, sel, mode, c in settings:
                if c not in dd:
                    continue
                w = _decline(dd, c, sel, mode)
                if w is None:
                    continue
                rows.append({"split": split, "factor": fac, "setting": k, **w})
                if prim is None:
                    prim = w
                else:
                    M.add("RQ2", "early > late-night decline", f"{split} {fac}: {k}", w["median_diff"],
                          prim["median_diff"], holds=bool(w["p"] < 0.05 and w["median_diff"] > 0),
                          note=f"p={w['p']:.2g}")
            # per cohort (detected N3, halves)
            for coh, gc in dd.groupby("cohort"):
                w = _decline(gc, col, gc.pred == 1, "halves")
                if w is not None:
                    rows.append({"split": split, "factor": fac, "setting": f"cohort {coh}", **w})
                    M.add("RQ2", "early > late-night decline", f"{split} {fac}: cohort {coh}",
                          w["median_diff"], prim["median_diff"] if prim else np.nan,
                          holds=bool(w["p"] < 0.05 and w["median_diff"] > 0) if w["p"] == w["p"] else np.nan,
                          note=f"p={w['p']:.2g}, n={w['n']}")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "sensitivity_rq2.csv"), index=False)


# ---------------------------------------------------------------------------
# RQ2b
# ---------------------------------------------------------------------------
def rq2b_sensitivity(out_dir, M, seed=0):
    p = os.path.join(out_dir, "n3_grades_detected_epochs.parquet")
    if not os.path.exists(p):
        return
    d = pd.read_parquet(p)
    if "rk_code" not in d:
        return
    cap = d[d.rk_code.isin([3, 4]) & d.sw_time_frac.notna()].copy()
    if cap.rk_code.nunique() < 2:
        return
    rows = []
    y4 = (cap.rk_code == 4).astype(int)
    rows.append({"analysis": "G3 AUROC", "setting": "detected N3 (primary)",
                 "value": roc_auc_score(y4, cap.sw_time_frac)})
    # participant-level AUROC: guards against pooling participants with different S4 share
    au = [roc_auc_score((g.rk_code == 4).astype(int), g.sw_time_frac)
          for _, g in cap.groupby("participant") if g.rk_code.nunique() == 2]
    if au:
        rows.append({"analysis": "G3 AUROC", "setting": "median per-participant AUROC",
                     "value": float(np.median(au)), "n_participants": len(au)})
    # exploratory: cross-fitted threshold for 'High' on participant halves
    parts = np.array(sorted(cap.participant.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(parts)
    halves = [set(parts[: len(parts) // 2]), set(parts[len(parts) // 2:])]
    preds = pd.Series(np.nan, index=cap.index)
    thr_used = []
    for k in (0, 1):
        fit = cap[cap.participant.isin(halves[k])]
        app = cap[cap.participant.isin(halves[1 - k])]
        yf = (fit.rk_code == 4).astype(int).to_numpy()
        grid = np.unique(np.round(fit.sw_time_frac.to_numpy(), 3))
        if len(grid) > 200:
            grid = np.quantile(fit.sw_time_frac, np.linspace(0.02, 0.98, 97))
        youden = [(yf[fit.sw_time_frac >= t].sum() / max(yf.sum(), 1)) -
                  (((1 - yf)[fit.sw_time_frac.to_numpy() >= t]).sum() / max((1 - yf).sum(), 1)) for t in grid]
        t = float(grid[int(np.argmax(youden))])
        thr_used.append(t)
        preds.loc[app.index] = (app.sw_time_frac >= t).astype(int)
    m = preds.notna()
    hi = preds[m].astype(int)
    yy = y4[m]
    rows += [{"analysis": "EXPLORATORY cross-fitted High threshold", "setting": "thresholds (half A, half B)",
              "value": np.nan, "note": f"{thr_used[0]:.3f}, {thr_used[1]:.3f} (manual anchor 0.50)"},
             {"analysis": "EXPLORATORY cross-fitted High threshold", "setting": "kappa High vs S4",
              "value": float(cohen_kappa_score(yy, hi))},
             {"analysis": "EXPLORATORY cross-fitted High threshold", "setting": "sensitivity High for S4",
              "value": float(hi[yy == 1].mean())},
             {"analysis": "EXPLORATORY cross-fitted High threshold", "setting": "specificity not-High for S3",
              "value": float(1 - hi[yy == 0].mean())}]
    # alternative grading variables (AUROC, threshold-free; exploratory)
    for col in ("rating_prior",):
        if col in cap and cap[col].notna().any():
            mm = cap[col].notna()
            rows.append({"analysis": "EXPLORATORY alternative grading variable", "setting": col,
                         "value": float(roc_auc_score(y4[mm], cap.loc[mm, col]))})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "sensitivity_rq2b.csv"), index=False)
    for r in rows:
        if r["setting"] == "median per-participant AUROC":
            M.add("RQ2b", "G3 S4-vs-S3 AUROC", "median per-participant AUROC", r["value"],
                  rows[0]["value"], holds=bool(r["value"] >= 0.75))
        elif r["analysis"].startswith("EXPLORATORY") and r["value"] == r["value"]:
            M.add("RQ2b", r["analysis"], r["setting"], r["value"], note="exploratory, not a verdict")


# ---------------------------------------------------------------------------
# RQ3 (collects the existing robustness outputs into the matrix)
# ---------------------------------------------------------------------------
def rq3_sensitivity(out_dir, M):
    import json
    p = os.path.join(out_dir, "rq3_overall_rating_stability.json")
    if not os.path.exists(p):
        return
    st = json.load(open(p))
    prim = st.get("weight_sensitivity", {}).get("reasonable(alpha=20*prior)", {}).get("p05_rho", np.nan)
    for k, v in st.get("weight_sensitivity", {}).items():
        M.add("RQ3", "weight perturbation", k, v.get("p05_rho"), prim,
              holds=bool(v.get("p05_rho", 0) >= 0.90), note="5th-percentile Spearman rho")
    for k, v in st.get("missing_factor_rho", {}).items():
        M.add("RQ3", "leave one factor out", k, v, prim, holds=bool(v >= 0.90))
    for k, v in st.get("alternative_weightings", {}).items():
        if not isinstance(v, dict):
            M.add("RQ3", "alternative weighting", k, v, prim, holds=bool(v >= 0.90))
    if "split_half_spearman_brown" in st:
        M.add("RQ3", "split-half reliability", "Spearman-Brown", st["split_half_spearman_brown"],
              note="descriptive")


# ---------------------------------------------------------------------------
# RQ6
# ---------------------------------------------------------------------------
def rq6_sensitivity(oof, ext, rel_oof, rel_ext, out_dir, M):
    rows = []
    for split, d, rel in (("OOF", oof, rel_oof), ("external", ext, rel_ext)):
        if d is None or rel is None or not len(d):
            continue
        g = d[d.label >= 0].merge(rel, on="idx", how="inner")
        g["error"] = (g.pred != g.y).astype(int)
        for coh, gc in [("all", g)] + list(g.groupby("cohort")):
            if gc.error.nunique() < 2:
                continue
            auc = float(roc_auc_score(gc.error, -gc.reliability))
            rows.append({"split": split, "cohort": coh, "setting": "combined reliability score",
                         "error_auroc": auc, "n": len(gc)})
            M.add("RQ6", "error-detection AUROC", f"{split} {coh}", auc, holds=bool(auc >= 0.75))
            if "maha" in gc:
                p = gc.prob_cal if "prob_cal" in gc else gc.prob
                conf = np.maximum(p, 1 - p)
                a2 = float(roc_auc_score(gc.error, -conf))
                rows.append({"split": split, "cohort": coh, "setting": "model confidence only (no OOD, no QC)",
                             "error_auroc": a2, "n": len(gc)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "sensitivity_rq6.csv"), index=False)


# ---------------------------------------------------------------------------
# RQ7
# ---------------------------------------------------------------------------
def rq7_sensitivity(oof, ext, out_dir, M):
    rows = []
    sets = [("OOF", oof[oof.is_primary & (oof.label >= 0)])]
    if ext is not None and len(ext):
        sets.append(("external", ext[ext.label >= 0]))
    for split, d in sets:
        groups = [("all", d)] + list(d.groupby("cohort"))
        for coh, g in groups:
            for min_ep in (1, 2, 3):
                for gap in (0, 2, 4):
                    a, b = [], []
                    for _, gr in g.groupby("rec"):
                        gr = gr.sort_values("epoch_idx")
                        ep_p = SC.find_episodes(gr.pred.to_numpy(), min_epochs=min_ep, gap_tol=gap)
                        ep_e = SC.find_episodes(gr.y.to_numpy(), min_epochs=min_ep, gap_tol=gap)
                        a.append((max([e[2] for e in ep_p]) if ep_p else 0) * 0.5)
                        b.append((max([e[2] for e in ep_e]) if ep_e else 0) * 0.5)
                    if len(a) >= 5:
                        icc = ST.icc(np.array(a), np.array(b))
                        rows.append({"split": split, "cohort": coh, "metric": "longest_episode_min",
                                     "min_epochs": min_ep, "gap_tol": gap, "icc21": icc, "n": len(a)})
                        if coh == "all" and not (min_ep == 2 and gap == 2):
                            M.add("RQ7", "episode definition",
                                  f"{split}: longest episode, min {min_ep} epochs, gap {gap}", icc,
                                  holds=bool(icc >= 0.75) if icc == icc else np.nan,
                                  note="primary definition: min 2, gap 2")
            # N3 minutes per night: all epochs vs QC-clean epochs
            for k, gg in (("all scored epochs", g),
                          ("QC-clean epochs", g[~g.flag_any_v1.astype(bool)] if "flag_any_v1" in g else None)):
                if gg is None:
                    continue
                pm = gg.groupby("rec").agg(p=("pred", "sum"), e=("y", "sum")) * 0.5
                if len(pm) >= 5:
                    icc = ST.icc(pm.p.to_numpy(), pm.e.to_numpy())
                    rows.append({"split": split, "cohort": coh, "metric": f"n3_minutes [{k}]",
                                 "icc21": icc, "n": len(pm)})
                    if k != "all scored epochs" or coh != "all":
                        M.add("RQ7", "N3 minutes ICC(2,1)", f"{split} {coh}: {k}", icc,
                              holds=bool(icc >= 0.75) if icc == icc else np.nan)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "sensitivity_rq7.csv"), index=False)


# ---------------------------------------------------------------------------
# RQ4 / RQ5 (their sensitivity outputs are written by analysis.rq4_extended and
# xai_ext.rq5_extended; this block only adds them to the matrix)
# ---------------------------------------------------------------------------
def rq4_rq5_matrix(out_dir, M):
    p = os.path.join(out_dir, "rq4_extended_component_effects.csv")
    if os.path.exists(p):
        for _, r in pd.read_csv(p).iterrows():
            if r.metric != "f1":
                continue
            M.add("RQ4", "leave-one-component-out from full model", r.component, r.median_diff,
                  holds=bool(r.p_holm < 0.05 and r.exceeds_seed_noise),
                  note=f"p_holm={r.p_holm:.2g}; seed-noise floor={r.seed_noise_floor:.3f}")
    p = os.path.join(out_dir, "rq4_seed_variability.csv")
    if os.path.exists(p):
        sv = pd.read_csv(p).set_index("variant")
        if "SD across seeds" in sv.index:
            M.add("RQ4", "training-seed variability", "SD of pooled F1 across 3 seeds",
                  sv.loc["SD across seeds", "f1"], note="descriptive")
    p = os.path.join(out_dir, "rq5x_sensitivity.csv")
    if os.path.exists(p):
        for _, r in pd.read_csv(p).iterrows():
            M.add("RQ5", r.analysis, r.setting, r.value, r.get("primary_value", np.nan),
                  holds=r.holds if r.holds == r.holds else np.nan, note=r.get("note", ""))


# ---------------------------------------------------------------------------
def run_all(corpus, oof, ext, S, Sx, out_dir, rel_oof=None, rel_ext=None, log=print):
    """Run every block; a failing block is logged and the others continue."""
    import traceback
    M = _Matrix()
    if "sw_time_frac" in corpus.feat:
        oof = oof.assign(sw_time_frac=corpus.feat.loc[oof.idx.to_numpy(), "sw_time_frac"].to_numpy())
        if ext is not None and len(ext):
            ext = ext.assign(sw_time_frac=corpus.feat.loc[ext.idx.to_numpy(), "sw_time_frac"].to_numpy())
    blocks = [("RQ1", lambda: rq1_sensitivity(oof, ext, corpus, out_dir, M)),
              ("RQ2", lambda: rq2_sensitivity(oof, S, ext, Sx, out_dir, M)),
              ("RQ2b", lambda: rq2b_sensitivity(out_dir, M)),
              ("RQ3", lambda: rq3_sensitivity(out_dir, M)),
              ("RQ6", lambda: rq6_sensitivity(oof, ext, rel_oof, rel_ext, out_dir, M)),
              ("RQ7", lambda: rq7_sensitivity(oof, ext, out_dir, M)),
              ("RQ4/RQ5", lambda: rq4_rq5_matrix(out_dir, M))]
    for name, fn in blocks:
        try:
            fn()
        except Exception as exc:
            log(f"[sensitivity {name}] FAILED: {type(exc).__name__}: {exc}")
            with open(os.path.join(out_dir, "analysis_errors.txt"), "a") as f:
                f.write(f"=== sensitivity {name}\n{traceback.format_exc()}\n")
    mat = M.frame()
    mat.to_csv(os.path.join(out_dir, "sensitivity_matrix.csv"), index=False)
    return mat
