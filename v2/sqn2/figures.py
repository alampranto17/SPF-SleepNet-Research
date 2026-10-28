"""Publication figures (300 dpi PNG) generated from the saved analysis outputs.

Palette: validated categorical slots 1-3 (blue / orange / aqua) on a white
surface; text stays in neutral ink; one y-axis per panel; legends for >= 2
series.  Every figure is built only from files written by analysis.py / xai.py.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# validated categorical slots 1-6 (fixed order, never cycled)
C1, C2, C3, C4, C5, C6 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COHORT_COLOR = {"ISRUC-I": C1, "HMC": C2, "ISRUC-III": C3, "ISRUC-II": C4, "DOD-H": C5, "DOD-O": C6,
                "CAP": INK2}   # 7th cohort: neutral (grading reference only)
GRADE_COLOR = {"Low": "#b9d4f3", "Moderate": "#5d9ce3", "High": "#1a4f91"}   # ordinal, one hue
FACTOR_COLOR = {"delta_intensity": C1, "sw_morphology": C2, "spectral_dominance": C3,
                "rating_prior": C4}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, out_dir, name):
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    p = os.path.join(out_dir, "figures", name)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_pr_curves(out_dir):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    oof = pd.read_parquet(os.path.join(out_dir, "oof_predictions_full_v2.parquet"))
    oof = oof[(oof.label >= 0) & oof.is_primary]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    sets = [(c, g) for c, g in oof.groupby("cohort")]
    ep = os.path.join(out_dir, "external_predictions.parquet")
    if os.path.exists(ep):
        e = pd.read_parquet(ep)
        sets += [(c, g) for c, g in e[e.label >= 0].groupby("cohort")]
    for c, g in sets:
        p, r, _ = precision_recall_curve(g.y, g.prob)
        ax.plot(r, p, color=COHORT_COLOR.get(c, INK2),
                label=f"{c} (AP={average_precision_score(g.y, g.prob):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("N3 detection (OOF: ISRUC-I, HMC; others external)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0, 1)
    lo = float(np.nanmin([np.nanmin(v) for v in vals if len(v)] or [0]))
    ax.set_ylim(max(0.0, np.floor((lo - 0.05) * 10) / 10), 1.02)
    return _save(fig, out_dir, "fig_rq1_pr_curves.png")


def fig_homeostatic(out_dir):
    S = pd.read_parquet(os.path.join(out_dir, "factor_scores.parquet"))
    S = S[S.split == "OOF"]
    S = S[S.pred == 1]
    n = S.groupby("rec").epoch_idx.transform("max") + 1
    S["half"] = np.where(S.epoch_idx < n / 2, "early", "late")
    facs = ["delta_intensity", "sw_morphology", "spectral_dominance", "n3_continuity", "rating_prior"]
    titles = ["Delta intensity", "SW morphology", "Spectral dominance", "Continuity", "Overall rating"]
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.8), sharey=True)
    for ax, f, t in zip(axes, facs, titles):
        pm = S.groupby(["participant", "half"])[f].median().unstack().dropna()
        for _, r in pm.iterrows():
            ax.plot([0, 1], [r.early, r.late], color=GRID, lw=0.8, zorder=1)
        ax.boxplot([pm.early, pm.late], positions=[0, 1], widths=0.45, showfliers=False,
                   medianprops=dict(color=C1, lw=2), boxprops=dict(color=INK2),
                   whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
        ax.set_xticks([0, 1], ["1st half", "2nd half"])
        ax.set_title(t)
    axes[0].set_ylabel("Participant median score (0-100)")
    fig.suptitle("Detected-N3 factor scores: early vs late night (homeostatic decline test)",
                 fontsize=10, y=1.03)
    return _save(fig, out_dir, "fig_rq2_early_vs_late.png")


def fig_ablation(out_dir):
    comp = pd.read_csv(os.path.join(out_dir, "rq4_component_effects.csv"))
    comp = comp[comp.metric == "f1"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.2, 0.45 * len(comp) + 1))
    y = np.arange(len(comp))[::-1]
    ax.axvline(0, color=INK2, lw=1)
    ax.errorbar(comp.median_diff, y, xerr=[comp.median_diff - comp.ci95_low,
                                          comp.ci95_high - comp.median_diff],
                fmt="o", color=C1, ms=6, capsize=3, lw=1.5)
    ax.set_yticks(y, comp.component)
    ax.set_xlabel("Median paired difference in participant F1 (95% bootstrap CI)")
    ax.set_title("Component contributions (identical folds)")
    return _save(fig, out_dir, "fig_rq4_component_forest.png")


def fig_risk_coverage(out_dir):
    rc = pd.read_csv(os.path.join(out_dir, "rq6_risk_coverage.csv"))
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot(rc.coverage * 100, rc.f1, "-o", color=C1, ms=6, label="F1")
    ax.invert_xaxis()
    ax.set_xlabel("Coverage: epochs retained by reliability (%)")
    ax.set_ylabel("N3 F1 on retained epochs")
    ax.set_title("Selective prediction")
    return _save(fig, out_dir, "fig_rq6_risk_coverage.png")


def fig_profiles(out_dir):
    prof = pd.read_csv(os.path.join(out_dir, "rq7_overnight_n3_eeg_profiles.csv"))
    pr = prof[prof.is_primary & (prof.split == "OOF")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for c, g in pr.groupby("cohort"):
        axes[0].scatter(g.expert_n3_minutes, g.pred_n3_minutes, s=18, color=COHORT_COLOR.get(c, INK2),
                        label=c, edgecolor="white", linewidth=0.5)
    m = np.nanmax([pr.expert_n3_minutes.max(), pr.pred_n3_minutes.max()]) * 1.05
    axes[0].plot([0, m], [0, m], color=INK2, lw=1, ls="--")
    axes[0].set_xlabel("Expert N3 minutes")
    axes[0].set_ylabel("Predicted N3 minutes")
    axes[0].set_title("Overnight N3 duration")
    axes[0].legend(fontsize=8)
    multi = prof.groupby("participant").rec.transform("nunique") >= 2
    tp = prof[multi].sort_values(["participant", "rec"])
    for metric, color, lab in (("pred_median_rating", C1, "Overall N3 EEG Rating"),
                               ("pred_n3_minutes", C2, "N3 minutes")):
        if not len(tp):
            break
        a = tp.groupby("participant")[metric].nth(0).to_numpy()
        b = tp.groupby("participant")[metric].nth(1).to_numpy()
        both = np.concatenate([a, b])
        mu, sd = np.nanmean(both), np.nanstd(both) + 1e-9
        axes[1].scatter((a - mu) / sd, (b - mu) / sd, s=18, color=color, label=lab,
                        edgecolor="white", linewidth=0.5)
    axes[1].plot([-3, 3], [-3, 3], color=INK2, lw=1, ls="--")
    axes[1].set_xlabel("Session 1 (z-score)")
    axes[1].set_ylabel("Session 2 (z-score)")
    axes[1].set_title("Test-retest (ISRUC-II)")
    axes[1].legend(fontsize=8)
    return _save(fig, out_dir, "fig_rq7_profiles.png")


def fig_example_night(out_dir, participant=None):
    S = pd.read_parquet(os.path.join(out_dir, "factor_scores.parquet"))
    S = S[S.split == "OOF"]
    oof = pd.read_parquet(os.path.join(out_dir, "oof_predictions_full_v2.parquet"))
    d = oof[oof.is_primary].merge(S[["idx", "rating_prior"]], on="idx", how="left")
    rec = (d[d.participant == participant].rec.iloc[0] if participant else
           d.groupby("rec").y.sum().sort_values().index[len(d.rec.unique()) // 2])
    g = d[d.rec == rec].sort_values("epoch_idx")
    h = g.epoch_idx / 120.0
    fig, axes = plt.subplots(3, 1, figsize=(8, 4.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.1, 1, 1]))
    stage_y = g.label.map({0: 4, 4: 3, 1: 2, 2: 1, 3: 0}).astype(float)
    axes[0].step(h, stage_y, where="post", color=INK2, lw=1)
    axes[0].set_yticks([0, 1, 2, 3, 4], ["N3", "N2", "N1", "REM", "W"])
    axes[0].set_title(f"Expert hypnogram - {rec}")
    axes[1].plot(h, g.prob, color=C1, lw=1)
    axes[1].axhline(g.threshold.iloc[0], color=INK2, lw=0.8, ls="--")
    axes[1].set_ylabel("P(N3)")
    r = np.where(g.pred == 1, g.rating_prior, np.nan)
    axes[2].plot(h, r, color=C2, lw=1.2)
    axes[2].set_ylabel("N3 EEG Rating")
    axes[2].set_ylim(0, 100)
    axes[2].set_xlabel("Hours from start of analysis window")
    return _save(fig, out_dir, "fig_example_overnight_profile.png")


def fig_xai(out_dir):
    X = pd.read_csv(os.path.join(out_dir, "rq5_epoch_explanations.csv"))
    bands = ["delta", "theta", "alpha", "sigma", "beta"]
    pm = X.groupby("participant")[[f"drop_{b}" for b in bands] + [f"ig_{b}" for b in bands]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, pre, title, lab in ((axes[0], "drop_", "Band-stop perturbation", "Drop in P(N3)"),
                                (axes[1], "ig_", "Band-decomposed integrated gradients",
                                 "Signed share of attribution")):
        vals = [pm[f"{pre}{b}"] for b in bands]
        ax.boxplot(vals, widths=0.5, showfliers=False, medianprops=dict(color=C1, lw=2),
                   boxprops=dict(color=INK2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
        ax.set_xticks(range(1, 6), bands)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(title)
        ax.set_ylabel(lab)
    return _save(fig, out_dir, "fig_rq5_band_explanations.png")


def fig_graded_agreement(out_dir):
    """Factor scores vs the fraction of experts who scored the epoch N3
    (epochs where every expert said N2 or N3)."""
    S = pd.read_parquet(os.path.join(out_dir, "factor_scores.parquet"))
    S = S[S.n3_vote_frac.notna()]
    facs = ["delta_intensity", "sw_morphology", "spectral_dominance", "rating_prior"]
    names = {"delta_intensity": "Delta intensity", "sw_morphology": "SW morphology",
             "spectral_dominance": "Spectral dominance", "rating_prior": "Overall rating"}
    cohorts = [c for c in ("DOD-H", "DOD-O", "ISRUC-I", "ISRUC-II", "ISRUC-III")
               if c in set(S.cohort)]
    fig, axes = plt.subplots(1, max(1, len(cohorts)), figsize=(2.6 * max(1, len(cohorts)), 2.9),
                             sharey=True, squeeze=False)
    for ax, c in zip(axes[0], cohorts):
        g = S[(S.cohort == c) & S.label.isin([2, 3])]
        g = g.assign(vote=g.n3_vote_frac.round(2))
        for f in facs:
            m = g.groupby("vote")[f].agg(["mean", "sem", "count"])
            m = m[m["count"] >= 20]
            ax.errorbar(m.index, m["mean"], yerr=1.96 * m["sem"], color=FACTOR_COLOR[f],
                        marker="o", ms=4, lw=1.6, capsize=2, label=names[f])
        ax.set_title(c)
        ax.set_xlabel("Fraction of experts scoring N3")
    axes[0][0].set_ylabel("Mean score (0-100), 95% CI")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Graded expert agreement (N2/N3 epochs)", fontsize=10, y=1.03)
    return _save(fig, out_dir, "fig_rq2_graded_expert_agreement.png")


def fig_sota(out_dir):
    t = pd.read_csv(os.path.join(out_dir, "rq1_sota_comparison.csv"))
    t = t[t["set"].str.startswith("cross-validation")].sort_values("f1")
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(t) + 1))
    y = np.arange(len(t))
    lo = [c[0] if isinstance(c, (list, tuple)) else np.nan for c in t.get("f1_ci95", [np.nan] * len(t))]
    col = [C1 if m == "full_v2" else INK2 for m in t.model]
    ax.barh(y, t.f1, color=col, height=0.6)
    for yi, (f, m) in enumerate(zip(t.f1, t.model)):
        ax.text(f + 0.005, yi, f"{f:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [("SleepQualityNet v2" if m == "full_v2" else m) for m in t.model])
    ax.set_xlim(max(0, t.f1.min() - 0.1), min(1, t.f1.max() + 0.06))
    ax.set_xlabel("Pooled out-of-fold N3 F1 (identical folds and protocol)")
    ax.set_title("Same-protocol comparison with published architectures")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "fig_rq1_sota.png")


def fig_n3_grade(out_dir):
    names = ["Low", "Moderate", "High"]
    panels = []
    p1 = os.path.join(out_dir, "rq2_rk_crosstab_detected.csv")
    if not os.path.exists(p1):
        p1 = os.path.join(out_dir, "rq2_rk_crosstab_all.csv")
    if os.path.exists(p1):
        ct = pd.read_csv(p1, index_col=0)
        ct = ct.reindex(columns=[n for n in names if n in ct.columns]).fillna(0)
        panels.append(("Expert R&K stage (CAP)", ct.div(ct.sum(1), axis=0)))
    p2 = os.path.join(out_dir, "rq2_sw_grade_by_expert_vote_fraction.csv")
    if os.path.exists(p2):
        v = pd.read_csv(p2)
        if v.cohort.isin(["DOD-H", "DOD-O"]).any():
            # 5 scorers -> votes are multiples of 20 %; other values come from epochs a scorer left unscored
            v = v[v.cohort.isin(["DOD-H", "DOD-O"]) & np.isclose((v.vote * 5) % 1, 0, atol=1e-6)]
        g = v.groupby("vote")[[n for n in names if n in v.columns]].mean()
        g.index = [f"{x:.0%}" for x in g.index]
        panels.append(("Fraction of DOD experts voting N3", g))
    if not panels:
        raise FileNotFoundError("no grade tables")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2), squeeze=False)
    for ax, (title, tab) in zip(axes[0], panels):
        left = np.zeros(len(tab))
        for n in tab.columns:
            ax.barh(range(len(tab)), tab[n], left=left, color=GRADE_COLOR[n], label=n, height=0.6,
                    edgecolor="white", linewidth=1)
            left += tab[n].to_numpy()
        ax.set_yticks(range(len(tab)), tab.index)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of detected N3 epochs")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    h, l = axes[0][-1].get_legend_handles_labels()
    fig.legend(h, l, title="N3 slow-wave grade", loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.28)
    return _save(fig, out_dir, "fig_rq2b_n3_grade.png")


# ---------------------------------------------------------------------------
# Additional paper figures
# ---------------------------------------------------------------------------
BLUES = ["#eef4fc", "#cfe0f7", "#9fc2ee", "#5d9ce3", "#2a78d6", "#1a4f91"]   # one hue, light -> dark
STAGES = ["W", "N1", "N2", "N3", "REM"]


def _pred_sets(out_dir):
    """[(name, df)] : pooled cross-validation (training cohorts) + each external cohort."""
    sets = []
    oof = pd.read_parquet(os.path.join(out_dir, "oof_predictions_full_v2.parquet"))
    oof = oof[(oof.label >= 0) & oof.is_primary]
    sets.append(("Cross-validation\n(ISRUC-I + HMC)", oof))
    ep = os.path.join(out_dir, "external_predictions.parquet")
    if os.path.exists(ep):
        e = pd.read_parquet(ep)
        e = e[e.label >= 0]
        for c in ("ISRUC-II", "ISRUC-III", "CAP", "DOD-H", "DOD-O"):
            g = e[e.cohort == c]
            if len(g):
                sets.append((c, g))
    return sets


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blues1", BLUES)


def fig_confusion_matrices(out_dir):
    sets = _pred_sets(out_dir)
    n = len(sets)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows), squeeze=False)
    cm_ = _cmap()
    for ax, (name, g) in zip(axes.ravel(), sets):
        y, pr = g.y.to_numpy().astype(int), g.pred.to_numpy().astype(int)
        M = np.array([[((y == a) & (pr == b)).sum() for b in (0, 1)] for a in (0, 1)])
        R = M / np.maximum(M.sum(1, keepdims=True), 1)
        ax.imshow(R, cmap=cm_, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{R[i, j]:.0%}\n(n={M[i, j]:,})", ha="center", va="center", fontsize=8,
                        color="white" if R[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["non-N3", "N3"])
        ax.set_yticks([0, 1], ["non-N3", "N3"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Expert")
        ax.set_title(name, fontsize=9)
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("N3 detection confusion matrices (row-normalised)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq1_confusion_matrices.png")


def fig_stage_confusion(out_dir):
    """Share of each EXPERT stage predicted as N3: where do false positives come from?"""
    sets = _pred_sets(out_dir)
    tab = []
    for name, g in sets:
        row = [(g.pred[g.label == k].mean() if (g.label == k).any() else np.nan) for k in range(5)]
        tab.append(row)
    T = np.array(tab, dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(sets) + 1.4))
    ax.imshow(T, cmap=_cmap(), vmin=0, vmax=1, aspect="auto")
    for i in range(T.shape[0]):
        for j in range(5):
            if not np.isnan(T[i, j]):
                ax.text(j, i, f"{T[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if T[i, j] > 0.55 else INK)
    ax.set_xticks(range(5), STAGES)
    ax.set_yticks(range(len(sets)), [s[0].replace("\n", " ") for s in sets], fontsize=8)
    ax.set_xlabel("Expert sleep stage")
    ax.set_title("Epochs detected as N3, by expert stage", fontsize=10)
    ax.grid(False)
    return _save(fig, out_dir, "fig_rq1_n3_by_expert_stage.png")


def fig_roc_curves(out_dir):
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for name, g in _pred_sets(out_dir):
        if g.y.nunique() < 2:
            continue
        f, t, _ = roc_curve(g.y, g.prob)
        lab = name.split("\n")[0]
        col = C1 if lab.startswith("Cross") else COHORT_COLOR.get(lab, INK2)
        ax.plot(f, t, color=col, lw=1.6, label=f"{lab} (AUC={roc_auc_score(g.y, g.prob):.3f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right", fontsize=7)
    return _save(fig, out_dir, "fig_rq1_roc_curves.png")


def fig_calibration(out_dir, bins=10):
    sets = _pred_sets(out_dir)
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    for name, g in sets:
        col_p = "prob_cal" if "prob_cal" in g else "prob"
        p, y = g[col_p].to_numpy(), g.y.to_numpy()
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        mp = [p[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 50]
        my = [y[idx == b].mean() for b in range(bins) if (idx == b).sum() >= 50]
        ece = sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                  for b in range(bins) if (idx == b).any())
        lab = name.split("\n")[0]
        col = C1 if lab.startswith("Cross") else COHORT_COLOR.get(lab, INK2)
        ax.plot(mp, my, marker="o", ms=4, lw=1.5, color=col, label=f"{lab} (ECE={ece:.3f})")
    ax.set_xlabel("Calibrated N3 probability")
    ax.set_ylabel("Observed N3 fraction")
    ax.set_title("Calibration (temperature-scaled)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save(fig, out_dir, "fig_rq1_calibration.png")


def fig_participant_f1(out_dir):
    from sklearn.metrics import f1_score
    data, labels, cols = [], [], []
    for name, g in _pred_sets(out_dir):
        if name.startswith("Cross"):
            parts = list(g.groupby("cohort"))
        else:
            parts = [(name, g)]
        for lab, gg in parts:
            f = [f1_score(x.y, x.pred, zero_division=0) for _, x in gg.groupby("participant") if x.y.sum() >= 10]
            if f:
                data.append(f)
                labels.append(lab + (" (CV)" if name.startswith("Cross") else ""))
                cols.append(COHORT_COLOR.get(lab, INK2))
    fig, ax = plt.subplots(figsize=(0.75 * len(data) + 1.6, 3.2))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.5))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)
    rng = np.random.default_rng(0)
    for i, (d, c) in enumerate(zip(data, cols), 1):
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=8, color=c, edgecolor="white", linewidth=0.4,
                   zorder=3)
    ax.set_xticks(range(1, len(data) + 1), labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Per-participant N3 F1")
    lo = min(min(d) for d in data) if data else 0.0
    ax.set_ylim(max(0.0, np.floor((lo - 0.05) * 10) / 10), 1.01)
    ax.set_title("N3 F1 per participant (>= 10 expert N3 epochs)")
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "fig_rq1_participant_f1.png")


PRETTY = {"deepsleepnet": "vs DeepSleepNet", "tinysleepnet": "vs TinySleepNet", "attnsleep": "vs AttnSleep",
          "lseqsleepnet": "vs L-SeqSleepNet"}


def fig_sota_forest(out_dir):
    t = pd.read_csv(os.path.join(out_dir, "rq1_sota_paired_tests.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.5 * t.comparison.nunique() + 1.4), sharey=True)
    for ax, metric in zip(axes, ("f1", "pr_auc")):
        g = t[t.metric == metric].reset_index(drop=True)
        y = np.arange(len(g))[::-1]
        ax.axvline(0, color=INK2, lw=1)
        ax.errorbar(g.median_participant_diff, y,
                    xerr=[g.median_participant_diff - g.ci95_low, g.ci95_high - g.median_participant_diff],
                    fmt="o", color=C1, ms=6, capsize=3, lw=1.5)
        span = max(float(g.ci95_high.max()), 0.0) - min(float(g.ci95_low.min()), 0.0)
        for yi, (_, r) in zip(y, g.iterrows()):
            ax.text(r.ci95_high + 0.03 * span, yi, f"p={r.p_holm:.2g}", fontsize=7, color=INK2, ha="left", va="center")
        ax.set_xlim(min(float(g.ci95_low.min()), 0.0) - 0.05 * span, max(float(g.ci95_high.max()), 0.0) + 0.30 * span)
        ax.set_ylim(-0.6, len(g) - 0.4)
        ax.set_yticks(y, [PRETTY.get(c.replace("full_v2 vs ", ""), c.replace("full_v2 vs ", "")) for c in g.comparison])
        ax.set_xlabel(f"Median paired difference in {'F1' if metric == 'f1' else 'PR-AUC'}")
        ax.grid(axis="y", visible=False)
    fig.suptitle("SleepQualityNet v2 minus baseline (participant-level, Holm-adjusted p)", fontweight="bold", fontsize=10)
    return _save(fig, out_dir, "fig_rq1_sota_forest.png")


def fig_bland_altman(out_dir):
    prof = pd.read_csv(os.path.join(out_dir, "rq7_overnight_n3_eeg_profiles.csv"))
    pr = prof[prof.is_primary & (prof.split == "OOF")].dropna(subset=["pred_n3_minutes", "expert_n3_minutes"])
    a, b = pr.pred_n3_minutes.to_numpy(), pr.expert_n3_minutes.to_numpy()
    m, d = (a + b) / 2, a - b
    bias, sd = d.mean(), d.std(ddof=1)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for c, g in pr.groupby("cohort"):
        gm = (g.pred_n3_minutes + g.expert_n3_minutes) / 2
        ax.scatter(gm, g.pred_n3_minutes - g.expert_n3_minutes, s=14, color=COHORT_COLOR.get(c, INK2),
                   edgecolor="white", linewidth=0.4, label=c)
    for v, ls, lab in ((bias, "-", f"bias {bias:+.1f}"), (bias + 1.96 * sd, "--", f"+1.96 SD {bias + 1.96 * sd:+.1f}"),
                       (bias - 1.96 * sd, "--", f"-1.96 SD {bias - 1.96 * sd:+.1f}")):
        ax.axhline(v, color=INK2, lw=1, ls=ls)
        ax.text(ax.get_xlim()[1], v, lab, fontsize=7, color=INK2, ha="right", va="bottom")
    ax.set_xlabel("Mean of predicted and expert N3 minutes")
    ax.set_ylabel("Predicted - expert (min)")
    ax.set_title("Bland-Altman: overnight N3 duration (OOF)")
    ax.legend(fontsize=7, loc="upper left")
    return _save(fig, out_dir, "fig_rq7_bland_altman.png")


def fig_factor_by_stage(out_dir):
    S = pd.read_parquet(os.path.join(out_dir, "factor_scores.parquet"))
    S = S[(S.split == "OOF") & S.label.isin([2, 3])]
    facs = [f for f in ("delta_intensity", "sw_morphology", "spectral_dominance", "rating_prior") if f in S]
    names = {"delta_intensity": "Delta intensity", "sw_morphology": "Slow-wave morphology",
             "spectral_dominance": "Spectral dominance", "rating_prior": "N3 EEG Rating"}
    fig, axes = plt.subplots(1, len(facs), figsize=(2.3 * len(facs), 3.0), sharey=True)
    for ax, f in zip(np.atleast_1d(axes), facs):
        parts = [S.loc[S.label == k, f].dropna().to_numpy() for k in (2, 3)]
        vp = ax.violinplot(parts, showextrema=False, showmedians=True, widths=0.8)
        for body, c in zip(vp["bodies"], (C3, C1)):
            body.set_facecolor(c)
            body.set_alpha(0.45)
            body.set_edgecolor(c)
        vp["cmedians"].set_color(INK)
        ax.set_xticks([1, 2], ["expert N2", "expert N3"])
        ax.set_title(names[f], fontsize=9)
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("Score (0-100)")
    fig.suptitle("Factor scores: expert N2 vs N3 epochs (OOF)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_factors_by_stage.png")


def fig_interrater(out_dir):
    ir = pd.read_csv(os.path.join(out_dir, "rq1_interrater_benchmark.csv"))
    fig, ax = plt.subplots(figsize=(4.6, 0.5 * len(ir) + 1.2))
    y = np.arange(len(ir))[::-1]
    for yi, (_, r) in zip(y, ir.iterrows()):
        ax.plot([r.expert_vs_expert_f1_mean, r.model_vs_expert_f1_mean], [yi, yi], color=GRID, lw=3, zorder=1)
    ax.scatter(ir.expert_vs_expert_f1_mean, y, s=50, color=INK2, label="expert vs expert", zorder=2)
    ax.scatter(ir.model_vs_expert_f1_mean, y, s=50, color=C1, label="model vs expert", zorder=3)
    ax.set_yticks(y, [f"{c} ({n} scorers)" for c, n in zip(ir.cohort, ir.n_raters)], fontsize=8)
    ax.set_xlabel("Mean N3 F1 against individual scorers")
    ax.set_title("Model vs human inter-rater agreement")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "fig_rq1_interrater.png")


def make_all(out_dir, log=print):
    made = []
    for fn in (fig_pr_curves, fig_homeostatic, fig_graded_agreement, fig_ablation,
               fig_risk_coverage, fig_profiles, fig_example_night, fig_xai, fig_sota, fig_n3_grade,
               fig_confusion_matrices, fig_stage_confusion, fig_roc_curves, fig_calibration,
               fig_participant_f1, fig_sota_forest, fig_bland_altman, fig_factor_by_stage,
               fig_interrater):
        try:
            made.append(fn(out_dir))
        except Exception as exc:
            log(f"[figures] {fn.__name__} skipped: {exc}")
    return made
