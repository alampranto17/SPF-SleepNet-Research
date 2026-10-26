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

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COHORT_COLOR = {"ISRUC-I": C1, "Sleep-EDF": C2, "ISRUC-III": C3}

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
    ep = os.path.join(out_dir, "external_isruc3_predictions.parquet")
    if os.path.exists(ep):
        e = pd.read_parquet(ep)
        sets.append(("ISRUC-III", e[e.label >= 0]))
    for c, g in sets:
        p, r, _ = precision_recall_curve(g.y, g.prob)
        ax.plot(r, p, color=COHORT_COLOR.get(c, INK2),
                label=f"{c} (AP={average_precision_score(g.y, g.prob):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("N3 detection, out-of-fold")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return _save(fig, out_dir, "fig_rq1_pr_curves.png")


def fig_homeostatic(out_dir):
    S = pd.read_parquet(os.path.join(out_dir, "factor_scores_oof.parquet"))
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
    pr = prof[prof.is_primary]
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
    sedf = prof[prof.cohort == "Sleep-EDF"]
    for metric, color, lab in (("pred_median_rating", C1, "Overall N3 EEG Rating"),
                               ("pred_n3_minutes", C2, "N3 minutes")):
        piv = sedf.pivot_table(index="participant", columns="is_primary", values=metric)
        if True in piv and False in piv:
            piv = piv.dropna()
            z = (piv - piv.stack().mean()) / (piv.stack().std() + 1e-9)
            axes[1].scatter(z[True], z[False], s=18, color=color, label=lab,
                            edgecolor="white", linewidth=0.5)
    axes[1].plot([-3, 3], [-3, 3], color=INK2, lw=1, ls="--")
    axes[1].set_xlabel("Night 1 (z-score)")
    axes[1].set_ylabel("Night 2 (z-score)")
    axes[1].set_title("Night-to-night reproducibility")
    axes[1].legend(fontsize=8)
    return _save(fig, out_dir, "fig_rq7_profiles.png")


def fig_example_night(out_dir, participant=None):
    S = pd.read_parquet(os.path.join(out_dir, "factor_scores_oof.parquet"))
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


def make_all(out_dir, log=print):
    made = []
    for fn in (fig_pr_curves, fig_homeostatic, fig_ablation, fig_risk_coverage, fig_profiles,
               fig_example_night, fig_xai):
        try:
            made.append(fn(out_dir))
        except Exception as exc:
            log(f"[figures] {fn.__name__} skipped: {exc}")
    return made
