"""v2.4 post hoc refinements requested by the third pre-submission review (31 Oct 2026).

Reported separately from the prespecified verdicts; nothing here changes a criterion.

1. Paired night-time decline without selection on the logit (review M2).  The paired
   early-vs-late AUROC comparison of posthoc2 used detected N3 epochs, which are
   selected by the logit itself (range restriction).  Here the same test is run on
   expert-scored N3 epochs and on all epochs scored N2 or N3 by the reference.
2. Age association adjusted for diagnosis (review M4).  CAP participants: partial
   Spearman correlation between age and participant median factor scores (expert N3
   epochs, >= 10 per participant), adjusted for the diagnosis group taken from the
   record name; plus the correlation within diagnosis groups with >= 10 participants.
3. YASA and the proposed detector on identical ISRUC-II nights (review M3): both
   nights for both models, and the first night for both models.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from . import analysis as A
from . import posthoc as PH
from . import posthoc2 as P2
from . import stats as ST
from .train import PRIMARY_VARIANT


# ---------------------------------------------------------------------------
# 1. Paired decline on epochs not selected by the logit
# ---------------------------------------------------------------------------
def paired_decline_unselected(d):
    out = []
    sets = {"expert N3 epochs": d[(d.label == 3) & d.is_primary],
            "expert N2 or N3 epochs": d[d.label.isin([2, 3]) & d.is_primary]}
    for name, sub in sets.items():
        for split, g in sub.groupby("split"):
            ref = P2._participant_auroc(g, P2.REFERENCE)
            res = []
            for col in P2.DECLINE_CANDIDATES:
                if col not in g or g[col].isna().all():
                    continue
                a = P2._participant_auroc(g, col)
                common = a.index.intersection(ref.index)
                w = ST.wilcoxon_paired(a[common], ref[common], "two-sided")
                res.append({"epochs": name, "split": split, "candidate": col, "reference": P2.REFERENCE,
                            "n_participants": int(len(common)),
                            "median_auroc_candidate": float(a[common].median()),
                            "median_auroc_reference": float(ref[common].median()),
                            "median_paired_difference": w["median_diff"], "p": w["p"]})
            if res:
                r = pd.DataFrame(res)
                r["p_holm"] = ST.holm(r.p.to_numpy())
                out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. Age association adjusted for diagnosis (CAP)
# ---------------------------------------------------------------------------
def _partial_spearman(x, y, Z):
    """Spearman correlation of x and y after removing the linear effect of Z on the ranks."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Z1 = np.column_stack([np.ones(len(x)), Z])
    ex = rx - Z1 @ np.linalg.lstsq(Z1, rx, rcond=None)[0]
    ey = ry - Z1 @ np.linalg.lstsq(Z1, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1])
    df = len(x) - 2 - Z.shape[1]
    t = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), df)), int(df)


def age_adjusted(d, dem, min_epochs=10):
    cols = P2.DECLINE_CANDIDATES[:4] + ["sw_time_frac"]
    ex = d[(d.label == 3) & d.is_primary & (d.cohort == "CAP")]
    cnt = ex.groupby("participant").size()
    keep = cnt[cnt >= min_epochs].index
    pm = ex[ex.participant.isin(keep)].groupby("participant")[cols].median().reset_index()
    pm = pm.merge(dem[["participant", "age", "sex"]], on="participant", how="inner")
    pm = pm[pm.age.notna()]
    pm["diagnosis"] = pm.participant.str.replace("CAP_", "").str.replace(r"\d+", "", regex=True)
    pm["diagnosis"] = pm.diagnosis.replace({"n": "healthy control"})
    Z = pd.get_dummies(pm.diagnosis, drop_first=True).to_numpy(float)
    rows = []
    info = {"n_participants": int(len(pm)), "min_expert_n3_epochs": int(min_epochs),
            "median_expert_n3_epochs": float(cnt[cnt.index.isin(pm.participant)].median()),
            "n_groups": int(pm.diagnosis.nunique())}
    for col in cols:
        rho, p = ST.spearman(pm[col], pm.age)
        pr, pp, df = _partial_spearman(pm[col].to_numpy(), pm.age.to_numpy(), Z)
        rows.append({"set": "CAP, all diagnoses", "score": col, **info, "spearman": rho, "p": p,
                     "partial_spearman_adj_diagnosis": pr, "p_partial": pp, "df": df})
        for grp, g in pm.groupby("diagnosis"):
            if len(g) >= 10:
                r2, p2 = ST.spearman(g[col], g.age)
                rows.append({"set": f"CAP, {grp} only", "score": col, "n_participants": int(len(g)),
                             "spearman": r2, "p": p2, "age_min": float(g.age.min()), "age_max": float(g.age.max())})
    grp = pm.groupby("diagnosis").age.agg(["size", "median", "min", "max"]).reset_index()
    return pd.DataFrame(rows), grp


# ---------------------------------------------------------------------------
# 3. YASA vs proposed detector on identical ISRUC-II nights
# ---------------------------------------------------------------------------
def isruc2_same_nights(corpus, run_dir, log=print):
    import mne
    import yasa
    mne.set_log_level("ERROR")
    rec = corpus.rec[(corpus.rec.cohort == "ISRUC-II") & corpus.rec.signal_valid]
    parts = []
    for r, m in rec.iterrows():
        off, n = int(m.offset), int(m.n)
        x = corpus.Xcpu[off:off + n].astype(np.float32) * float(m.raw_scale)
        raw = mne.io.RawArray(x.reshape(1, -1) * 1e-6, mne.create_info(["EEG"], 100.0, ["eeg"]), verbose=False)
        pred = yasa.SleepStaging(raw, eeg_name="EEG").predict()
        proba, hyp = (pred.proba, np.asarray(pred.hypno).astype(str)) if hasattr(pred, "proba") else \
            (None, np.asarray(pred).astype(str))
        k = min(len(hyp), n)
        parts.append(pd.DataFrame({"idx": np.arange(off, off + k), "yasa_p": proba["N3"].to_numpy()[:k],
                                   "yasa_pred": (hyp[:k] == "N3").astype(int)}))
    yv = pd.concat(parts, ignore_index=True)
    _, ext = A.load_oof(corpus, run_dir, PRIMARY_VARIANT)
    e = A.scored(ext, primary_only=False)
    e = e[e.cohort == "ISRUC-II"].merge(yv, on="idx", how="inner")
    rows = []
    for name, g in (("both nights", e), ("first night", e[e.is_primary]), ("second night", e[~e.is_primary])):
        for model, p, pr in (("proposed", g.prob, g.pred), ("YASA", g.yasa_p, g.yasa_pred)):
            rows.append({"set": name, "model": model, "n_epochs": len(g), "f1": f1_score(g.y, pr),
                         "pr_auc": average_precision_score(g.y, p), "roc_auc": roc_auc_score(g.y, p)})
    return pd.DataFrame(rows)


def run_all(corpus, run_dir, out_dir, log=print):
    log("[posthoc3] scores and detector outputs")
    d = PH._frame(corpus, run_dir)
    log("[posthoc3] 1. paired decline on expert N3 and expert N2/N3 epochs")
    paired_decline_unselected(d).to_csv(os.path.join(out_dir, "posthoc3_paired_decline_unselected.csv"), index=False)
    log("[posthoc3] 2. CAP age association adjusted for diagnosis")
    try:
        dem, _ = P2.source_demographics(out_dir, log)
        res, grp = age_adjusted(d, dem)
        res.to_csv(os.path.join(out_dir, "posthoc3_age_adjusted.csv"), index=False)
        grp.to_csv(os.path.join(out_dir, "posthoc3_cap_age_by_diagnosis.csv"), index=False)
    except Exception as exc:
        log(f"[posthoc3] age analysis failed: {exc}")
    for f in os.listdir(out_dir):                      # aggregated HMC text is not needed here
        if f.endswith(".txt"):
            os.remove(os.path.join(out_dir, f))
    log("[posthoc3] 3. YASA vs proposed detector on identical ISRUC-II nights")
    try:
        isruc2_same_nights(corpus, run_dir, log).to_csv(os.path.join(out_dir, "posthoc3_isruc2_same_nights.csv"),
                                                        index=False)
    except Exception as exc:
        log(f"[posthoc3] ISRUC-II comparison failed: {exc}")
    log("[posthoc3] done")
