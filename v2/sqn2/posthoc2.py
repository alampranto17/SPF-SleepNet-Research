"""v2.3 post hoc analyses requested by the second pre-submission review (31 Oct 2026).

Reported separately from the prespecified verdicts; nothing here changes a criterion.

1. Paired night-time decline test (review M2).  For every participant with >= 10
   detected N3 epochs in each half of the night, a scale-free early-vs-late effect is
   computed per candidate (AUROC of early vs late epochs; 0.5 = no decline).  Each
   rule-derived score is compared with the detector N3 logit by a two-sided paired
   Wilcoxon test, Holm-corrected within each split.
2. Selective prediction by class (review m11): share of expert N3 and expert N2 epochs
   that remain when the least reliable epochs are withheld.
3. Off-the-shelf comparator (review M5): YASA (pretrained, EEG only, inference mode)
   on every cohort, collapsed to N3 vs rest, scored against the same reference.
4. Demographics from source headers (review M4): EDF header patient field and the
   CAP hypnogram header are read with small HTTP range requests; if ages can be
   parsed, participant median factor scores are correlated with age.
"""
from __future__ import annotations

import os
import re
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from . import analysis as A
from . import posthoc as PH
from . import stats as ST
from .train import PRIMARY_VARIANT

DECLINE_CANDIDATES = ["delta_intensity", "sw_morphology", "spectral_dominance", "rating_prior",
                      "signal_composite", "sw_time_frac"]
REFERENCE = "n3_logit"


# ---------------------------------------------------------------------------
# 1. Paired night-time decline
# ---------------------------------------------------------------------------
def _participant_auroc(det, col, min_n=10):
    out = {}
    for p, g in det.groupby("participant"):
        e, l = g[g.half == "early"][col].dropna(), g[g.half == "late"][col].dropna()
        if len(e) >= min_n and len(l) >= min_n:
            y = np.r_[np.ones(len(e)), np.zeros(len(l))]
            out[p] = roc_auc_score(y, np.r_[e.to_numpy(), l.to_numpy()])
    return pd.Series(out, dtype=float)


def paired_decline(d):
    rows = []
    det = d[(d.pred == 1) & d.is_primary]
    for split, g in det.groupby("split"):
        ref = _participant_auroc(g, REFERENCE)
        res = []
        for col in DECLINE_CANDIDATES:
            if col not in g or g[col].isna().all():
                continue
            a = _participant_auroc(g, col)
            common = a.index.intersection(ref.index)
            w = ST.wilcoxon_paired(a[common], ref[common], "two-sided")
            res.append({"split": split, "candidate": col, "reference": REFERENCE,
                        "n_participants": int(len(common)),
                        "median_auroc_candidate": float(a[common].median()),
                        "median_auroc_reference": float(ref[common].median()),
                        "median_paired_difference": w["median_diff"], "p": w["p"]})
        if res:
            r = pd.DataFrame(res)
            r["p_holm"] = ST.holm(r.p.to_numpy())
            rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. Selective prediction by class
# ---------------------------------------------------------------------------
def coverage_by_class(corpus, run_dir, tmp_dir):
    oof, ext = A.load_oof(corpus, run_dir, PRIMARY_VARIANT)
    os.makedirs(tmp_dir, exist_ok=True)
    _, _, _, rel, rel_ext = A.rq6(oof, tmp_dir, ext)
    rows = []
    sets = [("OOF", A.scored(oof).merge(rel, on="idx"))]
    if rel_ext is not None:
        e = A.scored(ext, primary_only=False).merge(rel_ext, on="idx")
        sets += [(f"external {c}", g) for c, g in e.groupby("cohort")]
    for name, g in sets:
        g = g.sort_values("reliability", ascending=False)
        n3_all, n2_all = int((g.label == 3).sum()), int((g.label == 2).sum())
        for cov in (1.0, 0.9, 0.8):
            k = g.iloc[: int(round(cov * len(g)))]
            rows.append({"set": name, "coverage": cov, "n_epochs": len(k),
                         "f1": f1_score(k.y, k.pred, zero_division=0),
                         "expert_n3_retained": (k.label == 3).sum() / max(n3_all, 1),
                         "expert_n2_retained": (k.label == 2).sum() / max(n2_all, 1),
                         "detected_n3_retained": k.pred.sum() / max(g.pred.sum(), 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. YASA off-the-shelf comparator
# ---------------------------------------------------------------------------
def yasa_comparator(corpus, log=print):
    import mne
    import yasa
    mne.set_log_level("ERROR")
    rows, per_rec = [], []
    rec = corpus.rec[corpus.rec.is_primary & corpus.rec.signal_valid]
    for i, (r, m) in enumerate(rec.iterrows()):
        off, n = int(m.offset), int(m.n)
        lab = corpus.ep.label.to_numpy()[off:off + n]
        x = corpus.Xcpu[off:off + n].astype(np.float32) * float(m.raw_scale)   # back to microvolts
        info = mne.create_info(["EEG"], 100.0, ["eeg"])
        raw = mne.io.RawArray(x.reshape(1, -1) * 1e-6, info, verbose=False)
        try:
            sls = yasa.SleepStaging(raw, eeg_name="EEG")
            pred = sls.predict()
            if hasattr(pred, "proba"):                     # yasa >= 0.7: Hypnogram object
                proba, hyp = pred.proba, np.asarray(pred.hypno).astype(str)
            else:                                          # yasa 0.6: array + predict_proba()
                proba, hyp = sls.predict_proba(), np.asarray(pred).astype(str)
            hyp = np.where(np.isin(hyp, ["N3", "3"]), "N3", hyp)
            p3 = proba["N3"].to_numpy()
            k = min(len(p3), n)
            per_rec.append(pd.DataFrame({"rec": r, "participant": m.participant, "cohort": m.cohort,
                                         "label": lab[:k], "yasa_n3_prob": p3[:k],
                                         "yasa_pred_n3": (hyp[:k] == "N3").astype(int)}))
        except Exception as exc:
            log(f"[YASA] {r}: failed ({exc})")
        if (i + 1) % 50 == 0:
            log(f"[YASA] {i + 1}/{len(rec)} recordings")
    d = pd.concat(per_rec, ignore_index=True)
    d = d[d.label >= 0]
    d["y"] = (d.label == 3).astype(int)
    for c, g in [("pooled training cohorts", d[d.cohort.isin(["ISRUC-I", "HMC"])])] + list(d.groupby("cohort")):
        if g.y.nunique() < 2:
            continue
        boot = A.participant_bootstrap(g.assign(prob=g.yasa_n3_prob, pred=g.yasa_pred_n3), n_boot=1000,
                                       metrics=("f1",))
        rows.append({"cohort": c, "n_participants": g.participant.nunique(), "n_epochs": len(g),
                     "f1": f1_score(g.y, g.yasa_pred_n3), "f1_ci": str(boot.get("f1")),
                     "precision": precision_score(g.y, g.yasa_pred_n3, zero_division=0),
                     "recall": recall_score(g.y, g.yasa_pred_n3),
                     "pr_auc": average_precision_score(g.y, g.yasa_n3_prob),
                     "roc_auc": roc_auc_score(g.y, g.yasa_n3_prob),
                     "share_fp_expert_n2": float(((g.yasa_pred_n3 == 1) & (g.y == 0) & (g.label == 2)).sum() /
                                                 max(((g.yasa_pred_n3 == 1) & (g.y == 0)).sum(), 1))})
    return pd.DataFrame(rows), d


# ---------------------------------------------------------------------------
# 4. Demographics from source headers
# ---------------------------------------------------------------------------
HMC_BASE = "https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1/recordings/"
CAP_BASE = "https://physionet-open.s3.amazonaws.com/capslpdb/1.0.0/"


def _get(url, nbytes=None, timeout=30):
    req = urllib.request.Request(url)
    if nbytes:
        req.add_header("Range", f"bytes=0-{nbytes - 1}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(nbytes or -1)


def _edf_header(url):
    h = _get(url, 256).decode("latin-1", "ignore")
    return {"patient_field": h[8:88].strip(), "recording_field": h[88:168].strip(),
            "start_date": h[168:176].strip()}


def _age_from(texts):
    for t in texts:
        m = re.search(r"\bage\D{0,3}(\d{1,3})", t, re.I)
        if m and 1 <= int(m.group(1)) <= 100:
            return float(m.group(1))
    return np.nan


def _sex_from(texts):
    for t in texts:
        m = re.search(r"(?:\bsex|\bgender)\W{0,3}([MF]|male|female)", t, re.I)
        if m:
            return m.group(1)[0].upper()
        f = t.split()
        if len(f) >= 2 and f[1] in ("M", "F"):        # EDF+ patient field: code sex birthdate name
            return f[1]
    return None


def header_demographics(corpus, log=print):
    rows = []
    rec = corpus.rec[corpus.rec.cohort.isin(["HMC", "CAP"])]
    for r, m in rec.iterrows():
        key = r.split("_", 1)[1]
        try:
            if m.cohort == "HMC":
                h = _edf_header(HMC_BASE + f"{key}.edf")
                texts = [h["patient_field"], h["recording_field"]]
            else:
                h = _edf_header(CAP_BASE + f"{key}.edf")
                txt = _get(CAP_BASE + f"{key}.txt", 4000).decode("latin-1", "ignore")
                texts = [h["patient_field"], h["recording_field"], txt.split("Sleep Stage")[0]]
            rows.append({"rec": r, "participant": m.participant, "cohort": m.cohort,
                         "patient_field": h["patient_field"], "age": _age_from(texts), "sex": _sex_from(texts)})
        except Exception as exc:
            rows.append({"rec": r, "participant": m.participant, "cohort": m.cohort,
                         "patient_field": f"ERROR {exc}", "age": np.nan, "sex": None})
    dem = pd.DataFrame(rows)
    log(f"[demographics] parsed age for {dem.age.notna().sum()}/{len(dem)} HMC/CAP recordings, "
        f"sex for {dem.sex.notna().sum()}")
    return dem


def age_association(d, dem):
    """Participant median factor scores on expert N3 epochs vs age, per cohort."""
    rows = []
    if dem.age.notna().sum() < 10:
        return pd.DataFrame([{"note": "fewer than 10 participants with a parsable age; analysis not run"}])
    ex = d[(d.label == 3) & d.is_primary]
    pm = ex.groupby(["participant", "cohort"])[DECLINE_CANDIDATES[:4] + ["sw_time_frac"]].median().reset_index()
    pm = pm.merge(dem[["participant", "age", "sex"]], on="participant", how="inner")
    for c, g in pm.groupby("cohort"):
        g = g[g.age.notna()]
        for col in DECLINE_CANDIDATES[:4] + ["sw_time_frac"]:
            rho, p = ST.spearman(g[col], g.age)
            rows.append({"cohort": c, "score": col, "n": int(len(g)), "spearman_vs_age": rho, "p": p,
                         "age_median": float(g.age.median()), "age_min": float(g.age.min()),
                         "age_max": float(g.age.max())})
    return pd.DataFrame(rows)


def run_all(corpus, run_dir, out_dir, log=print):
    log("[posthoc2] scores and detector outputs")
    d = PH._frame(corpus, run_dir)
    log("[posthoc2] 1. paired night-time decline: rule-derived scores vs detector logit")
    paired_decline(d).to_csv(os.path.join(out_dir, "posthoc2_paired_decline.csv"), index=False)
    log("[posthoc2] 2. selective prediction by class")
    coverage_by_class(corpus, run_dir, os.path.join(out_dir, "_tmp_rq6")).to_csv(
        os.path.join(out_dir, "posthoc2_coverage_by_class.csv"), index=False)
    log("[posthoc2] 4. demographics from source headers")
    try:
        dem = header_demographics(corpus, log)
        dem.to_csv(os.path.join(out_dir, "posthoc2_header_demographics.csv"), index=False)
        age_association(d, dem).to_csv(os.path.join(out_dir, "posthoc2_age_association.csv"), index=False)
    except Exception as exc:
        log(f"[posthoc2] demographics failed: {exc}")
    log("[posthoc2] 3. YASA off-the-shelf comparator (inference only)")
    try:
        res, _ = yasa_comparator(corpus, log)
        res.to_csv(os.path.join(out_dir, "posthoc2_yasa_comparator.csv"), index=False)
    except Exception as exc:
        log(f"[posthoc2] YASA failed: {exc}")
    import shutil
    shutil.rmtree(os.path.join(out_dir, "_tmp_rq6"), ignore_errors=True)
    log("[posthoc2] done")
