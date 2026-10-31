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
4. Demographics from the source releases (review M4): CAP gender-age.xlsx (PhysioNet),
   ISRUC Details spreadsheets if the server answers, HMC aggregated file.  Participant
   median factor scores on expert N3 epochs are related to age and sex where available.
   No individual demographic value is written to the outputs, only summaries.
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
# 4. Demographics from the source releases
# ---------------------------------------------------------------------------
HMC_BASES = ("https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1/",
             "https://physionet.org/files/hmc-sleep-staging/1.1/")
CAP_BASES = ("https://physionet-open.s3.amazonaws.com/capslpdb/1.0.0/",
             "https://physionet.org/files/capslpdb/1.0.0/")
ISRUC_DETAILS = ("http://dataset.isr.uc.pt/ISRUC_Sleep/Details/Details_subgroup_{g}_Submission.xlsx",
                 "https://sleeptight.isr.uc.pt/ISRUC_Sleep/Details/Details_subgroup_{g}_Submission.xlsx")


def _get_any(urls, timeout=60):
    last = None
    for u in urls:
        try:
            with urllib.request.urlopen(u, timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            last = exc
    raise RuntimeError(f"all sources failed ({last})")


def _cap_demographics(tmp):
    from . import data as D
    raw = _get_any([b + "gender-age.xlsx" for b in CAP_BASES])
    p = os.path.join(tmp, "cap_gender_age.xlsx")
    open(p, "wb").write(raw)
    for header in range(0, 4):
        try:
            df = pd.read_excel(p, header=header)
        except Exception:
            break
        cs = D._find_col(df.columns, "subject", "patient", "record", "name", "id")
        ca = D._find_col(df.columns, "age")
        if cs is None or ca is None:
            continue
        cg = D._find_col(df.columns, "gender", "sex")
        rows = []
        for _, r in df.iterrows():
            key = str(r[cs]).strip().lower()
            if not re.match(r"^[a-z]+\d+$", key):
                continue
            try:
                age = float(r[ca])
            except (TypeError, ValueError):
                continue
            sex = str(r[cg]).strip()[:1].upper() if cg is not None else None
            rows.append({"participant": f"CAP_{key}", "cohort": "CAP", "age": age, "sex": sex})
        if rows:
            return pd.DataFrame(rows)
    # no usable header: detect the columns by content
    df = pd.read_excel(p, header=None).astype(str)
    is_key = df.apply(lambda c: c.str.strip().str.lower().str.match(r"^[a-z]+\d+$").mean())
    is_sex = df.apply(lambda c: c.str.strip().str.upper().str[:1].isin(["M", "F"]).mean())
    num = df.apply(lambda c: pd.to_numeric(c, errors="coerce"))
    is_age = num.apply(lambda c: c.between(1, 100).mean())
    if is_key.max() > 0.5 and is_age.max() > 0.5:
        ck, ca, cg = is_key.idxmax(), is_age.idxmax(), (is_sex.idxmax() if is_sex.max() > 0.5 else None)
        rows = [{"participant": f"CAP_{str(r[ck]).strip().lower()}", "cohort": "CAP",
                 "age": float(num.loc[i, ca]),
                 "sex": str(r[cg]).strip()[:1].upper() if cg is not None else None}
                for i, r in df.iterrows()
                if re.match(r"^[a-z]+\d+$", str(r[ck]).strip().lower()) and not np.isnan(num.loc[i, ca])]
        if rows:
            return pd.DataFrame(rows)
    raise RuntimeError(f"could not parse CAP gender-age.xlsx (columns {list(df.columns)})")


def _isruc_demographics(tmp):
    from . import data as D
    out = []
    for g, cohort in (("I", "ISRUC-I"), ("II", "ISRUC-II"), ("III", "ISRUC-III")):
        raw = _get_any([u.format(g=g) for u in ISRUC_DETAILS], timeout=30)
        p = os.path.join(tmp, f"isruc_{g}.xlsx")
        open(p, "wb").write(raw)
        for pid, v in D.read_isruc_details(p, D.COHORT_PREFIX[cohort]).items():
            sex = (v.get("sex") or "")[:1].upper() or None
            out.append({"participant": pid, "cohort": cohort, "age": v["age"], "sex": sex})
    return pd.DataFrame(out)


def source_demographics(out_dir, log=print):
    tmp = os.path.join(out_dir, "_tmp_dem")
    os.makedirs(tmp, exist_ok=True)
    parts, notes = [], []
    for name, fn in (("CAP gender-age.xlsx (PhysioNet)", _cap_demographics),
                     ("ISRUC Details spreadsheets", _isruc_demographics)):
        try:
            d = fn(tmp)
            parts.append(d)
            notes.append({"source": name, "status": "AVAILABLE", "n": len(d)})
        except Exception as exc:
            notes.append({"source": name, "status": f"NOT_AVAILABLE ({exc})", "n": 0})
    try:
        txt = _get_any([b + "Subjects_info_aggregated.txt" for b in HMC_BASES]).decode("latin-1", "ignore")
        open(os.path.join(out_dir, "posthoc2_hmc_subjects_info_aggregated.txt"), "w").write(txt)
        notes.append({"source": "HMC Subjects_info_aggregated.txt (PhysioNet)", "status": "AGGREGATED ONLY", "n": 0})
    except Exception as exc:
        notes.append({"source": "HMC Subjects_info_aggregated.txt", "status": f"NOT_AVAILABLE ({exc})", "n": 0})
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    for n in notes:
        log(f"[demographics] {n}")
    dem = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["participant", "cohort", "age", "sex"])
    return dem, pd.DataFrame(notes)


def demographics_summary(dem, corpus):
    rows = []
    evaluated = set(corpus.rec[corpus.rec.signal_valid].participant)
    for c, g in dem[dem.participant.isin(evaluated)].groupby("cohort"):
        rows.append({"cohort": c, "n_with_age": int(g.age.notna().sum()), "age_mean": g.age.mean(),
                     "age_sd": g.age.std(), "age_min": g.age.min(), "age_max": g.age.max(),
                     "n_male": int((g.sex == "M").sum()), "n_female": int((g.sex == "F").sum())})
    return pd.DataFrame(rows)


def age_association(d, dem):
    """Participant median factor scores (expert N3 epochs) vs age and sex, per cohort."""
    rows = []
    if dem.age.notna().sum() < 10:
        return pd.DataFrame([{"note": "fewer than 10 participants with age; analysis not run"}])
    cols = DECLINE_CANDIDATES[:4] + ["sw_time_frac"]
    ex = d[(d.label == 3) & d.is_primary]
    pm = ex.groupby(["participant", "cohort"])[cols].median().reset_index()
    pm = pm.merge(dem[["participant", "age", "sex"]], on="participant", how="inner")
    from scipy import stats
    for c, g in [("all with age", pm)] + list(pm.groupby("cohort")):
        g = g[g.age.notna()]
        for col in cols:
            rho, p = ST.spearman(g[col], g.age)
            m, f = g[g.sex == "M"][col].dropna(), g[g.sex == "F"][col].dropna()
            ps = float(stats.mannwhitneyu(m, f).pvalue) if len(m) >= 5 and len(f) >= 5 else np.nan
            rows.append({"cohort": c, "score": col, "n": int(len(g)), "spearman_vs_age": rho, "p_age": p,
                         "median_male": float(m.median()) if len(m) else np.nan,
                         "median_female": float(f.median()) if len(f) else np.nan, "p_sex": ps})
    return pd.DataFrame(rows)


def run_all(corpus, run_dir, out_dir, log=print):
    log("[posthoc2] scores and detector outputs")
    d = PH._frame(corpus, run_dir)
    log("[posthoc2] 1. paired night-time decline: rule-derived scores vs detector logit")
    paired_decline(d).to_csv(os.path.join(out_dir, "posthoc2_paired_decline.csv"), index=False)
    log("[posthoc2] 2. selective prediction by class")
    coverage_by_class(corpus, run_dir, os.path.join(out_dir, "_tmp_rq6")).to_csv(
        os.path.join(out_dir, "posthoc2_coverage_by_class.csv"), index=False)
    log("[posthoc2] 4. demographics from the source releases")
    try:
        dem, notes = source_demographics(out_dir, log)
        notes.to_csv(os.path.join(out_dir, "posthoc2_demographics_sources.csv"), index=False)
        demographics_summary(dem, corpus).to_csv(os.path.join(out_dir, "posthoc2_demographics_summary.csv"),
                                                 index=False)
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
