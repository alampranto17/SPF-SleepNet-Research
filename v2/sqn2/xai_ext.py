"""Extended explainability (v2.1): errors, external cohorts and robustness.

The pre-specified RQ5 analysis (xai.rq5) explains correctly detected N3 epochs of
the training cohorts only.  This module adds, with the same methods
(band-stop perturbation with recomputed features, band-decomposed integrated
gradients, attention-slow-wave alignment, SHAP surrogate):

  1. Outcome-stratified explanations: true positives (TP), false positives (FP),
     false negatives (FN) and expert-N2 true negatives (TN-N2), out-of-fold AND
     in every external cohort.  External epochs are explained with all fold
     models and the attributions are averaged (the external prediction is the
     5-fold ensemble).  Question answered: are the errors driven by delta
     activity (the physiology of N3) or by something else?
  2. SHAP surrogate per outcome (signed group attribution) and a per-feature
     ranking of mean |SHAP| over folds.
  3. Robustness of the explanations:
       * IG baseline: all-zero epoch (primary) vs Gaussian-noise baseline with
         the epoch's SD (mean of 4 draws): agreement of the top band;
       * IG steps: completeness error |sum IG - (f(x) - f(x0))| / |f(x) - f(x0)|
         at 32 and 64 steps;
       * participant bootstrap 95 % CI of the share of participants whose top
         band is delta (perturbation and IG);
       * the RQ5 criterion re-applied to each external cohort.
  4. Example traces (EEG, IG attribution over time, attention, slow-wave
     coverage) for figures.

None of this changes the pre-specified RQ5 verdict.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt

from . import features as FT
from . import stats as ST
from .train import VARIANTS, PRIMARY_VARIANT, build_model, load_run, variant_features
from .xai import BANDS, FEATURE_GROUPS, HALF_WIN, _band_contributions, _epoch_features, _sw_coverage

OUTCOMES = ("TP", "FP", "FN", "TN-N2")
_STOP = {b: butter(4, [lo, hi], btype="bandstop", fs=FT.FS, output="sos") for b, (lo, hi) in BANDS.items()}


def _outcome(df):
    o = np.full(len(df), "", dtype=object)
    o[(df.pred == 1) & (df.y == 1)] = "TP"
    o[(df.pred == 1) & (df.y == 0)] = "FP"
    o[(df.pred == 0) & (df.y == 1)] = "FN"
    o[(df.pred == 0) & (df.label == 2)] = "TN-N2"
    return o


class _Fold:
    """One fold model with its feature scaling (training participants of that fold)."""

    def __init__(self, corpus, run_dir, fold, variant):
        dev = corpus.device
        r = load_run(os.path.join(run_dir, f"{variant}__fold{fold}.npz"))
        meta = json.loads(r["meta"])
        self.v = VARIANTS[variant]
        self.model = build_model(self.v, corpus.Fraw.shape[1]).to(dev)
        self.model.load_state_dict(torch.load(os.path.join(run_dir, f"{variant}__fold{fold}.pt"),
                                              map_location=dev))
        self.model.eval()
        tr = corpus.idx_of(meta["train_recs"])
        tr = tr[corpus.ep.label.to_numpy()[tr] >= 0]
        self.scaler = FT.RobustScaler().fit(corpus.Fraw[tr])
        self.Fs = torch.from_numpy(self.scaler.transform(corpus.Fraw)).to(dev)
        self.Fv = variant_features(self.v, self.Fs)     # what the network receives


def _window(corpus, i):
    rec = corpus.rec.loc[corpus.ep.rec.iat[i]]
    o, n = int(rec.offset), int(rec.n)
    t = i - o
    lo, hi = max(0, t - HALF_WIN), min(n, t + HALF_WIN + 1)
    return rec, o + lo, o + hi, t - lo


def _ig(model, xw, fw, pos, baseline, steps):
    """Integrated gradients for the target epoch; returns (path-mean gradient,
    completeness error)."""
    dev = xw.device
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=dev)
    tgt = (baseline[None, :] + alphas[:, None] * (xw[pos] - baseline)[None, :]).detach().requires_grad_(True)
    xb = xw.unsqueeze(0).repeat(steps, 1, 1)
    xb = torch.cat([xb[:, :pos], tgt[:, None], xb[:, pos + 1:]], 1)
    fb = fw.unsqueeze(0).repeat(steps, 1, 1)
    logits = model(xb, fb)[:, pos]
    grads = torch.autograd.grad(logits.sum(), tgt)[0]
    gbar = grads.mean(0).detach().cpu().numpy().astype(np.float64)
    with torch.no_grad():
        x0 = xw.clone()
        x0[pos] = baseline
        f_x = model.predict_sequence(xw, fw)[pos].float().item()
        f_0 = model.predict_sequence(x0, fw)[pos].float().item()
    diff = (xw[pos] - baseline).detach().cpu().numpy().astype(np.float64)
    total = float(np.dot(diff, gbar))
    comp_err = abs(total - (f_x - f_0)) / max(abs(f_x - f_0), 1e-6)
    return gbar, diff, comp_err


def explain(corpus, folds, i, ig_steps=32, robustness=False, keep_trace=False):
    """Explanation of epoch i averaged over the given fold models."""
    rec, a, b, pos = _window(corpus, i)
    scale = float(rec.raw_scale)
    acc = []
    trace = None
    for fd in folds:
        model, Fs = fd.model, fd.Fs
        xw = corpus.X[a:b].float().clone()
        fw = fd.Fv[a:b].clone()
        with torch.no_grad():
            base = torch.sigmoid(model.predict_sequence(xw, fw)[pos]).item()
        x_uv = xw[pos].cpu().numpy() * scale
        out = {"p_orig": base}
        for band, sos in _STOP.items():
            xp = sosfiltfilt(sos, x_uv)
            xw2, fw2 = xw.clone(), Fs[a:b].clone()
            xw2[pos] = torch.from_numpy(xp / scale).to(xw.device)
            fw2[pos] = torch.from_numpy(fd.scaler.transform(_epoch_features(xp))[0]).to(xw.device)
            fw2 = variant_features(fd.v, fw2)
            with torch.no_grad():
                out[f"drop_{band}"] = base - torch.sigmoid(model.predict_sequence(xw2, fw2)[pos]).item()
        zero = torch.zeros_like(xw[pos])
        gbar, diff, err = _ig(model, xw, fw, pos, zero, ig_steps)
        for band, val in _band_contributions(diff, gbar).items():
            out[f"ig_{band}"] = val
        out["ig_completeness_err"] = err
        if robustness:
            _, _, err64 = _ig(model, xw, fw, pos, zero, 64)
            out["ig_completeness_err_64"] = err64
            sd = float(xw[pos].float().std().item())
            gen = torch.Generator(device="cpu").manual_seed(int(i) % 2 ** 31)
            contrib = {bd: 0.0 for bd in BANDS}
            for _ in range(4):
                nb = (torch.randn(xw.shape[1], generator=gen) * sd).to(xw.device, xw.dtype)
                g2, d2, _ = _ig(model, xw, fw, pos, nb, ig_steps)
                for bd, val in _band_contributions(d2, g2).items():
                    contrib[bd] += val / 4
            for bd, val in contrib.items():
                out[f"ignoise_{bd}"] = val
        if fd.v["use_attention"]:
            with torch.no_grad():
                _, att = model.encoder(xw[pos][None, None], return_attn=True)
            att = att[0].float().cpu().numpy()
            cov = _sw_coverage(x_uv)
            out["attn_sw_spearman"] = ST.spearman(att, cov)[0] if cov.std() > 0 else np.nan
            hb = cov > 0.5
            out["attn_sw_enrichment"] = (att[hb].sum() / hb.mean()) if hb.any() else np.nan
            if keep_trace and trace is None:
                trace = {"eeg_uv": x_uv.astype(np.float32), "ig_t": (diff * gbar).astype(np.float32),
                         "attention": att.astype(np.float32), "sw_coverage": cov.astype(np.float32)}
        acc.append(out)
    res = pd.DataFrame(acc).mean(numeric_only=True).to_dict()
    return res, trace


def _sample(df, n_per_group, seed):
    parts = []
    per_part = max(1, n_per_group // 8)          # spread over participants first, then cap
    for _, g in df.groupby(["cohort", "outcome"]):
        g = pd.concat([x.sample(min(len(x), per_part), random_state=seed) for _, x in g.groupby("participant")])
        parts.append(g.sample(min(len(g), n_per_group), random_state=seed))
    return pd.concat(parts) if parts else df.iloc[:0]


def rq5_extended(corpus, run_dir, out_dir, oof, ext=None, n_per_group=40, n_robust=60,
                 seed=0, variant=PRIMARY_VARIANT, log=print):
    fold_ids = sorted(oof.fold.unique())
    folds = {f: _Fold(corpus, run_dir, f, variant) for f in fold_ids}
    o = oof[oof.is_primary & (oof.label >= 0)].copy()
    o["outcome"] = _outcome(o)
    o = o[o.outcome != ""]
    rows, traces = [], {}
    pick = _sample(o, n_per_group, seed)
    log(f"[RQ5x] OOF: {len(pick)} epochs ({pick.groupby('outcome').size().to_dict()})")
    rob_ids = set(pick[pick.outcome == "TP"].sample(min(n_robust, int((pick.outcome == "TP").sum())),
                                                    random_state=seed).idx)
    for _, r in pick.iterrows():
        key = f"OOF|{r.cohort}|{r.outcome}"
        res, tr = explain(corpus, [folds[r.fold]], int(r.idx), robustness=int(r.idx) in rob_ids,
                          keep_trace=key not in traces)
        if tr is not None:
            traces[key] = tr
        rows.append({"split": "OOF", "cohort": r.cohort, "participant": r.participant, "idx": int(r.idx),
                     "outcome": r.outcome, "n3_vote_frac": r.n3_vote_frac, **res})
    if ext is not None and len(ext):
        e = ext[ext.label >= 0].copy()
        e["outcome"] = _outcome(e)
        e = e[e.outcome != ""]
        pe = _sample(e, n_per_group, seed)
        log(f"[RQ5x] external: {len(pe)} epochs x {len(folds)} fold models")
        for _, r in pe.iterrows():
            key = f"external|{r.cohort}|{r.outcome}"
            res, tr = explain(corpus, list(folds.values()), int(r.idx), keep_trace=key not in traces)
            if tr is not None:
                traces[key] = tr
            rows.append({"split": "external", "cohort": r.cohort, "participant": r.participant,
                         "idx": int(r.idx), "outcome": r.outcome, "n3_vote_frac": r.n3_vote_frac, **res})
    X = pd.DataFrame(rows)
    X.to_csv(os.path.join(out_dir, "rq5x_epoch_explanations.csv"), index=False)
    if traces:
        np.savez_compressed(os.path.join(out_dir, "rq5x_examples.npz"),
                            **{f"{k}|{name}": v for k, tr in traces.items() for name, v in tr.items()})
    bands = list(BANDS)
    # band importance by split x cohort x outcome (participant-level medians)
    agg = []
    for (split, coh, out), g in X.groupby(["split", "cohort", "outcome"]):
        pm = g.groupby("participant")[[f"drop_{b}" for b in bands] + [f"ig_{b}" for b in bands]].mean()
        r = {"split": split, "cohort": coh, "outcome": out, "n_epochs": len(g), "n_participants": len(pm)}
        for b in bands:
            r[f"median_drop_{b}"] = float(pm[f"drop_{b}"].median())
            r[f"median_ig_{b}"] = float(pm[f"ig_{b}"].median())
        r["top_band_perturbation"] = bands[int(np.argmax([r[f"median_drop_{b}"] for b in bands]))]
        r["top_band_ig"] = bands[int(np.argmax([r[f"median_ig_{b}"] for b in bands]))]
        if "attn_sw_spearman" in g:
            r["attn_sw_spearman_median"] = float(g.attn_sw_spearman.median())
        if out == "FP" and g.n3_vote_frac.notna().any():
            r["fp_with_>=1_expert_N3_vote"] = float((g.n3_vote_frac.dropna() > 0).mean())
        agg.append(r)
    pd.DataFrame(agg).to_csv(os.path.join(out_dir, "rq5x_band_importance_by_outcome.csv"), index=False)
    _shap_by_outcome(corpus, o, folds, out_dir, seed)
    _robustness(X, out_dir, bands, seed)
    return X


def _shap_by_outcome(corpus, o, folds, out_dir, seed):
    try:
        import shap
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import r2_score
    except Exception as exc:                                  # shap missing
        pd.DataFrame([{"error": str(exc)}]).to_csv(os.path.join(out_dir, "rq5x_shap_by_outcome.csv"))
        return
    names = FT.DETECTION_FEATURES
    rows, feat_rows = [], []
    for f, fd in folds.items():
        g = o[o.fold == f]
        g = g.sample(min(len(g), 8000), random_state=seed)
        Xs = fd.Fs[torch.as_tensor(g.idx.to_numpy().copy(), device=fd.Fs.device)].cpu().numpy()
        yl = g["logit"].to_numpy()
        half = len(g) // 2
        sur = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=seed)
        sur.fit(Xs[:half], yl[:half])
        r2 = float(r2_score(yl[half:], sur.predict(Xs[half:])))
        sv = shap.TreeExplainer(sur).shap_values(Xs[half:])
        out = g.outcome.to_numpy()[half:]
        for oc in OUTCOMES:
            m = out == oc
            if m.sum() < 10:
                continue
            r = {"fold": f, "outcome": oc, "n": int(m.sum()), "fidelity_r2": r2}
            for gname, cols in FEATURE_GROUPS.items():
                idx = [names.index(c) for c in cols]
                r[f"signed_{gname}"] = float(sv[m][:, idx].sum(1).mean())
                r[f"abs_{gname}"] = float(np.abs(sv[m][:, idx].sum(1)).mean())
            rows.append(r)
        for j, n in enumerate(names):
            feat_rows.append({"fold": f, "feature": n, "mean_abs_shap": float(np.abs(sv[:, j]).mean()),
                              "group": next((k for k, v in FEATURE_GROUPS.items() if n in v), "other")})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "rq5x_shap_by_outcome.csv"), index=False)
    fr = pd.DataFrame(feat_rows)
    if len(fr):
        fr = fr.groupby(["feature", "group"]).mean_abs_shap.agg(["mean", "std"]).reset_index() \
               .sort_values("mean", ascending=False)
    fr.to_csv(os.path.join(out_dir, "rq5x_shap_feature_ranking.csv"), index=False)


def _top_delta_share(P, prefix, bands):
    return float((P[[f"{prefix}{b}" for b in bands]].idxmax(axis=1) == f"{prefix}delta").mean())


def _robustness(X, out_dir, bands, seed):
    rows = []

    def add(analysis, setting, value, primary=np.nan, holds=np.nan, note=""):
        rows.append({"analysis": analysis, "setting": setting, "value": value,
                     "primary_value": primary, "holds": holds, "note": note})

    tp = X[(X.split == "OOF") & (X.outcome == "TP")]
    P = tp.groupby("participant")[[f"drop_{b}" for b in bands] + [f"ig_{b}" for b in bands]].mean()
    prim_both = float(((P[[f"drop_{b}" for b in bands]].idxmax(axis=1) == "drop_delta") &
                       (P[[f"ig_{b}" for b in bands]].idxmax(axis=1) == "ig_delta")).mean()) if len(P) else np.nan
    # participant bootstrap of the share with delta on top for both methods
    if len(P) >= 5:
        rng = np.random.default_rng(seed)
        bs = []
        for _ in range(1000):
            Q = P.iloc[rng.integers(0, len(P), len(P))]
            bs.append(((Q[[f"drop_{b}" for b in bands]].idxmax(axis=1) == "drop_delta") &
                       (Q[[f"ig_{b}" for b in bands]].idxmax(axis=1) == "ig_delta")).mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        add("participant bootstrap", "share delta top (perturbation and IG), 95% CI lower bound", float(lo),
            prim_both, holds=bool(lo >= 0.70), note=f"95% CI {lo:.2f}-{hi:.2f}")
    # IG baseline
    rb = tp[tp.filter(like="ignoise_").notna().all(axis=1)] if tp.filter(like="ignoise_").shape[1] else tp.iloc[:0]
    if len(rb):
        top0 = rb[[f"ig_{b}" for b in bands]].idxmax(axis=1).str.replace("ig_", "")
        topn = rb[[f"ignoise_{b}" for b in bands]].idxmax(axis=1).str.replace("ignoise_", "")
        add("IG baseline", "top-band agreement, zero vs noise baseline (epochs)", float((top0 == topn).mean()),
            holds=bool((top0 == topn).mean() >= 0.70))
        add("IG baseline", "share delta top with noise baseline (epochs)", float((topn == "delta").mean()),
            float((top0 == "delta").mean()), holds=bool((topn == "delta").mean() >= 0.70))
    if "ig_completeness_err" in tp:
        add("IG steps", "median completeness error, 32 steps", float(tp.ig_completeness_err.median()),
            note="|sum IG - (f(x)-f(x0))| / |f(x)-f(x0)|")
    if "ig_completeness_err_64" in tp and tp.ig_completeness_err_64.notna().any():
        e32 = tp.loc[tp.ig_completeness_err_64.notna(), "ig_completeness_err"].median()
        e64 = tp.ig_completeness_err_64.median()
        add("IG steps", "median completeness error, 64 steps", float(e64), float(e32),
            holds=bool(abs(e64 - e32) < 0.05), note="holds = 32 steps already converged")
    # criterion re-applied per external cohort (true positives) and to the errors
    for (split, coh, oc), g in X.groupby(["split", "cohort", "outcome"]):
        if oc not in ("TP", "FP") or g.participant.nunique() < 3:
            continue
        Pg = g.groupby("participant")[[f"drop_{b}" for b in bands] + [f"ig_{b}" for b in bands]].mean()
        both = float(((Pg[[f"drop_{b}" for b in bands]].idxmax(axis=1) == "drop_delta") &
                      (Pg[[f"ig_{b}" for b in bands]].idxmax(axis=1) == "ig_delta")).mean())
        add("delta top for both methods", f"{split} {coh} {oc}", both, prim_both,
            holds=bool(both >= 0.70) if oc == "TP" else np.nan,
            note="FP rows are descriptive: delta-driven errors = N2 epochs with N3-like delta")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "rq5x_sensitivity.csv"), index=False)
