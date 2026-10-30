"""RQ5 explainability with explicitly specified methods.

Methods (answers to the supervisor's method questions):
  * Band-stop perturbation: 4th-order Butterworth band-stop, zero-phase
    (scipy sosfiltfilt), applied to the target epoch only; the handcrafted
    features are RECOMPUTED from the perturbed epoch (coherent intervention);
    neighbouring epochs are unchanged.  Effect = P(N3) before - after.
  * Integrated gradients (IG): baseline = all-zero epoch, 32 Riemann steps,
    attribution w.r.t. the target-epoch logit, handcrafted features held fixed
    (explains the raw-EEG branch).  Vanilla gradient x input is also stored.
    Band attribution (band-decomposed IG): the input is split into band-limited
    components x_b (4th-order zero-phase Butterworth band-pass); contribution_b
    = sum_t x_b(t) * mean-path-gradient(t), reported as signed share of
    sum_b |contribution_b|.
  * SHAP: TreeExplainer on a Gradient-Boosting REGRESSION SURROGATE trained to
    reproduce the network's logit from the 19 standardised handcrafted
    features; fidelity = R^2 on held-out epochs.  Group attribution =
    |sum of SHAP values within a feature group|, averaged over epochs.
  * Attention-slow-wave alignment: Spearman correlation between the 32
    attention weights and the fraction of each of the 32 time segments
    covered by AASM slow waves; enrichment = attention mass on segments with
    >50 % slow-wave coverage divided by the fraction of such segments.
  * Exact factor decomposition of the overall rating: contribution_i =
    w_i * factor_i (sums to the rating up to floating-point error).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt

from . import features as FT
from . import scores as SC
from . import stats as ST
from .models import SleepQualityNetV2, CONTEXT_RADIUS
from .train import VARIANTS, PRIMARY_VARIANT, build_model, load_run

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "sigma": (12.0, 16.0), "beta": (16.0, 30.0)}
FEATURE_GROUPS = {
    "delta": FT.DELTA_FEATURES,
    "theta": ["logpow_theta", "rel_theta"], "alpha": ["logpow_alpha", "rel_alpha"],
    "sigma": ["logpow_sigma", "rel_sigma"], "beta": ["logpow_beta", "rel_beta"],
    "other": ["spectral_entropy"],
}
HALF_WIN = CONTEXT_RADIUS   # window covers the full temporal receptive field


def _epoch_features(x_uv):
    ft = FT.spectral_features(x_uv[None, :])
    ft.update(FT.slow_wave_epoch_features(FT.detect_slow_waves(x_uv), 1))
    return FT.detection_matrix(ft)


_BAND_SOS = {b: butter(4, [lo, hi], btype="band", fs=FT.FS, output="sos")
             for b, (lo, hi) in BANDS.items()}


def _band_contributions(x, gbar):
    """Band-decomposed attribution: x = sum_b x_b (+ residual outside 0.5-30 Hz);
    contribution_b = sum_t x_b(t) * gbar(t).  For IG (gbar = path-averaged
    gradient) the band contributions add up to the 0.5-30 Hz part of the total
    IG attribution.  Returned as signed shares of sum_b |contribution_b|."""
    c = {b: float(np.dot(sosfiltfilt(sos, x), gbar)) for b, sos in _BAND_SOS.items()}
    tot = sum(abs(v) for v in c.values()) + 1e-12
    return {b: v / tot for b, v in c.items()}


def _sw_coverage(x_uv, nseg=32):
    w = FT.detect_slow_waves(x_uv)
    mask = np.zeros(len(x_uv))
    for r in w[w["p2p"] >= FT.SW_P2P_UV]:
        mask[max(r["start"], 0):min(r["end"], len(x_uv))] = 1
    return np.array([seg.mean() for seg in np.array_split(mask, nseg)])


def rq5(corpus, run_dir, out_dir, oof, S=None, n_per_fold=120, per_participant=8,
        ig_steps=32, seed=0, variant=PRIMARY_VARIANT):
    dev = corpus.device
    rng = np.random.default_rng(seed)
    v = VARIANTS[variant]
    rows, shap_rows, fid = [], [], []
    for f in sorted(oof.fold.unique()):
        r = load_run(os.path.join(run_dir, f"{variant}__fold{f}.npz"))
        meta = json.loads(r["meta"])
        model = build_model(v, corpus.Fraw.shape[1]).to(dev)
        model.load_state_dict(torch.load(os.path.join(run_dir, f"{variant}__fold{f}.pt"),
                                         map_location=dev))
        model.eval()
        tr = corpus.idx_of(meta["train_recs"])
        tr = tr[corpus.ep.label.to_numpy()[tr] >= 0]
        scaler = FT.RobustScaler().fit(corpus.Fraw[tr])
        Fs = torch.from_numpy(scaler.transform(corpus.Fraw)).to(dev)
        g = oof[(oof.fold == f) & (oof.pred == 1) & (oof.y == 1) & oof.is_primary]
        parts_ = [d.sample(min(len(d), per_participant), random_state=seed)
                  for _, d in g.groupby("participant")]
        pick = pd.concat(parts_) if parts_ else g.iloc[:0]
        pick = pick.sample(min(len(pick), n_per_fold), random_state=seed)
        for _, row in pick.iterrows():
            i = int(row.idx)
            rec = corpus.rec.loc[row.rec]
            o, n = int(rec.offset), int(rec.n)
            t = i - o
            lo, hi = max(0, t - HALF_WIN), min(n, t + HALF_WIN + 1)
            pos = t - lo
            xw = corpus.X[o + lo:o + hi].float().clone()
            fw = Fs[o + lo:o + hi].clone()
            with torch.no_grad():
                base = torch.sigmoid(model.predict_sequence(xw, fw)[pos]).item()
            x_uv = xw[pos].cpu().numpy() * float(rec.raw_scale)
            out = {"fold": f, "participant": row.participant, "cohort": row.cohort,
                   "idx": i, "p_orig": base}
            for b, (blo, bhi) in BANDS.items():
                sos = butter(4, [blo, bhi], btype="bandstop", fs=FT.FS, output="sos")
                xp = sosfiltfilt(sos, x_uv)
                xw2, fw2 = xw.clone(), fw.clone()
                xw2[pos] = torch.from_numpy(xp / float(rec.raw_scale)).to(dev)
                fw2[pos] = torch.from_numpy(scaler.transform(_epoch_features(xp))[0]).to(dev)
                with torch.no_grad():
                    pp = torch.sigmoid(model.predict_sequence(xw2, fw2)[pos]).item()
                out[f"drop_{b}"] = base - pp
            # integrated gradients on the raw target epoch
            alphas = torch.linspace(1.0 / ig_steps, 1.0, ig_steps, device=dev)
            xb = xw.unsqueeze(0).repeat(ig_steps, 1, 1)
            tgt = (alphas[:, None] * xw[pos][None, :]).detach().requires_grad_(True)
            xb = torch.cat([xb[:, :pos], tgt[:, None], xb[:, pos + 1:]], 1)
            fb = fw.unsqueeze(0).repeat(ig_steps, 1, 1)
            logits = model(xb, fb)[:, pos]
            grads = torch.autograd.grad(logits.sum(), tgt)[0]
            xs = xw[pos].detach().cpu().numpy().astype(np.float64)
            gbar = grads.mean(0).detach().cpu().numpy().astype(np.float64)
            g1 = grads[-1].detach().cpu().numpy().astype(np.float64)
            out["ig_total"] = float(np.dot(xs, gbar))
            for b, val in _band_contributions(xs, gbar).items():
                out[f"ig_{b}"] = val
            for b, val in _band_contributions(xs, g1).items():
                out[f"gradxinput_{b}"] = val
            # attention alignment
            if v["use_attention"]:
                with torch.no_grad():
                    _, a = model.encoder(xw[pos][None, None], return_attn=True)
                a = a[0].float().cpu().numpy()
                cov = _sw_coverage(x_uv)
                rho = ST.spearman(a, cov)[0] if cov.std() > 0 else np.nan
                hb = cov > 0.5
                enr = (a[hb].sum() / hb.mean()) if hb.any() else np.nan
                out.update({"attn_sw_spearman": rho, "attn_sw_enrichment": enr,
                            "attn_entropy": float(-(a * np.log(a + 1e-12)).sum())})
            rows.append(out)
        # SHAP surrogate of the network logit from handcrafted features
        try:
            import shap
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.metrics import r2_score
            gg = oof[(oof.fold == f) & oof.is_primary & (oof.label >= 0)]
            gg = gg.sample(min(len(gg), 6000), random_state=seed)
            Xs = Fs[torch.as_tensor(gg.idx.to_numpy().copy(), device=Fs.device)].cpu().numpy()
            yl = gg["logit"].to_numpy()
            half = len(gg) // 2
            sur = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=seed)
            sur.fit(Xs[:half], yl[:half])
            r2 = r2_score(yl[half:], sur.predict(Xs[half:]))
            sv = shap.TreeExplainer(sur).shap_values(Xs[half:])
            names = FT.DETECTION_FEATURES
            grp = {gname: np.abs(sv[:, [names.index(c) for c in cols]].sum(1)).mean()
                   for gname, cols in FEATURE_GROUPS.items()}
            shap_rows.append({"fold": f, "fidelity_r2": r2, **{f"shap_{k}": val for k, val in grp.items()},
                              **{f"shapfeat_{n}": float(np.abs(sv[:, j]).mean())
                                 for j, n in enumerate(names)}})
        except Exception as exc:  # shap not installed
            shap_rows.append({"fold": f, "error": str(exc)})
    X = pd.DataFrame(rows)
    X.to_csv(os.path.join(out_dir, "rq5_epoch_explanations.csv"), index=False)
    sh = pd.DataFrame(shap_rows)
    sh.to_csv(os.path.join(out_dir, "rq5_shap_surrogate.csv"), index=False)
    # participant-level agreement
    bands = list(BANDS)
    P = X.groupby("participant")[[f"drop_{b}" for b in bands] + [f"ig_{b}" for b in bands]].mean()
    agree = []
    for p, r_ in P.iterrows():
        d = np.array([r_[f"drop_{b}"] for b in bands])
        ig = np.array([r_[f"ig_{b}"] for b in bands])
        agree.append({"participant": p, "rank_rho_perturb_vs_ig": ST.spearman(d, ig)[0],
                      "top_perturb": bands[int(np.argmax(d))], "top_ig": bands[int(np.argmax(ig))]})
    A = pd.DataFrame(agree)
    A["top_agree"] = A.top_perturb == A.top_ig
    A.to_csv(os.path.join(out_dir, "rq5_method_agreement_by_participant.csv"), index=False)
    frac_delta_pert = float((A.top_perturb == "delta").mean()) if len(A) else np.nan
    frac_delta_ig = float((A.top_ig == "delta").mean()) if len(A) else np.nan
    frac_both = float(((A.top_perturb == "delta") & (A.top_ig == "delta")).mean()) if len(A) else np.nan
    shap_top, fidelity = None, np.nan
    if "fidelity_r2" in sh:
        fidelity = float(sh.fidelity_r2.mean())
        cols = [c for c in sh.columns if c.startswith("shap_")]
        shap_top = sh[cols].mean().idxmax().replace("shap_", "")
    dec_err = np.nan
    if S is not None:
        dec = SC.rating_decomposition(S)
        dec_err = float(np.nanmax(np.abs(dec.rating - S.rating_prior)))
    delta_sign = float((X.drop_delta > 0).mean()) if len(X) else np.nan
    alpha_ctrl = float(X.drop_alpha.median()) if len(X) else np.nan
    attn = (float(X.attn_sw_spearman.median()), float(X.attn_sw_enrichment.median())) \
        if "attn_sw_spearman" in X else (np.nan, np.nan)
    ok = (frac_both >= 0.70) and (shap_top == "delta") and (fidelity >= 0.70)
    partial = (frac_delta_pert >= 0.70) or (frac_delta_ig >= 0.70) or (shap_top == "delta")
    summary = {
        "n_epochs": int(len(X)), "n_participants": int(X.participant.nunique()) if len(X) else 0,
        "median_delta_drop": float(X.drop_delta.median()) if len(X) else np.nan,
        "median_alpha_drop(control)": alpha_ctrl, "frac_epochs_delta_drop_positive": delta_sign,
        "frac_participants_top_band_delta_perturbation": frac_delta_pert,
        "frac_participants_top_band_delta_IG": frac_delta_ig,
        "frac_participants_delta_top_both": frac_both,
        "median_rank_rho_perturb_vs_ig": float(A.rank_rho_perturb_vs_ig.median()) if len(A) else np.nan,
        "top_band_agreement_rate": float(A.top_agree.mean()) if len(A) else np.nan,
        "shap_surrogate_fidelity_r2": fidelity, "shap_top_group": shap_top,
        "attention_sw_spearman_median": attn[0], "attention_sw_enrichment_median": attn[1],
        "exact_decomposition_max_abs_error": dec_err,
    }
    summary["observed"] = (f"delta top in {frac_both:.0%} of participants for both perturbation "
                           f"and IG (perturb {frac_delta_pert:.0%}, IG {frac_delta_ig:.0%}); "
                           f"SHAP top group={shap_top} (fidelity R2={fidelity:.2f}); "
                           f"attention-SW rho={attn[0]:.2f}")
    summary["status"] = "SUPPORTED" if ok else ("PARTIALLY_SUPPORTED" if partial else "NOT_SUPPORTED")
    with open(os.path.join(out_dir, "rq5_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    return summary
