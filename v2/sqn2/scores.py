"""Rule-derived N3 EEG factor scores, overall N3 EEG Rating and profiles.

All scores are RULE-DERIVED research-prototype descriptors of the EEG signal.
They are not learned from, or validated against, expert quality ratings (none
exist in the data) and must not be described as clinical sleep quality.

Scaling: each raw metric is converted to a percentile (0-100) of the
distribution of that metric over EXPERT-SCORED N3 epochs of TRAINING
participants recorded with the SAME derivation (C4-M1 family vs Fpz-Cz).
A score of 50 therefore means "as strong as the median N3 epoch of the
training population for this montage".  References are re-fitted inside every
cross-validation fold, so test participants never influence their own scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_DEFS = {
    # factor           : [(metric, sign)]   sign=-1 -> lower is "stronger"
    "delta_intensity": [("logpow_sdelta", 1), ("logpow_delta", 1)],
    "sw_morphology": [("sw_time_frac", 1), ("cand_p2p_med", 1), ("cand_slope_up_med", 1)],
    "spectral_dominance": [("rel_delta", 1), ("logratio_delta_theta", 1),
                           ("logratio_delta_beta", 1), ("spectral_entropy", -1)],
}
FACTORS = ["delta_intensity", "sw_morphology", "spectral_dominance", "n3_continuity"]
PRIOR_WEIGHTS = {"delta_intensity": 0.35, "sw_morphology": 0.30,
                 "spectral_dominance": 0.20, "n3_continuity": 0.15}
EQUAL_WEIGHTS = {k: 0.25 for k in FACTORS}
CONTINUITY_HALF_WINDOW = 5        # epochs on each side (= +/- 2.5 min)


def montage_family(derivation: str) -> str:
    return "Fpz-Cz" if "fpz" in str(derivation).lower() else "central"


class FactorReference:
    """Per-montage sorted reference distributions (fit on training N3 epochs)."""

    def fit(self, df: pd.DataFrame):
        self.ref = {}
        for fam, g in df.groupby("montage"):
            self.ref[fam] = {m: np.sort(g[m].to_numpy(np.float64))
                             for defs in FACTOR_DEFS.values() for m, _ in defs}
        pooled = df
        self.ref["__pooled__"] = {m: np.sort(pooled[m].to_numpy(np.float64))
                                  for defs in FACTOR_DEFS.values() for m, _ in defs}
        return self

    def pct(self, values, metric, montage):
        ref = self.ref.get(montage, self.ref["__pooled__"])[metric]
        return 100.0 * np.searchsorted(ref, values, side="right") / max(len(ref), 1)


def factor_scores(df: pd.DataFrame, ref: FactorReference) -> pd.DataFrame:
    """Signal factors (delta, morphology, spectral) for every row of df."""
    out = pd.DataFrame(index=df.index)
    for fac, defs in FACTOR_DEFS.items():
        parts = []
        for metric, sign in defs:
            p = np.empty(len(df))
            for fam, idx in df.groupby("montage").groups.items():
                p[df.index.get_indexer(idx)] = ref.pct(df.loc[idx, metric].to_numpy(),
                                                       metric, fam)
            parts.append(p if sign > 0 else 100.0 - p)
        out[fac] = np.mean(parts, axis=0)
    return out


def continuity_score(pred_n3: np.ndarray, rec_ids: np.ndarray, epoch_idx: np.ndarray,
                     half_window: int = CONTINUITY_HALF_WINDOW) -> np.ndarray:
    """% of the surrounding +/-half_window epochs (same recording, true
    chronological neighbours) that are predicted N3.  Uses model output only,
    so it is available at inference time without expert labels."""
    out = np.full(len(pred_n3), np.nan)
    df = pd.DataFrame({"r": rec_ids, "e": epoch_idx, "p": pred_n3.astype(float),
                       "i": np.arange(len(pred_n3))})
    for _, g in df.groupby("r"):
        g = g.sort_values("e")
        n_max = int(g["e"].max()) + 1
        seq = np.full(n_max, np.nan)
        seq[g["e"].to_numpy()] = g["p"].to_numpy()
        valid = ~np.isnan(seq)
        s = np.where(valid, seq, 0.0)
        cs = np.concatenate([[0], np.cumsum(s)])
        cv = np.concatenate([[0], np.cumsum(valid)])
        e = g["e"].to_numpy()
        lo = np.clip(e - half_window, 0, n_max)
        hi = np.clip(e + half_window + 1, 0, n_max)
        num = cs[hi] - cs[lo] - s[e]
        den = cv[hi] - cv[lo] - 1
        out[g["i"].to_numpy()] = 100.0 * num / np.maximum(den, 1)
    return out


def overall_rating(scores: pd.DataFrame, weights=None) -> np.ndarray:
    w = weights or PRIOR_WEIGHTS
    tot = sum(w.values())
    return sum(scores[k].to_numpy() * (w[k] / tot) for k in FACTORS)


def rating_decomposition(scores: pd.DataFrame, weights=None) -> pd.DataFrame:
    """Exact additive decomposition: contribution_i = w_i * factor_i."""
    w = weights or PRIOR_WEIGHTS
    tot = sum(w.values())
    dec = pd.DataFrame({f"contrib_{k}": scores[k] * (w[k] / tot) for k in FACTORS})
    dec["rating"] = dec.sum(axis=1)
    return dec


def pca_weights(train_scores: pd.DataFrame) -> dict:
    """Data-derived weights: |loadings| of the first principal component of
    standardised factor scores on TRAINING N3 epochs, normalised to sum 1."""
    X = train_scores[FACTORS].to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    w = np.abs(vt[0])
    w = w / w.sum()
    return dict(zip(FACTORS, w))


# ---------------------------------------------------------------------------
# Episodes and whole-night profile
# ---------------------------------------------------------------------------
def find_episodes(is_n3: np.ndarray, min_epochs: int = 2, gap_tol: int = 2):
    """Return list of (start, end_exclusive, n_n3_epochs, n_interruptions).

    A run = consecutive N3 epochs.  Runs separated by <= gap_tol non-N3 epochs
    are merged into one episode; each merged gap is an interruption.  Episodes
    with fewer than ``min_epochs`` N3 epochs are dropped.
    """
    x = np.asarray(is_n3, bool)
    if not x.any():
        return []
    d = np.diff(np.concatenate([[0], x.astype(int), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    eps = []
    cs, ce, n3, intr = starts[0], ends[0], ends[0] - starts[0], 0
    for s, e in zip(starts[1:], ends[1:]):
        if s - ce <= gap_tol:
            ce, n3, intr = e, n3 + (e - s), intr + 1
        else:
            eps.append((cs, ce, n3, intr))
            cs, ce, n3, intr = s, e, e - s, 0
    eps.append((cs, ce, n3, intr))
    return [e for e in eps if e[2] >= min_epochs]


def night_profile(seq_n3, seq_rating=None, seq_cont=None, seq_rel=None,
                  rel_high=None, epoch_minutes=0.5, gap_tol=2):
    """Overnight N3 EEG Profile for one recording (arrays in chronological order;
    NaN rating where not rated)."""
    seq_n3 = np.asarray(seq_n3, bool)
    n = len(seq_n3)
    eps = find_episodes(seq_n3, gap_tol=gap_tol)
    durs = np.array([e[2] * epoch_minutes for e in eps]) if eps else np.array([])
    prof = {
        "analysed_minutes": n * epoch_minutes,
        "n3_minutes": seq_n3.sum() * epoch_minutes,
        "n3_pct_of_window": 100.0 * seq_n3.mean() if n else np.nan,
        "n_episodes": len(eps),
        "median_episode_min": float(np.median(durs)) if len(durs) else 0.0,
        "longest_episode_min": float(durs.max()) if len(durs) else 0.0,
        "total_interruptions": int(sum(e[3] for e in eps)),
        "n3_first_half_fraction": (seq_n3[: n // 2].sum() / max(seq_n3.sum(), 1)
                                   if n else np.nan),
    }
    if seq_rating is not None:
        r = np.asarray(seq_rating, float)[seq_n3]
        r = r[~np.isnan(r)]
        prof.update({
            "median_rating": float(np.median(r)) if len(r) else np.nan,
            "rating_iqr": float(np.subtract(*np.percentile(r, [75, 25]))) if len(r) > 1 else np.nan,
            "high_rating_prop": float((r >= 50).mean()) if len(r) else np.nan,
        })
    if seq_cont is not None:
        c = np.asarray(seq_cont, float)[seq_n3]
        prof["median_continuity"] = float(np.nanmedian(c)) if len(c) else np.nan
    if seq_rel is not None and rel_high is not None:
        rr = np.asarray(seq_rel, float)[seq_n3]
        prof["reliable_coverage"] = float((rr >= rel_high).mean()) if len(rr) else np.nan
    return prof, eps
