"""Small statistics helpers (no extra dependencies)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def icc(x, y, kind="2,1"):
    """Shrout & Fleiss ICC for two raters/occasions (n subjects x 2).

    '2,1' two-way random, absolute agreement, single measure.
    '3,1' two-way mixed, consistency, single measure.
    """
    X = np.column_stack([x, y]).astype(float)
    X = X[~np.isnan(X).any(1)]
    n, k = X.shape
    if n < 3:
        return np.nan
    gm = X.mean()
    ssr = k * ((X.mean(1) - gm) ** 2).sum()
    ssc = n * ((X.mean(0) - gm) ** 2).sum()
    sst = ((X - gm) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    if kind == "2,1":
        return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))
    return float((msr - mse) / (msr + (k - 1) * mse))


def bland_altman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    bias, sd = d.mean(), d.std(ddof=1)
    return {"bias": float(bias), "loa_low": float(bias - 1.96 * sd),
            "loa_high": float(bias + 1.96 * sd), "n": int(m.sum())}


def wilcoxon_paired(a, b, alternative="two-sided"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    if len(d) < 5 or np.allclose(d, 0):
        return {"n": int(len(d)), "median_diff": float(np.median(d)) if len(d) else np.nan,
                "p": np.nan, "W": np.nan}
    r = stats.wilcoxon(d, alternative=alternative, zero_method="wilcox")
    return {"n": int(len(d)), "median_diff": float(np.median(d)), "p": float(r.pvalue),
            "W": float(r.statistic)}


def boot_ci_median_diff(d, n_boot=2000, seed=0):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = [np.median(rng.choice(d, len(d))) for _ in range(n_boot)]
    return tuple(float(v) for v in np.percentile(bs, [2.5, 97.5]))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj = np.full_like(p, np.nan)
    m = np.sum(~np.isnan(p))
    running = 0.0
    for rank, i in enumerate(order):
        if np.isnan(p[i]):
            continue
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 4:
        return np.nan, np.nan
    r = stats.spearmanr(a[m], b[m])
    return float(r.correlation), float(r.pvalue)
