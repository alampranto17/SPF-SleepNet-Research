"""Epoch features for SleepQualityNet v2.

* Spectral features: Welch PSD (4-s windows, 50 % overlap, 0.25 Hz resolution)
  computed for all epochs of a recording at once.
* Slow-wave detector: zero-crossing full-wave detector on the continuous
  (whole-night) 0.3-4 Hz signal, replacing the v1 heuristic peak counter.

  A candidate wave = negative half-wave (down-going zero crossing -> next
  up-going zero crossing) followed by a positive half-wave (-> next down-going
  crossing).  Duration-valid waves: negative half-wave 0.25-1.0 s and full wave
  0.5-2.0 s (0.5-2 Hz, AASM slow-wave frequency band).  AASM slow waves
  additionally require peak-to-peak amplitude >= 75 uV.

  Per wave: negative-peak amplitude, positive-peak amplitude, peak-to-peak
  amplitude, negative / positive / full durations, down-slope (zero crossing
  -> negative peak) and up-slope (negative peak -> next zero crossing).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.integrate import trapezoid

FS = 100
EPOCH = 3000

BANDS = {"sdelta": (0.5, 2.0), "delta": (0.5, 4.0), "theta": (4.0, 8.0),
         "alpha": (8.0, 12.0), "sigma": (12.0, 16.0), "beta": (16.0, 30.0)}

SW_P2P_UV = 75.0


def spectral_features(epochs: np.ndarray) -> dict:
    x = np.asarray(epochs, dtype=np.float32)
    f, pxx = welch(x, fs=FS, nperseg=400, noverlap=200, axis=1)
    out = {}
    m_tot = (f >= 0.5) & (f < 30.0)
    tot = trapezoid(pxx[:, m_tot], f[m_tot], axis=1) + 1e-6
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        p = trapezoid(pxx[:, m], f[m], axis=1) + 1e-6
        out[f"pow_{name}"] = p
        out[f"logpow_{name}"] = np.log10(p)
        if name != "sdelta":
            out[f"rel_{name}"] = p / tot
    out["pow_total"] = tot
    out["logratio_delta_theta"] = np.log10(out["pow_delta"] / out["pow_theta"])
    out["logratio_delta_alpha"] = np.log10(out["pow_delta"] / out["pow_alpha"])
    out["logratio_delta_beta"] = np.log10(out["pow_delta"] / out["pow_beta"])
    p = pxx[:, m_tot]
    p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
    out["spectral_entropy"] = -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(p.shape[1])
    # time-domain delta amplitude (RMS of 0.5-4 Hz) and envelope proxy
    return {k: v.astype(np.float32) for k, v in out.items()}


def detect_slow_waves(signal: np.ndarray, fs: int = FS):
    """Return a structured array of duration-valid candidate waves."""
    sos = butter(2, [0.3, 4.0], btype="band", fs=fs, output="sos")
    x = sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))
    s = np.signbit(x)                      # True = negative
    down = np.flatnonzero(~s[:-1] & s[1:]) + 1   # first negative sample
    up = np.flatnonzero(s[:-1] & ~s[1:]) + 1     # first non-negative sample
    if len(down) < 2 or len(up) < 1:
        return _empty_waves()
    rows = []
    ui = np.searchsorted(up, down)          # first up crossing after each down
    for k in range(len(down) - 1):
        if ui[k] >= len(up):
            break
        d, u = down[k], up[ui[k]]
        d2 = down[k + 1]
        if not (d < u < d2):
            continue
        neg_dur = (u - d) / fs
        full_dur = (d2 - d) / fs
        if not (0.25 <= neg_dur <= 1.0 and 0.5 <= full_dur <= 2.0):
            continue
        seg_n = x[d:u]
        seg_p = x[u:d2]
        i_n = int(np.argmin(seg_n))
        i_p = int(np.argmax(seg_p))
        neg = seg_n[i_n]
        pos = seg_p[i_p]
        t_neg = d + i_n
        rows.append((t_neg, neg, pos, pos - neg, neg_dur, (d2 - u) / fs, full_dur,
                     abs(neg) / max((i_n + 1) / fs, 1e-3),
                     abs(neg) / max((u - t_neg) / fs, 1e-3), d, d2))
    if not rows:
        return _empty_waves()
    return np.array(rows, dtype=_WAVE_DTYPE)


_WAVE_DTYPE = [("t_neg", "i8"), ("neg_amp", "f4"), ("pos_amp", "f4"), ("p2p", "f4"),
               ("neg_dur", "f4"), ("pos_dur", "f4"), ("full_dur", "f4"),
               ("slope_down", "f4"), ("slope_up", "f4"), ("start", "i8"), ("end", "i8")]


def _empty_waves():
    return np.zeros(0, dtype=_WAVE_DTYPE)


def slow_wave_epoch_features(waves, n_epochs: int, p2p_thr: float = SW_P2P_UV):
    """Aggregate wave table to per-epoch morphology features."""
    out = {k: np.zeros(n_epochs, np.float32) for k in (
        "sw_count_per_min", "sw_time_frac", "sw_p2p_med", "sw_negamp_med",
        "sw_slope_up_med", "sw_slope_down_med", "sw_negdur_med", "sw_posdur_med",
        "sw_fulldur_med", "cand_p2p_med", "cand_slope_up_med", "cand_count_per_min")}
    if len(waves) == 0:
        return out
    ep = np.clip(waves["t_neg"] // EPOCH, 0, n_epochs - 1)
    aasm = waves["p2p"] >= p2p_thr
    order = np.argsort(ep, kind="stable")
    ep_s, w_s, a_s = ep[order], waves[order], aasm[order]
    bounds = np.searchsorted(ep_s, np.arange(n_epochs + 1))
    for e in range(n_epochs):
        lo, hi = bounds[e], bounds[e + 1]
        if hi <= lo:
            continue
        w = w_s[lo:hi]
        a = a_s[lo:hi]
        out["cand_count_per_min"][e] = (hi - lo) * 2.0
        out["cand_p2p_med"][e] = np.median(w["p2p"])
        out["cand_slope_up_med"][e] = np.median(w["slope_up"])
        if a.any():
            wa = w[a]
            out["sw_count_per_min"][e] = a.sum() * 2.0
            # time covered by AASM waves inside this epoch
            s0 = np.maximum(wa["start"], e * EPOCH)
            s1 = np.minimum(wa["end"], (e + 1) * EPOCH)
            out["sw_time_frac"][e] = min(1.0, np.clip(s1 - s0, 0, None).sum() / EPOCH)
            out["sw_p2p_med"][e] = np.median(wa["p2p"])
            out["sw_negamp_med"][e] = np.median(-wa["neg_amp"])
            out["sw_slope_up_med"][e] = np.median(wa["slope_up"])
            out["sw_slope_down_med"][e] = np.median(wa["slope_down"])
            out["sw_negdur_med"][e] = np.median(wa["neg_dur"])
            out["sw_posdur_med"][e] = np.median(wa["pos_dur"])
            out["sw_fulldur_med"][e] = np.median(wa["full_dur"])
    return out


def recording_features(epochs: np.ndarray) -> dict:
    """All per-epoch features for one recording (epochs must be contiguous)."""
    epochs = np.asarray(epochs, dtype=np.float32)
    feats = spectral_features(epochs)
    waves = detect_slow_waves(epochs.reshape(-1))
    feats.update(slow_wave_epoch_features(waves, len(epochs)))
    return feats


# Handcrafted features fed to the detection network (RQ4 "feature fusion").
DETECTION_FEATURES = [
    "logpow_sdelta", "logpow_delta", "logpow_theta", "logpow_alpha", "logpow_sigma",
    "logpow_beta", "rel_delta", "rel_theta", "rel_alpha", "rel_sigma", "rel_beta",
    "logratio_delta_theta", "logratio_delta_alpha", "logratio_delta_beta",
    "spectral_entropy", "sw_count_per_min", "sw_time_frac", "log_cand_p2p_med",
    "cand_slope_up_med",
]


def detection_matrix(feat_table) -> np.ndarray:
    """Build the (n, len(DETECTION_FEATURES)) matrix from a feature dict/DataFrame."""
    cols = []
    for k in DETECTION_FEATURES:
        if k == "log_cand_p2p_med":
            cols.append(np.log1p(np.asarray(feat_table["cand_p2p_med"], np.float32)))
        else:
            cols.append(np.asarray(feat_table[k], np.float32))
    return np.stack(cols, axis=1)


class RobustScaler:
    """Median / IQR scaler fitted on training participants only, with clipping."""

    def fit(self, X):
        self.med = np.nanmedian(X, axis=0)
        q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
        self.iqr = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)
        return self

    def transform(self, X):
        Z = (X - self.med) / self.iqr
        return np.clip(np.nan_to_num(Z), -10, 10).astype(np.float32)
