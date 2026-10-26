# Local exporter: identical to sqn2/data.py (self-contained). Usage:
#   python export_compact.py --isruc1 D:/AI_Project/ISRUC-Sleep-I --isruc3 D:/AI_Project/ISRUC-Sleep-III --hmc D:/AI_Project/HMC --out D:/AI_Project/sqn2_cache.zip
"""Data layer for SleepQualityNet v2.

Cohorts (all central-mastoid EEG, AASM-era scoring):
  * ISRUC-Sleep subgroup I   (100 participants, sleep disorders)      -> training / CV
  * HMC (Haaglanden Medisch Centrum, PhysioNet, ~150 clinical PSGs)   -> training / CV
  * ISRUC-Sleep subgroup II  (8 healthy, 2 sessions each)             -> external, test-retest
  * ISRUC-Sleep subgroup III (10 healthy)                             -> external
  * Dreem DOD-H (25 healthy) and DOD-O (55 OSA), 5 scorers each       -> external, graded expert agreement

Key differences from the v1 notebook (motivated by the v1 audit):
1. Channel selection accepts the A1/M1 spellings of the same central-mastoid
   derivation (v1 missed 'C4-M1' and silently used E1-M2, an EOG channel, for
   83/100 ISRUC-I subjects) and NEVER falls back to a non-EEG channel.
2. Every epoch of the analysis window is kept in chronological order with its
   index; the v1 artifact / quality rules are stored as flags, not deletions.
3. Filtering on the continuous recording, then resampling to 100 Hz.
4. All individual expert hypnograms are stored (``labels_multi``: ISRUC 2,
   DOD 5 scorers) so the model can be compared with inter-rater agreement and
   the factor scores can be tested against GRADED expert agreement.

Self-contained (numpy / scipy / pandas / mne; h5py only for DOD) so it can also
run on a local PC as a command-line exporter:

    python export_compact.py --isruc1 D:/AI_Project/ISRUC-Sleep-I \
        --isruc3 D:/AI_Project/ISRUC-Sleep-III --hmc D:/AI_Project/HMC \
        --out D:/AI_Project/sqn2_cache.zip
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly, welch
from scipy.integrate import trapezoid

TARGET_FS = 100            # Hz, analysis sampling rate
EPOCH_SEC = 30
EPOCH_SAMPLES = TARGET_FS * EPOCH_SEC
CACHE_VERSION = 3

# Stage codes used everywhere: 0 W, 1 N1, 2 N2, 3 N3, 4 REM, -1 unscored/movement
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_MAP_ISRUC = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: -1, 7: -1, 9: -1}
STAGE_MAP_HMC = {"Sleep stage W": 0, "Sleep stage N1": 1, "Sleep stage N2": 2,
                 "Sleep stage N3": 3, "Sleep stage R": 4}

# Central-mastoid EEG derivations, in priority order.  A1/M1 (A2/M2) are two
# names for the same mastoid reference.
EEG_CANDIDATES = ["C4-A1", "C4-M1", "C3-A2", "C3-M2"]
ISRUC_EEG_CANDIDATES = EEG_CANDIDATES          # backward-compatible name

COHORT_PREFIX = {"ISRUC-I": "ISRUC1", "ISRUC-II": "ISRUC2", "ISRUC-III": "ISRUC3",
                 "HMC": "HMC", "DOD-H": "DODH", "DOD-O": "DODO"}


def _norm_ch(name: str) -> str:
    n = str(name).upper().replace("EEG", "").replace(" ", "").replace("_", "-")
    return n.replace("/", "-").strip("-").strip()


def pick_channel(ch_names, candidates=EEG_CANDIDATES):
    """First channel in ``ch_names`` matching ``candidates`` (case, spaces,
    'EEG' prefix, '_' and '/' ignored).  None if absent -- never a fallback."""
    normed = {_norm_ch(c): c for c in ch_names}
    for cand in candidates:
        if _norm_ch(cand) in normed:
            return normed[_norm_ch(cand)]
    return None


def standard_derivation(channel: str) -> str:
    n = _norm_ch(channel)
    return {"C4-A1": "C4-M1", "C4-M1": "C4-M1", "C3-A2": "C3-M2", "C3-M2": "C3-M2"}.get(n, channel)


def _to_uv(sig):
    """Guard against signals stored in volts (or nV) instead of microvolts."""
    a = np.nanpercentile(np.abs(sig), 99)
    if 0 < a < 0.01:
        return sig * 1e6
    if a > 1e5:
        return sig * 1e-3
    return sig


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
def _band_power(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    if not m.any():
        return np.zeros(pxx.shape[0])
    return trapezoid(pxx[:, m], f[m], axis=1)


def epoch_qc(epochs_native: np.ndarray, fs: float) -> dict:
    """Per-epoch quality metrics on the 0.5-45 Hz native-rate signal.

    Same thresholds as the v1 ``detect_artifacts`` / ``compute_signal_quality``
    / clipping rules, stored as flags instead of deleting epochs.
    """
    x = epochs_native.astype(np.float64)
    n = x.shape[0]
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + 1e-8
    peak_abs = np.abs(x).max(axis=1)
    zmax = (np.abs(x - mean) / std).max(axis=1)
    n_clipped = (np.abs(x - mean) > 4.0 * std).sum(axis=1)
    rms = np.sqrt((x ** 2).mean(axis=1))
    nper = int(min(x.shape[1], 4 * fs))
    f, pxx = welch(x, fs=fs, nperseg=nper, axis=1)
    nyq = fs / 2.0
    hf_hi = min(40.0, nyq - 1.0)
    p_delta = _band_power(f, pxx, 0.5, 4.0)
    p_lf = _band_power(f, pxx, 0.5, 15.0)
    p_hf = _band_power(f, pxx, 30.0, hf_hi) if hf_hi > 30 else np.zeros(n)
    p_hf2 = _band_power(f, pxx, 30.0, min(45.0, nyq - 0.5)) if nyq > 31 else np.zeros(n)
    p_tot = _band_power(f, pxx, 0.5, min(45.0, nyq - 0.5))
    noise_ratio = p_hf / (p_lf + 1e-8)
    snr_db = 10 * np.log10((p_delta + 1e-12) / (p_hf2 + 1e-8))
    hf_ratio = p_hf2 / (p_tot + 1e-8)
    flat = rms < 1.0
    clip_flag = n_clipped > 100 * fs / 200.0
    artifact_flag = (peak_abs > 300) | (zmax > 5.0) | (p_hf > 0.5 * p_delta)
    q = np.ones(n) - 0.3 * (noise_ratio > 0.3) - 0.3 * (snr_db < 3) - 0.4 * flat
    lowq_flag = q <= 0.6
    return {
        "qc_peak_abs": peak_abs.astype(np.float32), "qc_zmax": zmax.astype(np.float32),
        "qc_n_clipped": n_clipped.astype(np.float32), "qc_rms": rms.astype(np.float32),
        "qc_noise_ratio": noise_ratio.astype(np.float32), "qc_snr_db": snr_db.astype(np.float32),
        "qc_hf_ratio": hf_ratio.astype(np.float32), "qc_flat": flat,
        "qc_score_v1": q.astype(np.float32), "flag_clip": clip_flag,
        "flag_artifact": artifact_flag, "flag_lowq": lowq_flag,
        "flag_any_v1": clip_flag | artifact_flag | lowq_flag,
    }


def process_signal(sig_uv: np.ndarray, fs: float, n_epochs: int):
    """Continuous signal (uV, native fs) -> (epochs at 100 Hz float16, qc dict, n)."""
    sig = np.nan_to_num(np.asarray(sig_uv, dtype=np.float64))
    ns_native = int(round(EPOCH_SEC * fs))
    n_epochs = int(min(n_epochs, len(sig) // ns_native))
    sig = sig[: n_epochs * ns_native]
    nyq = fs / 2.0
    sos_qc = butter(4, [0.5, min(45.0, nyq * 0.95)], btype="band", fs=fs, output="sos")
    qc = epoch_qc(sosfiltfilt(sos_qc, sig).reshape(n_epochs, ns_native), fs)
    sos_m = butter(4, [0.3, 35.0], btype="band", fs=fs, output="sos")
    sig_m = sosfiltfilt(sos_m, sig)
    if not np.isclose(fs, TARGET_FS):
        frac = Fraction(TARGET_FS / fs).limit_denominator(1000)
        sig_m = resample_poly(sig_m, frac.numerator, frac.denominator)
    sig_m = sig_m[: n_epochs * EPOCH_SAMPLES]
    if len(sig_m) < n_epochs * EPOCH_SAMPLES:
        sig_m = np.pad(sig_m, (0, n_epochs * EPOCH_SAMPLES - len(sig_m)))
    return sig_m.reshape(n_epochs, EPOCH_SAMPLES).astype(np.float16), qc, n_epochs


def consensus_labels(multi: np.ndarray) -> np.ndarray:
    """Majority vote over raters (rows = epochs, -1/-2 = not scored).  Ties are
    broken by the rater with the highest mean pairwise agreement with the other
    raters on this recording (the rule used for the Dreem consensus)."""
    k = multi.shape[1]
    agree = np.zeros(k)
    for a in range(k):
        for b in range(k):
            if a != b:
                m = (multi[:, a] >= 0) & (multi[:, b] >= 0)
                agree[a] += (multi[m, a] == multi[m, b]).mean() if m.any() else 0
    order = np.argsort(-agree)
    out = np.full(len(multi), -1, np.int8)
    for i, row in enumerate(multi):
        v = row[row >= 0]
        if not len(v):
            continue
        cnt = np.bincount(v, minlength=5)
        top = np.flatnonzero(cnt == cnt.max())
        if len(top) == 1:
            out[i] = top[0]
        else:
            for r in order:
                if row[r] in top:
                    out[i] = row[r]
                    break
    return out


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def _read_edf(path, preload=True):
    import mne
    p = str(path)
    if not p.lower().endswith(".edf"):            # ISRUC .rec files are EDF
        tmp = os.path.join(tempfile.gettempdir(), f"_sqn2_{os.getpid()}_{Path(p).stem}.edf")
        shutil.copy2(p, tmp)
        try:
            return mne.io.read_raw_edf(tmp, preload=preload, verbose="error", infer_types=True)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return mne.io.read_raw_edf(p, preload=preload, verbose="error", infer_types=True)


def parse_isruc_txt(path):
    stages = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or not s[0].isdigit():
                continue
            try:
                stages.append(STAGE_MAP_ISRUC.get(int(s.split()[0]), -1))
            except ValueError:
                continue
    return np.asarray(stages, dtype=np.int8)


def load_isruc(rec_path, txt_paths, cohort="ISRUC-I", sid=None, session=1, n_sessions=1):
    raw = _read_edf(rec_path)
    ch = pick_channel(raw.ch_names)
    if ch is None:
        raise RuntimeError(f"no central EEG derivation in {Path(rec_path).name}; "
                           f"channels={raw.ch_names}")
    fs = float(raw.info["sfreq"])
    sig = raw.get_data(picks=[ch], units="uV")[0]
    labs = [parse_isruc_txt(t) for t in txt_paths if t and os.path.exists(t)]
    if not labs:
        raise RuntimeError("no annotation file")
    n_sig = len(sig) // int(round(EPOCH_SEC * fs))
    n = min(n_sig, len(labs[0]))
    epochs, qc, n = process_signal(sig, fs, n)
    multi = np.full((n, max(2, len(labs))), -2, np.int8)
    for j, l in enumerate(labs):
        m = min(n, len(l))
        multi[:m, j] = l[:m]
    pre = COHORT_PREFIX[cohort]
    rec_id = f"{pre}_{int(sid):03d}" + (f"_s{session}" if n_sessions > 1 else "")
    meta = {"cohort": cohort, "participant": f"{pre}_{int(sid):03d}", "recording": rec_id,
            "night": int(session), "is_primary": session == 1, "channel_raw": ch,
            "derivation": standard_derivation(ch), "native_fs": fs, "n_epochs": int(n),
            "crop_start_s": 0.0, "label_epochs_in_file": int(len(labs[0])),
            "signal_epochs": int(n_sig), "n_raters": len(labs), "reference": "scorer 1",
            "source_file": Path(rec_path).name}
    return epochs, multi[:, 0].copy(), multi, qc, meta


def _hmc_desc(d):
    return str(d).split("@@")[0].strip()


def load_hmc(psg_path, scoring_path):
    import mne
    raw = _read_edf(psg_path)
    ch = pick_channel(raw.ch_names)
    if ch is None:
        raise RuntimeError(f"no central EEG derivation in {Path(psg_path).name}: {raw.ch_names}")
    ann = mne.read_annotations(str(scoring_path))
    fs = float(raw.info["sfreq"])
    stage = [(float(a["onset"]), float(a["duration"]), STAGE_MAP_HMC[_hmc_desc(a["description"])])
             for a in ann if _hmc_desc(a["description"]) in STAGE_MAP_HMC]
    if not stage:
        raise RuntimeError("no stage annotations")
    t0 = np.floor(min(s for s, _, _ in stage) / EPOCH_SEC) * EPOCH_SEC   # first scored epoch
    t1 = min(raw.n_times / fs, max(s + d for s, d, _ in stage))          # end of last scored
    sig = raw.get_data(picks=[ch], units="uV", start=int(round(t0 * fs)),
                       stop=int(round(t1 * fs)))[0]
    n = len(sig) // int(round(EPOCH_SEC * fs))
    labels = np.full(n, -1, np.int8)
    for s, d, code in stage:
        si = int(np.floor((s - t0) / EPOCH_SEC + 1e-6))
        ei = int(np.ceil((s - t0 + d) / EPOCH_SEC - 1e-6))
        labels[max(si, 0):min(max(ei, si + 1), n)] = code
    epochs, qc, n = process_signal(sig, fs, n)
    labels = labels[:n]
    sn = re.search(r"(SN\d+)", Path(psg_path).name)
    key = sn.group(1) if sn else Path(psg_path).stem
    meta = {"cohort": "HMC", "participant": f"HMC_{key}", "recording": f"HMC_{key}",
            "night": 1, "is_primary": True, "channel_raw": ch,
            "derivation": standard_derivation(ch), "native_fs": fs, "n_epochs": int(n),
            "crop_start_s": float(t0), "n_raters": 1, "reference": "single technician (AASM v2.4)",
            "source_file": Path(psg_path).name}
    multi = np.full((n, 2), -2, np.int8)
    multi[:, 0] = labels
    return epochs, labels, multi, qc, meta


def _h5_datasets(h5):
    out = {}

    def visit(name, obj):
        import h5py
        if isinstance(obj, h5py.Dataset):
            out[name] = obj
    h5.visititems(visit)
    return out


def dod_short_id(rid):
    return f"{rid[:8]}{rid[-4:]}"


def load_dod(h5_path, scorer_hyps=None, cohort="DOD-H"):
    """Dreem DOD .h5 record.  ``scorer_hyps``: {scorer_name: list of stage codes}."""
    import h5py
    with h5py.File(h5_path, "r") as h5:
        ds = _h5_datasets(h5)
        eeg_keys = [k for k in ds if "eeg" in k.lower()] or list(ds)
        ch_key = pick_channel([Path(k).name for k in eeg_keys])
        if ch_key is None:
            raise RuntimeError(f"no central EEG in {Path(h5_path).name}: {eeg_keys}")
        full = [k for k in eeg_keys if Path(k).name == ch_key][0]
        d = ds[full]
        fs = d.attrs.get("fs", None)
        if fs is None:
            parent = h5[str(Path(full).parent)] if str(Path(full).parent) != "." else h5
            fs = parent.attrs.get("fs", h5.attrs.get("fs", 250.0))
        fs = float(np.asarray(fs).ravel()[0])
        sig = _to_uv(np.asarray(d[()], dtype=np.float64).ravel())
        h5_hyp = None
        for k in ds:
            if Path(k).name.lower() == "hypnogram":
                h5_hyp = np.asarray(ds[k][()]).astype(int).ravel()
    n_sig = len(sig) // int(round(EPOCH_SEC * fs))
    hyps = [np.asarray(v, int) for _, v in sorted((scorer_hyps or {}).items())]
    if hyps:
        n = min([n_sig] + [len(h) for h in hyps])
        multi = np.full((n, len(hyps)), -1, np.int8)
        for j, h in enumerate(hyps):
            multi[:, j] = np.clip(h[:n], -1, 4)
        labels = consensus_labels(multi)
        ref = f"consensus of {len(hyps)} scorers"
    elif h5_hyp is not None:
        n = min(n_sig, len(h5_hyp))
        labels = np.clip(h5_hyp[:n], -1, 4).astype(np.int8)
        multi = np.full((n, 2), -2, np.int8)
        multi[:, 0] = labels
        ref = "h5 hypnogram"
    else:
        raise RuntimeError("no hypnogram")
    epochs, qc, n = process_signal(sig, fs, n)
    rid = Path(h5_path).stem
    pre = COHORT_PREFIX[cohort]
    sid = dod_short_id(rid)
    meta = {"cohort": cohort, "participant": f"{pre}_{sid}", "recording": f"{pre}_{sid}",
            "night": 1, "is_primary": True, "channel_raw": ch_key,
            "derivation": standard_derivation(ch_key), "native_fs": fs, "n_epochs": int(n),
            "crop_start_s": 0.0, "n_raters": len(hyps) if hyps else 1, "reference": ref,
            "source_file": Path(h5_path).name, "record_uuid": rid}
    return epochs, labels[:n], multi[:n], qc, meta


# ---------------------------------------------------------------------------
# Participant metadata (age / sex / diagnosis) -- only if a sheet exists
# ---------------------------------------------------------------------------
def _find_col(cols, *keys):
    for c in cols:
        lc = str(c).lower()
        if any(k in lc for k in keys):
            return c
    return None


def read_isruc_details(xlsx_path, prefix):
    """Parse an ISRUC 'Details_subgroup_*' spreadsheet -> {participant: {...}}."""
    import pandas as pd
    for header in range(0, 4):
        try:
            df = pd.read_excel(xlsx_path, header=header)
        except Exception:
            return {}
        cs = _find_col(df.columns, "subject", "patient", "id")
        ca = _find_col(df.columns, "age")
        if cs is None or ca is None:
            continue
        cx = _find_col(df.columns, "sex", "gender")
        cd = _find_col(df.columns, "diagnos")
        out = {}
        for _, r in df.iterrows():
            m = re.search(r"\d+", str(r[cs]))
            if not m:
                continue
            try:
                age = float(r[ca])
            except (TypeError, ValueError):
                continue
            out[f"{prefix}_{int(m.group()):03d}"] = {
                "age": age, "sex": str(r[cx]) if cx is not None else None,
                "diagnosis": str(r[cd]) if cd is not None else None}
        if out:
            return out
    return {}


def update_participant_metadata(cache_dir, rows, note_key, note):
    pm_path = os.path.join(cache_dir, "participant_metadata.json")
    old = {"participants": {}, "notes": {}}
    if os.path.exists(pm_path):
        with open(pm_path) as f:
            old = json.load(f)
    old["participants"].update(rows)
    old["notes"][note_key] = note
    with open(pm_path, "w") as f:
        json.dump(old, f, indent=1)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_isruc_dir(root):
    """[(sid, session, n_sessions, rec, [txt1, txt2])] for ISRUC folder layouts.
    sid = first number in the path below ``root``; sessions ordered by path."""
    recs = sorted(glob.glob(os.path.join(str(root), "**", "*.rec"), recursive=True))
    by_sid = {}
    for rec in recs:
        rel = os.path.relpath(rec, str(root))
        nums = re.findall(r"\d+", rel)
        if not nums:
            continue
        stem = Path(rec).stem
        d = os.path.dirname(rec)
        txts = [os.path.join(d, f"{stem}_{k}.txt") for k in (1, 2)]
        if not os.path.exists(txts[0]):
            continue
        by_sid.setdefault(int(nums[0]), []).append((rec, [t for t in txts if os.path.exists(t)]))
    out = []
    for sid, items in sorted(by_sid.items()):
        for k, (rec, txts) in enumerate(sorted(items), 1):
            out.append((sid, k, len(items), rec, txts))
    return out


def discover_hmc_dir(root):
    psg = sorted(glob.glob(os.path.join(str(root), "**", "SN*.edf"), recursive=True))
    rows = []
    for p in psg:
        if "sleepscoring" in Path(p).name.lower():
            continue
        sc = re.sub(r"\.edf$", "_sleepscoring.edf", p)
        if os.path.exists(sc):
            rows.append((p, sc))
    return rows


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def save_cache(path, epochs, labels, labels_multi, qc, meta):
    meta = dict(meta)
    meta["cache_version"] = CACHE_VERSION
    arrays = {"epochs": epochs.astype(np.float16), "labels": labels.astype(np.int8),
              "labels_multi": labels_multi.astype(np.int8),
              "epoch_index": np.arange(len(labels), dtype=np.int32)}
    arrays.update({k: np.asarray(v) for k, v in qc.items()})
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    arr = {k: z[k] for k in z.files if k != "meta"}
    if "labels_multi" not in arr:                       # cache v2 compatibility
        l2 = arr.get("labels2", np.full(len(arr["labels"]), -2, np.int8))
        arr["labels_multi"] = np.stack([arr["labels"], l2], 1)
    return arr, meta


def read_dod_scorers(scorers_root):
    """{'dodh'|'dodo': {record: {scorer: [stages]}}} from a local copy of
    dreem-learning-evaluation/scorers."""
    out = {"dodh": {}, "dodo": {}}
    for f in glob.glob(os.path.join(str(scorers_root), "**", "*.json"), recursive=True):
        m = re.search(r"(dod[ho])[/\\]([^/\\]+)[/\\]([^/\\]+)\.json$", f)
        if m:
            ds, scorer, rec = m.groups()
            with open(f) as fh:
                out[ds].setdefault(rec, {})[scorer] = json.load(fh)
    return out


def build_cache(cache_dir, isruc1=None, isruc2=None, isruc3=None, hmc=None, log=print,
                max_recordings=None, dodh=None, dodo=None, dod_scorers=None):
    """Process local folders into ``cache_dir`` (resumable)."""
    os.makedirs(cache_dir, exist_ok=True)
    jobs = []
    for root, cohort in ((isruc1, "ISRUC-I"), (isruc2, "ISRUC-II"), (isruc3, "ISRUC-III")):
        if not root:
            continue
        pre = COHORT_PREFIX[cohort]
        for sid, ses, nses, rec, txts in discover_isruc_dir(root):
            name = f"{pre}_{sid:03d}" + (f"_s{ses}" if nses > 1 else "") + ".npz"
            jobs.append((cohort, name, lambda rec=rec, txts=txts, cohort=cohort, sid=sid,
                         ses=ses, nses=nses: load_isruc(rec, txts, cohort, sid, ses, nses)))
        for x in glob.glob(os.path.join(root, "**", "Details*.xls*"), recursive=True):
            update_participant_metadata(cache_dir, read_isruc_details(x, pre), cohort,
                                        f"AVAILABLE ({Path(x).name})")
    if hmc:
        for p, sc in discover_hmc_dir(hmc):
            key = re.search(r"(SN\d+)", Path(p).name).group(1)
            jobs.append(("HMC", f"HMC_{key}.npz", lambda p=p, sc=sc: load_hmc(p, sc)))
    scor = read_dod_scorers(dod_scorers) if dod_scorers else {"dodh": {}, "dodo": {}}
    for root, cohort, key in ((dodh, "DOD-H", "dodh"), (dodo, "DOD-O", "dodo")):
        if not root:
            continue
        for h5 in sorted(glob.glob(os.path.join(str(root), "**", "*.h5"), recursive=True)):
            rid = Path(h5).stem
            jobs.append((cohort, f"{COHORT_PREFIX[cohort]}_{dod_short_id(rid)}.npz",
                         lambda h5=h5, rid=rid, cohort=cohort, key=key:
                         load_dod(h5, scor[key].get(rid), cohort)))
    if max_recordings is not None:
        jobs = jobs[:max_recordings]
    log(f"[cache] {len(jobs)} recordings discovered")
    rows = []
    for i, (cohort, name, fn) in enumerate(jobs, 1):
        if os.path.exists(os.path.join(cache_dir, name)):
            rows.append({"recording": name, "cohort": cohort, "status": "cached"})
            continue
        try:
            out = fn()
            save_cache(os.path.join(cache_dir, name), *out)
            meta, labels = out[4], out[1]
            rows.append({"recording": name, "cohort": cohort, "status": "ok",
                         "channel": meta["channel_raw"], "n_epochs": meta["n_epochs"]})
            log(f"[cache {i}/{len(jobs)}] {name}: ch={meta['channel_raw']} "
                f"epochs={meta['n_epochs']} N3={int((labels == 3).sum())} raters={meta['n_raters']}")
        except Exception as exc:
            rows.append({"recording": name, "cohort": cohort, "status": f"FAILED: {exc}"})
            log(f"[cache {i}/{len(jobs)}] {name}: FAILED {exc}")
    return rows


def zip_cache(cache_dir, out_zip):
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(glob.glob(os.path.join(cache_dir, "*"))):
            if os.path.isfile(p):
                zf.write(p, arcname=os.path.join("sqn2_cache", os.path.basename(p)))
    return out_zip


# ---------------------------------------------------------------------------
# Google Drive helper (Kaggle with Internet ON)
# ---------------------------------------------------------------------------
def drive_id(link_or_id: str) -> str:
    s = str(link_or_id).strip()
    for pat in (r"/d/([A-Za-z0-9_-]{20,})", r"id=([A-Za-z0-9_-]{20,})",
                r"/folders/([A-Za-z0-9_-]{20,})"):
        m = re.search(pat, s)
        if m:
            return m.group(1)
    return s


# ---------------------------------------------------------------------------
# CLI (local export)
# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Export compact SleepQualityNet v2 cache")
    ap.add_argument("--isruc1", default=None)
    ap.add_argument("--isruc2", default=None)
    ap.add_argument("--isruc3", default=None)
    ap.add_argument("--hmc", default=None)
    ap.add_argument("--cache_dir", default="sqn2_cache")
    ap.add_argument("--out", default="sqn2_cache.zip")
    a = ap.parse_args(argv)
    build_cache(a.cache_dir, a.isruc1, a.isruc2, a.isruc3, a.hmc)
    zip_cache(a.cache_dir, a.out)
    n = len(glob.glob(os.path.join(a.cache_dir, "*.npz")))
    print(f"\nDONE: {n} recordings -> {a.out} ({os.path.getsize(a.out) / 1e9:.2f} GB). "
          f"Upload this zip as a Kaggle Dataset (it unpacks to sqn2_cache/) and set "
          f"PREVIOUS_OUTPUT_DIR = '/kaggle/input/<dataset-name>' in the notebook.")


if __name__ == "__main__":
    main()
