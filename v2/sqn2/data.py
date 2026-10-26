"""Data layer for SleepQualityNet v2.

Key differences from the v1 notebook (all motivated by the v1 audit):

1. ISRUC channel selection.  v1 searched only for 'C4-A1'/'F4-A1'/'C3-A2' and
   silently fell back to ``raw.ch_names[0]`` (E1-M2, an EOG channel) for the
   83/100 ISRUC-I recordings that label the derivation 'C4-M1'.  v2 accepts the
   A1/M1 spellings of the same central-mastoid derivation and NEVER falls back
   to a non-EEG channel: a recording without a central EEG derivation is
   skipped and logged.
2. Chronological order is preserved.  v1 deleted artifact / low-quality epochs
   inside ``create_epochs`` without storing their index, so N3 continuity,
   episodes and whole-night profiles could not be computed.  v2 keeps every
   epoch of the analysis window with its index and stores the v1 rejection
   rules as boolean QC flags that downstream code can apply or not.
3. Filtering is applied to the continuous signal (not per 30-s epoch) to avoid
   edge transients, and every recording is resampled to 100 Hz (Sleep-EDF is
   natively 100 Hz; v1 up-sampled it to 200 Hz, which adds no information).
4. A second ISRUC scorer (``*_2.txt``) is loaded when present so model-vs-
   expert agreement can be compared with expert-vs-expert agreement.
5. Sleep-EDF night-2 recordings are cached (flagged non-primary) so night-to-
   night stability of the Overnight N3 EEG Profile can be tested without using
   them for training.

This file is self-contained (numpy / scipy / pandas / mne only) so it can also
be run on a local PC as a command-line exporter:

    python export_compact.py --isruc1 D:/AI_Project/ISRUC-Sleep-I \
        --isruc3 D:/AI_Project/ISRUC-Sleep-III --sleepedf D:/AI_Project/Sleep-EDF \
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
CACHE_VERSION = 2

# Stage codes used everywhere: 0 W, 1 N1, 2 N2, 3 N3, 4 REM, -1 unscored/movement
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

STAGE_MAP_ISRUC = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: -1, 7: -1, 9: -1}
STAGE_MAP_SLEEP_EDF = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
    "Sleep stage ?": -1, "Movement time": -1,
}
SCORED_SLEEP = {"Sleep stage 1", "Sleep stage 2", "Sleep stage 3",
                "Sleep stage 4", "Sleep stage R"}

# Central-mastoid EEG derivations, in priority order.  A1/M1 (and A2/M2) are
# two names for the same left (right) mastoid reference.
ISRUC_EEG_CANDIDATES = ["C4-A1", "C4-M1", "C3-A2", "C3-M2"]
SLEEP_EDF_EEG_CANDIDATES = ["Fpz-Cz", "EEG Fpz-Cz"]


def _norm_ch(name: str) -> str:
    n = str(name).upper().replace("EEG", "").replace(" ", "").strip()
    return n


def pick_channel(ch_names, candidates):
    """Return the first channel in ``ch_names`` matching ``candidates``.

    Matching ignores case, spaces and an 'EEG ' prefix.  Returns None when no
    candidate exists -- callers must skip the recording rather than fall back.
    """
    normed = {_norm_ch(c): c for c in ch_names}
    for cand in candidates:
        key = _norm_ch(cand)
        if key in normed:
            return normed[key]
    return None


def standard_derivation(channel: str) -> str:
    n = _norm_ch(channel)
    return {"C4-A1": "C4-M1", "C4-M1": "C4-M1", "C3-A2": "C3-M2",
            "C3-M2": "C3-M2", "FPZ-CZ": "Fpz-Cz"}.get(n, channel)


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

    Re-implements the v1 ``detect_artifacts`` / ``compute_signal_quality`` /
    clipping rules (same thresholds) but stores them as flags instead of
    deleting epochs.  Band powers use Welch PSDs instead of per-epoch IIR
    filtering (equivalent quantity, much faster, no edge transients).
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
    # v1 rules (thresholds unchanged)
    clip_limit = 100 * fs / 200.0          # v1: >100 clipped samples at 200 Hz
    clip_flag = n_clipped > clip_limit
    artifact_flag = (peak_abs > 300) | (zmax > 5.0) | (p_hf > 0.5 * p_delta)
    q = np.ones(n)
    q -= 0.3 * (noise_ratio > 0.3)
    q -= 0.3 * (snr_db < 3)
    q -= 0.4 * flat
    lowq_flag = q <= 0.6
    return {
        "qc_peak_abs": peak_abs.astype(np.float32),
        "qc_zmax": zmax.astype(np.float32),
        "qc_n_clipped": n_clipped.astype(np.float32),
        "qc_rms": rms.astype(np.float32),
        "qc_noise_ratio": noise_ratio.astype(np.float32),
        "qc_snr_db": snr_db.astype(np.float32),
        "qc_hf_ratio": hf_ratio.astype(np.float32),
        "qc_flat": flat,
        "qc_score_v1": q.astype(np.float32),
        "flag_clip": clip_flag,
        "flag_artifact": artifact_flag,
        "flag_lowq": lowq_flag,
        "flag_any_v1": clip_flag | artifact_flag | lowq_flag,
    }


def process_signal(sig_uv: np.ndarray, fs: float, n_epochs: int):
    """Continuous signal (uV, native fs) -> (epochs at 100 Hz float16, qc dict).

    * QC metrics: computed on a 0.5-45 Hz (or Nyquist-limited) band-passed
      native-rate copy, epoch by epoch.
    * Model signal: 0.3-35 Hz band-pass on the continuous recording, then
      polyphase resampling to 100 Hz.
    """
    sig = np.asarray(sig_uv, dtype=np.float64)
    sig = np.nan_to_num(sig)
    ns_native = int(round(EPOCH_SEC * fs))
    n_avail = len(sig) // ns_native
    n_epochs = int(min(n_epochs, n_avail))
    sig = sig[: n_epochs * ns_native]
    nyq = fs / 2.0
    hi_qc = min(45.0, nyq * 0.95)
    sos_qc = butter(4, [0.5, hi_qc], btype="band", fs=fs, output="sos")
    sig_qc = sosfiltfilt(sos_qc, sig)
    qc = epoch_qc(sig_qc.reshape(n_epochs, ns_native), fs)
    sos_m = butter(4, [0.3, 35.0], btype="band", fs=fs, output="sos")
    sig_m = sosfiltfilt(sos_m, sig)
    if not np.isclose(fs, TARGET_FS):
        frac = Fraction(TARGET_FS / fs).limit_denominator(1000)
        sig_m = resample_poly(sig_m, frac.numerator, frac.denominator)
    sig_m = sig_m[: n_epochs * EPOCH_SAMPLES]
    if len(sig_m) < n_epochs * EPOCH_SAMPLES:  # rounding guard
        sig_m = np.pad(sig_m, (0, n_epochs * EPOCH_SAMPLES - len(sig_m)))
    epochs = sig_m.reshape(n_epochs, EPOCH_SAMPLES).astype(np.float16)
    return epochs, qc, n_epochs


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def _read_edf(path, preload=True):
    import mne
    p = str(path)
    if not p.lower().endswith(".edf"):
        # ISRUC .rec files are EDF; mne needs the extension.
        tmp = os.path.join(tempfile.gettempdir(), "_sqn2_" + Path(p).stem + ".edf")
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


def load_isruc(rec_path, txt1, txt2=None, cohort="ISRUC-I", sid=None):
    raw = _read_edf(rec_path)
    ch = pick_channel(raw.ch_names, ISRUC_EEG_CANDIDATES)
    if ch is None:
        raise RuntimeError(f"no central EEG derivation in {Path(rec_path).name}; "
                           f"channels={raw.ch_names}")
    fs = float(raw.info["sfreq"])
    sig = raw.get_data(picks=[ch], units="uV")[0]
    lab1 = parse_isruc_txt(txt1)
    lab2 = parse_isruc_txt(txt2) if txt2 and os.path.exists(txt2) else None
    n_sig = len(sig) // int(round(EPOCH_SEC * fs))
    n = min(n_sig, len(lab1))
    epochs, qc, n = process_signal(sig, fs, n)
    labels = lab1[:n]
    labels2 = (lab2[:n] if lab2 is not None and len(lab2) >= n else
               np.full(n, -2, np.int8))
    pid_prefix = "ISRUC1" if cohort == "ISRUC-I" else "ISRUC3"
    meta = {
        "cohort": cohort, "participant": f"{pid_prefix}_{int(sid):03d}",
        "recording": f"{pid_prefix}_{int(sid):03d}", "night": 1, "is_primary": True,
        "channel_raw": ch, "derivation": standard_derivation(ch),
        "native_fs": fs, "n_epochs": int(n), "crop_start_s": 0.0,
        "label_epochs_in_file": int(len(lab1)), "signal_epochs": int(n_sig),
        "has_scorer2": lab2 is not None, "source_file": Path(rec_path).name,
    }
    return epochs, labels, labels2, qc, meta


def load_sleep_edf(psg_path, hyp_path, margin_min=30.0, is_primary=True):
    import mne
    raw = _read_edf(psg_path)
    ch = pick_channel(raw.ch_names, SLEEP_EDF_EEG_CANDIDATES)
    if ch is None:
        raise RuntimeError(f"Fpz-Cz not found in {Path(psg_path).name}")
    ann = mne.read_annotations(str(hyp_path))
    fs = float(raw.info["sfreq"])
    total_s = raw.n_times / fs
    sleep_on = [(float(a["onset"]), float(a["onset"]) + float(a["duration"]))
                for a in ann if a["description"] in SCORED_SLEEP]
    if not sleep_on:
        raise RuntimeError("no scored sleep")
    t0 = max(0.0, min(s for s, _ in sleep_on) - margin_min * 60)
    t1 = min(total_s, max(e for _, e in sleep_on) + margin_min * 60)
    # align crop start to the hypnogram's 30-s grid (annotations start at 0)
    t0 = np.floor(t0 / EPOCH_SEC) * EPOCH_SEC
    i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
    sig = raw.get_data(picks=[ch], units="uV", start=i0, stop=i1)[0]
    n = len(sig) // int(round(EPOCH_SEC * fs))
    labels = np.full(n, -1, np.int8)
    for a in ann:
        code = STAGE_MAP_SLEEP_EDF.get(a["description"], -1)
        if code == -1:
            continue
        s = (float(a["onset"]) - t0) / EPOCH_SEC
        e = s + float(a["duration"]) / EPOCH_SEC
        si, ei = int(np.floor(max(s, 0) + 1e-6)), int(np.ceil(min(e, n) - 1e-6))
        if ei > si:
            labels[si:ei] = code
    epochs, qc, n = process_signal(sig, fs, n)
    labels = labels[:n]
    m = re.match(r"^(SC\d{3})([12])", Path(psg_path).name, re.I)
    pkey, night = (m.group(1).upper(), int(m.group(2))) if m else (Path(psg_path).stem, 1)
    meta = {
        "cohort": "Sleep-EDF", "participant": f"SEDF_{pkey}",
        "recording": f"SEDF_{pkey}{night}", "night": night, "is_primary": bool(is_primary),
        "channel_raw": ch, "derivation": "Fpz-Cz", "native_fs": fs,
        "n_epochs": int(n), "crop_start_s": float(t0), "has_scorer2": False,
        "source_file": Path(psg_path).name,
    }
    info = raw.info.get("subject_info") or {}
    if isinstance(info, dict) and info.get("sex") is not None:
        meta["sex_edf_header"] = int(info.get("sex"))
    return epochs, labels, np.full(n, -2, np.int8), qc, meta


# ---------------------------------------------------------------------------
# Participant metadata (age / sex) -- loaded only if a sheet actually exists
# ---------------------------------------------------------------------------
def _find_col(cols, *keys):
    for c in cols:
        lc = str(c).lower()
        if any(k in lc for k in keys):
            return c
    return None


def read_sleep_edf_subjects(root):
    """Parse SC-subjects.xls(x) if present -> {participant: {age, sex}}."""
    import pandas as pd
    hits = [p for p in glob.glob(os.path.join(str(root), "**", "SC-subjects.xls*"),
                                 recursive=True)]
    if not hits:
        return {}, "NOT_AVAILABLE (SC-subjects.xls not found)"
    try:
        df = pd.read_excel(hits[0])
    except Exception as exc:  # xlrd missing etc.
        return {}, f"NOT_VERIFIED (could not read {Path(hits[0]).name}: {exc})"
    cs, ca, cx = (_find_col(df.columns, "subject"), _find_col(df.columns, "age"),
                  _find_col(df.columns, "sex"))
    out = {}
    if cs is None or ca is None:
        return {}, f"NOT_VERIFIED (columns {list(df.columns)})"
    for _, r in df.iterrows():
        try:
            pkey = f"SEDF_SC4{int(r[cs]):02d}"
        except (TypeError, ValueError):
            continue
        out[pkey] = {"age": float(r[ca]),
                     "sex": (str(r[cx]) if cx is not None else None)}
    return out, f"AVAILABLE ({Path(hits[0]).name}; columns {list(df.columns)})"


def read_isruc_subjects(root, prefix):
    """Best-effort parse of an ISRUC subject-details spreadsheet if one exists."""
    import pandas as pd
    hits = [p for p in glob.glob(os.path.join(str(root), "**", "*.xls*"), recursive=True)]
    for h in hits:
        try:
            df = pd.read_excel(h)
        except Exception:
            continue
        cs, ca = _find_col(df.columns, "subject"), _find_col(df.columns, "age")
        cx = _find_col(df.columns, "sex", "gender")
        if cs is None or ca is None:
            continue
        out = {}
        for _, r in df.iterrows():
            try:
                out[f"{prefix}_{int(r[cs]):03d}"] = {
                    "age": float(r[ca]), "sex": str(r[cx]) if cx is not None else None}
            except (TypeError, ValueError):
                continue
        if out:
            return out, f"AVAILABLE ({Path(h).name})"
    return {}, "NOT_AVAILABLE (no ISRUC subject spreadsheet found)"


# ---------------------------------------------------------------------------
# Discovery (directories and zip archives)
# ---------------------------------------------------------------------------
def discover_isruc_dir(root):
    """Yield (sid, rec, txt1, txt2) for nested '<sid>/<sid>.rec' or flat layouts."""
    recs = sorted(glob.glob(os.path.join(str(root), "**", "*.rec"), recursive=True))
    out = []
    for rec in recs:
        stem = Path(rec).stem
        m = re.search(r"(\d+)", stem)
        if not m:
            continue
        sid = int(m.group(1))
        d = os.path.dirname(rec)
        t1 = os.path.join(d, f"{stem}_1.txt")
        t2 = os.path.join(d, f"{stem}_2.txt")
        if not os.path.exists(t1):
            continue
        out.append((sid, rec, t1, t2 if os.path.exists(t2) else None))
    out.sort(key=lambda r: r[0])
    return out


def discover_sleep_edf_dir(root):
    psg = sorted(glob.glob(os.path.join(str(root), "**", "*-PSG.edf"), recursive=True))
    hyp = sorted(glob.glob(os.path.join(str(root), "**", "*-Hypnogram.edf"), recursive=True))
    hyp_by = {}
    for h in hyp:
        m = re.match(r"^(SC\d{4})", Path(h).name, re.I)
        if m:
            hyp_by[m.group(1).upper()] = h
    rows = []
    for p in psg:
        m = re.match(r"^(SC\d{3})([12])", Path(p).name, re.I)
        if not m:
            continue
        key = (m.group(1) + m.group(2)).upper()
        if key in hyp_by:
            rows.append({"participant": m.group(1).upper(), "night": int(m.group(2)),
                         "psg": p, "hyp": hyp_by[key]})
    # primary = night 1, else night 2 (same rule as v1)
    rows.sort(key=lambda r: (r["participant"], r["night"]))
    seen = set()
    for r in rows:
        r["is_primary"] = r["participant"] not in seen
        seen.add(r["participant"])
    return rows


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def save_cache(path, epochs, labels, labels2, qc, meta):
    meta = dict(meta)
    meta["cache_version"] = CACHE_VERSION
    arrays = {"epochs": epochs.astype(np.float16), "labels": labels.astype(np.int8),
              "labels2": labels2.astype(np.int8),
              "epoch_index": np.arange(len(labels), dtype=np.int32)}
    arrays.update({k: np.asarray(v) for k, v in qc.items()})
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    arr = {k: z[k] for k in z.files if k != "meta"}
    return arr, meta


def _cache_name(meta):
    return f"{meta['recording']}.npz"


def build_cache(cache_dir, isruc1=None, isruc3=None, sleepedf=None,
                include_sleepedf_night2=True, max_recordings=None, log=print):
    """Process every recording found under the given roots into ``cache_dir``.

    Existing cache files are skipped (resumable).  Returns a list of log rows.
    """
    os.makedirs(cache_dir, exist_ok=True)
    rows = []

    def _done(name):
        return os.path.exists(os.path.join(cache_dir, name))

    jobs = []
    if isruc1:
        for sid, rec, t1, t2 in discover_isruc_dir(isruc1):
            jobs.append(("ISRUC-I", f"ISRUC1_{sid:03d}.npz",
                         lambda sid=sid, rec=rec, t1=t1, t2=t2:
                         load_isruc(rec, t1, t2, "ISRUC-I", sid)))
    if isruc3:
        for sid, rec, t1, t2 in discover_isruc_dir(isruc3):
            jobs.append(("ISRUC-III", f"ISRUC3_{sid:03d}.npz",
                         lambda sid=sid, rec=rec, t1=t1, t2=t2:
                         load_isruc(rec, t1, t2, "ISRUC-III", sid)))
    if sleepedf:
        for r in discover_sleep_edf_dir(sleepedf):
            if not r["is_primary"] and not include_sleepedf_night2:
                continue
            jobs.append(("Sleep-EDF", f"SEDF_{r['participant']}{r['night']}.npz",
                         lambda r=r: load_sleep_edf(r["psg"], r["hyp"],
                                                    is_primary=r["is_primary"])))
    if max_recordings is not None:
        jobs = jobs[:max_recordings]
    log(f"[cache] {len(jobs)} recordings discovered")
    for i, (cohort, name, fn) in enumerate(jobs, 1):
        if _done(name):
            rows.append({"recording": name, "cohort": cohort, "status": "cached"})
            continue
        try:
            epochs, labels, labels2, qc, meta = fn()
            save_cache(os.path.join(cache_dir, name), epochs, labels, labels2, qc, meta)
            rows.append({"recording": name, "cohort": cohort, "status": "ok",
                         "channel": meta["channel_raw"], "n_epochs": meta["n_epochs"],
                         "n_scored": int((labels >= 0).sum()),
                         "n_N3": int((labels == 3).sum())})
            log(f"[cache {i}/{len(jobs)}] {name}: ch={meta['channel_raw']} "
                f"epochs={meta['n_epochs']} N3={int((labels == 3).sum())}")
        except Exception as exc:
            rows.append({"recording": name, "cohort": cohort, "status": f"FAILED: {exc}"})
            log(f"[cache {i}/{len(jobs)}] {name}: FAILED {exc}")
    # participant metadata sheets (if present)
    meta_rows = {}
    notes = {}
    if sleepedf:
        m, notes["Sleep-EDF"] = read_sleep_edf_subjects(sleepedf)
        meta_rows.update(m)
    if isruc1:
        m, notes["ISRUC-I"] = read_isruc_subjects(isruc1, "ISRUC1")
        meta_rows.update(m)
    if isruc3:
        m, notes["ISRUC-III"] = read_isruc_subjects(isruc3, "ISRUC3")
        meta_rows.update(m)
    pm_path = os.path.join(cache_dir, "participant_metadata.json")
    old = {}
    if os.path.exists(pm_path):
        with open(pm_path) as f:
            old = json.load(f)
    old_rows, old_notes = old.get("participants", {}), old.get("notes", {})
    old_rows.update(meta_rows)
    old_notes.update(notes)
    with open(pm_path, "w") as f:
        json.dump({"participants": old_rows, "notes": old_notes}, f, indent=1)
    with open(os.path.join(cache_dir, "cache_log.json"), "a") as f:
        f.write(json.dumps(rows) + "\n")
    return rows


# ---------------------------------------------------------------------------
# Raw zip archives (Kaggle): extract one recording at a time to save disk
# ---------------------------------------------------------------------------
def build_cache_from_zip(zip_path, kind, cache_dir, tmp_root=None, log=print, **kw):
    """Stream-process a raw dataset zip (kind in {'isruc1','isruc3','sleepedf'})."""
    tmp_root = tmp_root or tempfile.mkdtemp(prefix="sqn2_raw_")
    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    if kind in ("isruc1", "isruc3"):
        recs = [n for n in names if n.lower().endswith(".rec")]
        groups = []
        for r in recs:
            base = r[:-4]
            members = [r] + [m for m in (base + "_1.txt", base + "_2.txt") if m in names]
            groups.append(members)
    else:
        psg = [n for n in names if n.endswith("-PSG.edf")]
        hyp = [n for n in names if n.endswith("-Hypnogram.edf")]
        groups = []
        for p in psg:
            key = Path(p).name[:6].upper()
            hs = [h for h in hyp if Path(h).name[:6].upper() == key]
            if hs:
                groups.append([p, hs[0]])
        xls = [n for n in names if Path(n).name.lower().startswith("sc-subjects")]
        if xls:
            zf.extract(xls[0], tmp_root)
    xls_any = [n for n in names if n.lower().endswith((".xls", ".xlsx"))]
    for x in xls_any:
        zf.extract(x, tmp_root)
    log(f"[zip] {Path(zip_path).name}: {len(groups)} recordings")
    rows = []
    for g in groups:
        for m in g:
            zf.extract(m, tmp_root)
        sub = os.path.dirname(os.path.join(tmp_root, g[0]))
        args = {"isruc1": dict(isruc1=sub), "isruc3": dict(isruc3=sub),
                "sleepedf": dict(sleepedf=sub)}[kind]
        if kind == "sleepedf":
            # night-1/night-2 primacy needs the full listing; decide from names
            pkey = Path(g[0]).name[:5].upper()
            night = int(Path(g[0]).name[5])
            has_n1 = any(Path(p).name[:6].upper() == f"{pkey}1" for p in
                         [n for n in names if n.endswith("-PSG.edf")])
            is_primary = (night == 1) or not has_n1
            name = f"SEDF_{pkey}{night}.npz"
            if not os.path.exists(os.path.join(cache_dir, name)):
                try:
                    out = load_sleep_edf(os.path.join(tmp_root, g[0]),
                                         os.path.join(tmp_root, g[1]),
                                         is_primary=is_primary)
                    save_cache(os.path.join(cache_dir, name), *out)
                    log(f"[zip] {name} ok")
                except Exception as exc:
                    log(f"[zip] {name} FAILED {exc}")
        else:
            rows += build_cache(cache_dir, log=log, **args, **kw)
        for m in g:
            try:
                os.remove(os.path.join(tmp_root, m))
            except OSError:
                pass
    # metadata sheets
    if kind == "sleepedf":
        build_cache(cache_dir, sleepedf=tmp_root, log=lambda *a: None, max_recordings=0)
    elif kind == "isruc1":
        build_cache(cache_dir, isruc1=tmp_root, log=lambda *a: None, max_recordings=0)
    else:
        build_cache(cache_dir, isruc3=tmp_root, log=lambda *a: None, max_recordings=0)
    shutil.rmtree(tmp_root, ignore_errors=True)
    return rows


def zip_cache(cache_dir, out_zip):
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(glob.glob(os.path.join(cache_dir, "*"))):
            zf.write(p, arcname=os.path.join("sqn2_cache", os.path.basename(p)))
    return out_zip


# ---------------------------------------------------------------------------
# Google Drive helpers (Kaggle with Internet ON)
# ---------------------------------------------------------------------------
def drive_id(link_or_id: str) -> str:
    s = str(link_or_id).strip()
    for pat in (r"/d/([A-Za-z0-9_-]{20,})", r"id=([A-Za-z0-9_-]{20,})",
                r"/folders/([A-Za-z0-9_-]{20,})"):
        m = re.search(pat, s)
        if m:
            return m.group(1)
    return s


def drive_download(link_or_id, out_dir, is_folder=False):
    import gdown
    os.makedirs(out_dir, exist_ok=True)
    fid = drive_id(link_or_id)
    if is_folder or "/folders/" in str(link_or_id):
        return gdown.download_folder(id=fid, output=out_dir, quiet=False,
                                     remaining_ok=True)
    return gdown.download(id=fid, output=out_dir.rstrip("/") + "/", quiet=False)


# ---------------------------------------------------------------------------
# CLI (local export)
# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Export compact SleepQualityNet v2 cache")
    ap.add_argument("--isruc1", default=None)
    ap.add_argument("--isruc3", default=None)
    ap.add_argument("--sleepedf", default=None)
    ap.add_argument("--cache_dir", default="sqn2_cache")
    ap.add_argument("--out", default="sqn2_cache.zip")
    ap.add_argument("--no_night2", action="store_true")
    a = ap.parse_args(argv)
    build_cache(a.cache_dir, a.isruc1, a.isruc3, a.sleepedf,
                include_sleepedf_night2=not a.no_night2)
    zip_cache(a.cache_dir, a.out)
    n = len(glob.glob(os.path.join(a.cache_dir, "*.npz")))
    size = os.path.getsize(a.out) / 1e9
    print(f"\nDONE: {n} recordings -> {a.out} ({size:.2f} GB). "
          f"Upload this zip as a Kaggle Dataset (it unpacks to sqn2_cache/) and set "
          f"PREVIOUS_OUTPUT_DIR = '/kaggle/input/<dataset-name>' in the notebook.")


if __name__ == "__main__":
    main()
