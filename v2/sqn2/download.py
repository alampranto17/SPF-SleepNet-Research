"""Download raw recordings on Kaggle (Internet ON) and convert them to the
compact per-recording cache one recording at a time; raw files are deleted
right after caching so peak disk use stays small.

Sources
-------
* HMC sleep staging database v1.1   : PhysioNet (open access, CC BY 4.0)
* ISRUC-Sleep subgroups I / II / III: official ISRUC server (.rar per subject)
                                      + per-subgroup 'Details' spreadsheets
* Dreem DOD-H / DOD-O               : Zenodo record 15900394 (MIT); .h5 records are
                                      streamed out of the remote zip with HTTP range
                                      requests (remotezip); scorer hypnograms from
                                      the Dreem-Organization/dreem-learning-evaluation repo
* any Google-Drive folder           : gdown (folders limited to 50 items each)
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile

from . import data as D

# PhysioNet open data is mirrored on AWS S3 (physionet-open). From Kaggle the S3
# mirror is typically 10-50x faster than physionet.org, so it is tried first.
HMC_MIRRORS = ("https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1/",
               "https://physionet.org/files/hmc-sleep-staging/1.1/")
HMC_URL = HMC_MIRRORS[-1]
ISRUC_URL = "http://dataset.isr.uc.pt/ISRUC_Sleep/subgroup{grp}/{sid}.rar"
ISRUC_DETAILS = "http://dataset.isr.uc.pt/ISRUC_Sleep/Details/Details_subgroup_{grp}_Submission.xlsx"
DOD_ZIP = {"DOD-H": "https://zenodo.org/records/15900394/files/dodh.zip?download=1",
           "DOD-O": "https://zenodo.org/records/15900394/files/dodo.zip?download=1"}
DOD_SCORERS_REPO = "https://codeload.github.com/Dreem-Organization/dreem-learning-evaluation/zip/refs/heads/{br}"
DOD_EXCLUDE = {"5d8e93a5-d26a-436d-bc2c-e9527dc92cbf"}   # DOD-O record without consensus


def _fetch(url, out, retries=4, timeout=180, log=None, label=None, progress_every=30):
    """Download ``url`` (or the first working URL of a list/tuple) to ``out``.

    * socket ``timeout`` doubles as stall detection: no bytes for ``timeout`` s
      aborts the attempt and retries (next mirror first).
    * with ``log`` set, prints size / MB downloaded / speed every
      ``progress_every`` seconds, so a slow server never looks frozen.
    """
    urls = [url] if isinstance(url, str) else list(url)
    label = label or os.path.basename(out)
    last_exc = None
    for k in range(retries):
        for u in urls:
            tmp = out + ".part"
            try:
                req = urllib.request.Request(u, headers={"User-Agent": "sqn2/2.0"})
                t0 = t_last = time.time()
                done = 0
                with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                    total = int(r.headers.get("Content-Length") or 0)
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        now = time.time()
                        if log and now - t_last >= progress_every:
                            t_last = now
                            pct = f" ({100 * done / total:.0f}% of {total / 1e6:.0f} MB)" if total else ""
                            log(f"    .. {label}: {done / 1e6:.0f} MB{pct}, "
                                f"{done / 1e6 / max(now - t0, 1e-3):.1f} MB/s")
                if total and done < total:
                    raise IOError(f"incomplete: {done}/{total} bytes")
                os.replace(tmp, out)
                return out
            except Exception as exc:
                last_exc = exc
                if log:
                    host = u.split("/")[2]
                    log(f"    !! {label} from {host}: {type(exc).__name__}: {str(exc)[:120]} - retrying")
                if os.path.exists(tmp):
                    os.remove(tmp)
        if k < retries - 1:
            time.sleep(5 * (k + 1))
    raise RuntimeError(f"download failed {urls[0]}: {last_exc}")


def _run_jobs(jobs, workers, log, label):
    """Run jobs in parallel; print each line as soon as that job finishes."""
    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(j) for j in jobs]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                name, st = fut.result()
            except Exception as exc:                       # never kill the whole stage
                name, st = "?", f"FAILED {exc}"
            el = (time.time() - t0) / 60
            eta = el / i * (len(jobs) - i)
            log(f"[{label} {i}/{len(jobs)}] {name}: {st}   ({el:.1f} min, ETA {eta:.0f} min)")


# ---------------------------------------------------------------------------
# HMC (PhysioNet)
# ---------------------------------------------------------------------------
def hmc_listing(log=print):
    """Recording keys from the RECORDS file (S3 mirror first), else the HTML index."""
    for base in HMC_MIRRORS:
        try:
            with urllib.request.urlopen(base + "RECORDS", timeout=60) as r:
                keys = sorted(set(re.findall(r"(SN\d{3})", r.read().decode("utf-8", "ignore"))))
            if keys:
                return keys
        except Exception as exc:
            log(f"[HMC] RECORDS from {base.split('/')[2]} failed: {exc}")
    with urllib.request.urlopen(HMC_URL + "recordings/", timeout=60) as r:
        html = r.read().decode("utf-8", "ignore")
    return sorted(set(re.findall(r'href="(SN\d{3})\.edf"', html)))


def _hmc(path):
    return [m + path for m in HMC_MIRRORS]


def cache_hmc_physionet(cache_dir, max_recordings=None, workers=3, log=print):
    os.makedirs(cache_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="sqn2_hmc_")
    keys = hmc_listing(log)
    if max_recordings is not None:
        keys = keys[:max_recordings]
    log(f"[HMC] {len(keys)} recordings to cache (~110 MB each; AWS S3 mirror first, "
        f"physionet.org fallback). One line per finished recording, progress every 30 s.")
    try:
        _fetch(_hmc("Subjects_info_aggregated.txt"),
               os.path.join(cache_dir, "HMC_Subjects_info_aggregated.txt"))
    except Exception as exc:
        log(f"[HMC] subject info not downloaded: {exc}")

    def make(key):
        def job():
            name = f"HMC_{key}.npz"
            if os.path.exists(os.path.join(cache_dir, name)):
                return name, "cached"
            p = os.path.join(tmp, f"{key}.edf")
            s = os.path.join(tmp, f"{key}_sleepscoring.edf")
            try:
                _fetch(_hmc(f"recordings/{key}.edf"), p, timeout=90, log=log, label=f"HMC {key}")
                _fetch(_hmc(f"recordings/{key}_sleepscoring.edf"), s, timeout=90)
                out = D.load_hmc(p, s)
                D.save_cache(os.path.join(cache_dir, name), *out)
                return name, f"ok ch={out[4]['channel_raw']} fs={out[4]['native_fs']:.0f}"
            except Exception as exc:
                return name, f"FAILED {exc}"
            finally:
                for f in (p, s):
                    if os.path.exists(f):
                        os.remove(f)
        return job
    _run_jobs([make(k) for k in keys], workers, log, "HMC")
    D.update_participant_metadata(cache_dir, {}, "HMC",
                                  "NOT_AVAILABLE per participant (aggregated demographics only)")
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# ISRUC (.rar) from the official server
# ---------------------------------------------------------------------------
def _ensure_unrar(log=print):
    for tool in ("unrar", "7z", "bsdtar"):
        if shutil.which(tool):
            return tool
    for pkg in ("unrar", "p7zip-full"):
        try:
            subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=300)
            subprocess.run(["apt-get", "install", "-y", "-qq", pkg], check=True,
                           capture_output=True, timeout=300)
        except Exception:
            continue
        for tool in ("unrar", "7z"):
            if shutil.which(tool):
                return tool
    try:
        subprocess.run(["pip", "install", "-q", "unrar-cffi"], check=True, timeout=300)
        return "unrar-cffi"
    except Exception:
        pass
    raise RuntimeError("No RAR extractor available (unrar/7z/bsdtar/unrar-cffi)")


def _extract_rar(rar, out_dir, tool):
    os.makedirs(out_dir, exist_ok=True)
    if tool == "unrar":
        subprocess.run(["unrar", "x", "-o+", "-inul", rar, out_dir + os.sep], check=True)
    elif tool == "7z":
        subprocess.run(["7z", "x", "-y", f"-o{out_dir}", rar], check=True, capture_output=True)
    elif tool == "bsdtar":
        subprocess.run(["bsdtar", "-xf", rar, "-C", out_dir], check=True)
    else:
        from unrar.cffi import rarfile
        rf = rarfile.RarFile(rar)
        for n in rf.namelist():
            dest = os.path.join(out_dir, n)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(rf.read(n))


def cache_isruc_details(cache_dir, log=print):
    for grp, cohort in (("I", "ISRUC-I"), ("II", "ISRUC-II"), ("III", "ISRUC-III")):
        p = os.path.join(cache_dir, f"Details_subgroup_{grp}.xlsx")
        try:
            if not os.path.exists(p):
                _fetch(ISRUC_DETAILS.format(grp=grp), p)
            rows = D.read_isruc_details(p, D.COHORT_PREFIX[cohort])
            D.update_participant_metadata(cache_dir, rows, cohort,
                                          f"AVAILABLE ({len(rows)} participants with age)")
            log(f"[ISRUC details] subgroup {grp}: {len(rows)} participants with age/sex")
        except Exception as exc:
            D.update_participant_metadata(cache_dir, {}, cohort, f"NOT_AVAILABLE ({exc})")
            log(f"[ISRUC details] subgroup {grp}: not available ({exc})")


def cache_isruc_official(cache_dir, subgroup="I", subject_ids=None, workers=3, log=print):
    os.makedirs(cache_dir, exist_ok=True)
    cohort = {"I": "ISRUC-I", "II": "ISRUC-II", "III": "ISRUC-III"}[subgroup]
    pre = D.COHORT_PREFIX[cohort]
    ids = list(subject_ids or {"I": range(1, 101), "II": range(1, 9), "III": range(1, 11)}[subgroup])
    tool = _ensure_unrar(log)
    log(f"[{cohort}] extractor: {tool}")
    tmp = tempfile.mkdtemp(prefix=f"sqn2_{pre}_")

    def make(sid):
        def job():
            if glob.glob(os.path.join(cache_dir, f"{pre}_{sid:03d}*.npz")):
                return f"{pre}_{sid:03d}", "cached"
            rar = os.path.join(tmp, f"{sid}.rar")
            sub = os.path.join(tmp, f"x{sid}")
            try:
                _fetch(ISRUC_URL.format(grp=subgroup, sid=sid), rar, timeout=120, log=log, label=f"{cohort} {sid}")
                _extract_rar(rar, sub, tool)
                os.remove(rar)
                found = D.discover_isruc_dir(sub)
                if not found:
                    return f"{pre}_{sid:03d}", "FAILED no .rec/_1.txt in archive"
                msgs = []
                for _, ses, nses, rec, txts in found:
                    out = D.load_isruc(rec, txts, cohort, sid, ses, nses)
                    D.save_cache(os.path.join(cache_dir, out[4]["recording"] + ".npz"), *out)
                    msgs.append(f"s{ses} ch={out[4]['channel_raw']}")
                return f"{pre}_{sid:03d}", "ok " + ", ".join(msgs)
            except Exception as exc:
                return f"{pre}_{sid:03d}", f"FAILED {exc}"
            finally:
                shutil.rmtree(sub, ignore_errors=True)
                if os.path.exists(rar):
                    os.remove(rar)
        return job
    _run_jobs([make(s) for s in ids], workers, log, cohort)
    shutil.rmtree(tmp, ignore_errors=True)


def cache_isruc_from_drive_folder(folder_link, cache_dir, subgroup="III", log=print):
    import gdown
    tmp = tempfile.mkdtemp(prefix="sqn2_drive_")
    gdown.download_folder(id=D.drive_id(folder_link), output=tmp, quiet=True, remaining_ok=True)
    kw = {"I": "isruc1", "II": "isruc2", "III": "isruc3"}[subgroup]
    rows = D.build_cache(cache_dir, log=log, **{kw: tmp})
    shutil.rmtree(tmp, ignore_errors=True)
    return rows


# ---------------------------------------------------------------------------
# Dreem DOD-H / DOD-O
# ---------------------------------------------------------------------------
def dod_scorers(tmp, log=print):
    """{'dodh'|'dodo': {record_uuid: {scorer: [stages]}}} from the evaluation repo."""
    data = None
    for br in ("master", "main"):
        try:
            with urllib.request.urlopen(DOD_SCORERS_REPO.format(br=br), timeout=120) as r:
                data = r.read()
            break
        except Exception:
            continue
    if data is None:
        raise RuntimeError("could not download dreem-learning-evaluation scorers")
    out = {"dodh": {}, "dodo": {}}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for n in zf.namelist():
            m = re.search(r"scorers/(dod[ho])/([^/]+)/([^/]+)\.json$", n)
            if m:
                ds, scorer, rec = m.groups()
                out[ds].setdefault(rec, {})[scorer] = json.loads(zf.read(n))
    log(f"[DOD] scorer hypnograms: DOD-H {len(out['dodh'])} records, "
        f"DOD-O {len(out['dodo'])} records")
    return out


def _open_remote_zip(url, log):
    try:
        import remotezip  # noqa: F401
    except ImportError:
        subprocess.run(["pip", "install", "-q", "remotezip"], check=False, timeout=300)
    from remotezip import RemoteZip
    return RemoteZip(url)


def cache_dod(cache_dir, which=("DOD-H", "DOD-O"), max_recordings=None, log=print,
              allow_full_download=False):
    os.makedirs(cache_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="sqn2_dod_")
    scorers = dod_scorers(tmp, log)
    for cohort in which:
        key = "dodh" if cohort == "DOD-H" else "dodo"
        pre = D.COHORT_PREFIX[cohort]
        zf, local_zip = None, None
        try:
            zf = _open_remote_zip(DOD_ZIP[cohort], log)
            log(f"[{cohort}] streaming records from Zenodo with HTTP range requests")
        except Exception as exc:
            if not allow_full_download:
                log(f"[{cohort}] range streaming unavailable ({exc}); set "
                    f"allow_full_download=True if the disk has room for the full zip")
                continue
            local_zip = os.path.join(tmp, f"{key}.zip")
            _fetch(DOD_ZIP[cohort], local_zip, timeout=300, log=log, label=f"{cohort} zip", progress_every=60)
            zf = zipfile.ZipFile(local_zip)
        members = [n for n in zf.namelist() if n.endswith(".h5")]
        members = [n for n in members if os.path.splitext(os.path.basename(n))[0] not in DOD_EXCLUDE]
        if max_recordings is not None:
            members = members[:max_recordings]
        log(f"[{cohort}] {len(members)} records")
        for i, m in enumerate(members, 1):
            rid = os.path.splitext(os.path.basename(m))[0]
            name = f"{pre}_{D.dod_short_id(rid)}.npz"
            if os.path.exists(os.path.join(cache_dir, name)):
                log(f"[{cohort} {i}/{len(members)}] {name}: cached")
                continue
            try:
                path = zf.extract(m, tmp)
                out = D.load_dod(path, scorers[key].get(rid), cohort)
                D.save_cache(os.path.join(cache_dir, name), *out)
                log(f"[{cohort} {i}/{len(members)}] {name}: ok ch={out[4]['channel_raw']} "
                    f"fs={out[4]['native_fs']:.0f} raters={out[4]['n_raters']}")
            except Exception as exc:
                log(f"[{cohort} {i}/{len(members)}] {name}: FAILED {exc}")
            finally:
                for f in glob.glob(os.path.join(tmp, "**", "*.h5"), recursive=True):
                    os.remove(f)
        if local_zip and os.path.exists(local_zip):
            os.remove(local_zip)
        D.update_participant_metadata(cache_dir, {}, cohort,
                                      "NOT_VERIFIED per participant (cohort-level demographics in paper)")
    shutil.rmtree(tmp, ignore_errors=True)
