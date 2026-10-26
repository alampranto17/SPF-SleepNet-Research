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
    alive = host_alive(ISRUC_DETAILS)
    for grp, cohort in (("I", "ISRUC-I"), ("II", "ISRUC-II"), ("III", "ISRUC-III")):
        p = os.path.join(cache_dir, f"Details_subgroup_{grp}.xlsx")
        try:
            if not os.path.exists(p):
                if not alive:
                    raise RuntimeError("Details spreadsheet not found locally and the old ISRUC "
                                       "server is retired (age/sex for this subgroup unavailable)")
                _fetch(ISRUC_DETAILS.format(grp=grp), p)
            rows = D.read_isruc_details(p, D.COHORT_PREFIX[cohort])
            D.update_participant_metadata(cache_dir, rows, cohort,
                                          f"AVAILABLE ({len(rows)} participants with age)")
            log(f"[ISRUC details] subgroup {grp}: {len(rows)} participants with age/sex")
        except Exception as exc:
            D.update_participant_metadata(cache_dir, {}, cohort, f"NOT_AVAILABLE ({exc})")
            log(f"[ISRUC details] subgroup {grp}: not available ({exc})")


def host_alive(url, timeout=10):
    import socket
    host = url.split("/")[2].split(":")[0]
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        return True
    except Exception:
        return False
    finally:
        socket.setdefaulttimeout(None)


def cache_isruc_official(cache_dir, subgroup="I", subject_ids=None, workers=3, log=print):
    os.makedirs(cache_dir, exist_ok=True)
    cohort = {"I": "ISRUC-I", "II": "ISRUC-II", "III": "ISRUC-III"}[subgroup]
    pre = D.COHORT_PREFIX[cohort]
    ids = list(subject_ids or {"I": range(1, 101), "II": range(1, 9), "III": range(1, 11)}[subgroup])
    ids = [i for i in ids if not glob.glob(os.path.join(cache_dir, f"{pre}_{i:03d}*.npz"))]
    if not ids:
        log(f"[{cohort}] all requested subjects already cached")
        return
    if not host_alive(ISRUC_URL):
        log(f"[{cohort}] old ISRUC server dataset.isr.uc.pt does not resolve (retired) - skipped")
        return
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


# ---------------------------------------------------------------------------
# ISRUC from a public Google-Drive folder (no 50-item limit)
# ---------------------------------------------------------------------------
ISRUC_WANTED_EXT = (".rec", ".txt", ".rar", ".zip", ".edf", ".xlsx", ".xls")
ISRUC_MEGA = {   # official links, sleeptight.isr.uc.pt (2025-); free MEGA quota ~5 GB / 6 h
    "I": "https://mega.nz/folder/QJgDQDDZ#ZMDj3w82msavACurqP48IA",      # 14.1 GB
    "II": "https://mega.nz/folder/wVgH2ZAJ#TTxlduGNt4TwaR1eNnl7IQ",     # 2.2 GB
    "III": "https://mega.nz/folder/hMIDEBBJ#jzU5WOF7V5fQCgykEkOptA",    # 1.35 GB
}


def _drive_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def _parse_embedded_folder(html):
    """[(name, id, is_folder)] from drive.google.com/embeddedfolderview HTML."""
    import html as _html
    out = []
    for blk in html.split('class="flip-entry"')[1:]:
        mid = re.search(r'id="entry-([\w-]+)"', blk)
        href = re.search(r'href="([^"]+)"', blk)
        title = re.search(r'class="flip-entry-title">([^<]*)<', blk)
        if mid and title:
            h = href.group(1) if href else ""
            out.append((_html.unescape(title.group(1)).strip(), mid.group(1),
                        "/folders/" in h or "folderview" in h))
    return out


def drive_list_folder(folder, log=print, _rel="", _depth=0):
    """Recursive listing of a PUBLIC Drive folder -> [(relpath, file_id)].

    Uses the embeddedfolderview page (lists every item); falls back to gdown's
    parser (max 50 items per folder) if that page cannot be parsed."""
    fid = D.drive_id(folder)
    items = []
    try:
        items = _parse_embedded_folder(_drive_get(f"https://drive.google.com/embeddedfolderview?id={fid}"))
    except Exception as exc:
        log(f"[drive] embeddedfolderview failed for {fid}: {exc}")
    files = []
    if len(items) >= 50:          # embedded view may truncate: merge with gdown's listing
        try:
            g = _gdown()
            seen = {n for n, _, _ in items}
            for f in g.download_folder(**_kw(g.download_folder, id=fid, skip_download=True,
                                             quiet=True, remaining_ok=True)) or []:
                top = f.path.split("/")[0]
                if top not in seen and "/" not in f.path:
                    items.append((top, f.id, False))
                    seen.add(top)
                elif top not in seen:
                    files.append((f"{_rel}/{f.path}" if _rel else f.path, f.id))
        except Exception as exc:
            log(f"[drive] gdown cross-check skipped: {exc}")
    if items:
        for name, iid, is_folder in items:
            rel = f"{_rel}/{name}" if _rel else name
            if is_folder:
                if _depth < 6:
                    files += drive_list_folder(iid, log, rel, _depth + 1)
            else:
                files.append((rel, iid))
        return files
    if _depth > 0:
        return files
    gdown = _gdown()
    log(f"[drive] falling back to gdown {gdown.__version__} folder listing")
    for f in gdown.download_folder(**_kw(gdown.download_folder, id=fid, skip_download=True,
                                         quiet=True, remaining_ok=True)) or []:
        files.append((f.path, f.id))
    return files


def _kw(fn, **kw):
    import inspect
    ok = inspect.signature(fn).parameters
    return {k: v for k, v in kw.items() if k in ok}


def _gdown():
    """gdown >= 6 (no 50-files-per-folder limit) when it can be installed."""
    try:
        import gdown
        if int(gdown.__version__.split(".")[0]) >= 6:
            return gdown
    except Exception:
        pass
    subprocess.run(["pip", "install", "-q", "-U", "gdown>=6"], check=False, timeout=300,
                   capture_output=True)
    import importlib
    import gdown
    return importlib.reload(gdown)


def _gdown_file(gdown, fid, dst):
    try:
        out = gdown.download(**_kw(gdown.download, id=fid, output=dst, quiet=True, retries=2,
                                   timeout=120))
    except Exception:
        return None
    return out if out and os.path.exists(dst) and os.path.getsize(dst) > 0 else None


def _subject_groups(paths):
    """Group relative paths by subject: descend through single-child wrapper
    folders, then each child (folder or '<n>...' file) is one subject."""
    parts = [p.split("/") for p in paths]
    depth = 0
    while True:
        heads = {tuple(q[:depth + 1]) for q in parts if len(q) > depth + 1}
        if len(heads) == 1 and all(len(q) > depth + 1 for q in parts):
            depth += 1
            continue
        break
    groups = {}
    for p, q in zip(paths, parts):
        if len(q) <= depth:
            continue
        m = re.match(r"\D*(\d+)", q[depth])
        if m:
            groups.setdefault(int(m.group(1)), []).append(p)
    return depth, groups


def _cache_subject_dir(sub, cache_dir, cohort, sid, tool, log):
    for arc in glob.glob(os.path.join(sub, "**", "*"), recursive=True):
        low = arc.lower()
        if low.endswith(".rar"):
            _extract_rar(arc, os.path.splitext(arc)[0], tool or _ensure_unrar(log))
            os.remove(arc)
        elif low.endswith(".zip"):
            shutil.unpack_archive(arc, os.path.splitext(arc)[0])
            os.remove(arc)
    found = D.discover_isruc_dir(sub)
    if not found:
        return "FAILED no .rec + _1.txt found"
    msgs = []
    for _, ses, nses, rec, txts in found:
        out = D.load_isruc(rec, txts, cohort, sid, ses, nses)
        D.save_cache(os.path.join(cache_dir, out[4]["recording"] + ".npz"), *out)
        msgs.append(f"s{ses} ch={out[4]['channel_raw']} scorers={len(txts)}")
    return "ok " + ", ".join(msgs)


def _grab_details(paths_local, cache_dir, grp, log):
    for f in paths_local:
        if f.lower().endswith((".xlsx", ".xls")) and "detail" in os.path.basename(f).lower():
            dst = os.path.join(cache_dir, f"Details_subgroup_{grp}.xlsx")
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
                log(f"[ISRUC-{grp}] subject details spreadsheet found: {os.path.basename(f)}")


def cache_isruc_from_drive_folder(folder_link, cache_dir, subgroup="III", subject_ids=None,
                                  workers=3, log=print):
    """Download a public Drive copy of an ISRUC subgroup subject by subject and cache it.
    Accepts per-subject folders (n/n.rec, n/n_1.txt, ...) or per-subject .rar/.zip files."""
    gdown = _gdown()
    cohort = {"I": "ISRUC-I", "II": "ISRUC-II", "III": "ISRUC-III"}[subgroup]
    pre = D.COHORT_PREFIX[cohort]
    os.makedirs(cache_dir, exist_ok=True)
    listing = [(p, i) for p, i in drive_list_folder(folder_link, log)
               if p.lower().endswith(ISRUC_WANTED_EXT)]
    by_path = dict(listing)
    depth, groups = _subject_groups([p for p, _ in listing])
    log(f"[{cohort} Drive] {len(listing)} files listed, {len(groups)} subjects "
        f"(ids {min(groups) if groups else '-'}..{max(groups) if groups else '-'})")
    tmp = tempfile.mkdtemp(prefix=f"sqn2_drive_{pre}_")
    # loose spreadsheets (subject details) at any level outside subject folders
    extra = [p for p in by_path if p.lower().endswith((".xlsx", ".xls"))]
    loc = []
    for p in extra[:5]:
        dst = os.path.join(tmp, os.path.basename(p))
        if _gdown_file(gdown, by_path[p], dst):
            loc.append(dst)
    _grab_details(loc, cache_dir, subgroup, log)
    want = set(subject_ids) if subject_ids else None
    sids = sorted(s for s in groups if (want is None or s in want))
    tool = None
    if any(p.lower().endswith(".rar") for p, _ in listing):
        tool = _ensure_unrar(log)

    def make(sid):
        def job():
            name = f"{pre}_{sid:03d}"
            if glob.glob(os.path.join(cache_dir, f"{name}*.npz")):
                return name, "cached"
            sub = os.path.join(tmp, f"s{sid}")
            try:
                n_bytes = 0
                for p in groups[sid]:
                    if p.lower().endswith((".xlsx", ".xls")):
                        continue
                    rel = "/".join(p.split("/")[depth:])
                    dst = os.path.join(sub, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    ok = None
                    for k in range(3):
                        ok = _gdown_file(gdown, by_path[p], dst)
                        if ok:
                            break
                        time.sleep(10 * (k + 1))
                    if not ok:
                        return name, (f"FAILED Drive download of {rel} (Drive quota 'too many "
                                      f"downloads' or file not public)")
                    n_bytes += os.path.getsize(dst)
                return name, _cache_subject_dir(sub, cache_dir, cohort, sid, tool, log) + \
                    f" [{n_bytes / 1e6:.0f} MB]"
            except Exception as exc:
                return name, f"FAILED {exc}"
            finally:
                shutil.rmtree(sub, ignore_errors=True)
        return job
    if sids:
        _run_jobs([make(s) for s in sids], workers, log, f"{cohort} Drive")
    shutil.rmtree(tmp, ignore_errors=True)
    return len(glob.glob(os.path.join(cache_dir, f"{pre}_*.npz")))


# ---------------------------------------------------------------------------
# ISRUC from the official MEGA folders (megatools static build)
# ---------------------------------------------------------------------------
MEGATOOLS_URL = ("https://xff.cz/megatools/builds/builds/"
                 "megatools-1.11.5.20250706-linux-x86_64.tar.gz")


def _ensure_megatools(log=print):
    if shutil.which("megatools"):
        return shutil.which("megatools")
    d = os.path.join(tempfile.gettempdir(), "megatools")
    exe = glob.glob(os.path.join(d, "**", "megatools"), recursive=True)
    if not exe:
        try:
            os.makedirs(d, exist_ok=True)
            tgz = os.path.join(d, "mt.tar.gz")
            _fetch(MEGATOOLS_URL, tgz, timeout=120)
            shutil.unpack_archive(tgz, d)
            exe = glob.glob(os.path.join(d, "**", "megatools"), recursive=True)
        except Exception as exc:
            log(f"[MEGA] static megatools download failed ({exc}); trying apt")
    if not exe:
        subprocess.run(["apt-get", "install", "-y", "-qq", "megatools"], capture_output=True,
                       timeout=600)
        return shutil.which("megatools")
    os.chmod(exe[0], 0o755)
    return exe[0]


def cache_isruc_from_mega(cache_dir, subgroup="II", log=print, timeout_h=3.0):
    """Whole-folder download with megatools, then cache every subject.
    Free MEGA transfer quota is roughly 5 GB per 6 h per IP: fine for II (2.2 GB) and
    III (1.35 GB); subgroup I (14 GB) may stop part-way - re-run later to continue."""
    cohort = {"I": "ISRUC-I", "II": "ISRUC-II", "III": "ISRUC-III"}[subgroup]
    pre = D.COHORT_PREFIX[cohort]
    os.makedirs(cache_dir, exist_ok=True)
    exe = _ensure_megatools(log)
    if not exe:
        log(f"[{cohort} MEGA] megatools unavailable - skipped")
        return 0
    dl = os.path.join(tempfile.gettempdir(), f"sqn2_mega_{pre}")
    os.makedirs(dl, exist_ok=True)
    log(f"[{cohort} MEGA] downloading the official folder with megatools (progress every 60 s)")
    proc = subprocess.Popen([exe, "dl", "--path", dl, ISRUC_MEGA[subgroup]],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = t_last = time.time()
    tail = []
    for line in proc.stdout:
        line = line.strip()
        if line:
            tail = (tail + [line])[-5:]
        now = time.time()
        if now - t_last > 60:
            t_last = now
            size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(dl, "**", "*"),
                                                               recursive=True) if os.path.isfile(f))
            log(f"    .. {cohort} MEGA: {size / 1e9:.2f} GB on disk, {(now - t0) / 60:.0f} min")
        if now - t0 > timeout_h * 3600:
            proc.kill()
            log(f"[{cohort} MEGA] stopped after {timeout_h} h")
            break
    proc.wait()
    if proc.returncode not in (0, None):
        log(f"[{cohort} MEGA] megatools exit {proc.returncode}: {' | '.join(tail)[-400:]}"
            " (often the MEGA transfer quota - re-run in a few hours; finished subjects are kept)")
    files = [os.path.relpath(f, dl) for f in glob.glob(os.path.join(dl, "**", "*"), recursive=True)
             if os.path.isfile(f)]
    _grab_details([os.path.join(dl, f) for f in files], cache_dir, subgroup, log)
    depth, groups = _subject_groups(files)
    tool = _ensure_unrar(log) if any(f.lower().endswith(".rar") for f in files) else None
    for sid in sorted(groups):
        name = f"{pre}_{sid:03d}"
        if glob.glob(os.path.join(cache_dir, f"{name}*.npz")):
            continue
        sub = os.path.join(dl, "_s", str(sid))
        try:
            for f in groups[sid]:
                dst = os.path.join(sub, "/".join(f.split("/")[depth:]))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(os.path.join(dl, f), dst)
            st = _cache_subject_dir(sub, cache_dir, cohort, sid, tool, log)
        except Exception as exc:
            st = f"FAILED {exc}"
        finally:
            shutil.rmtree(sub, ignore_errors=True)
        log(f"[{cohort} MEGA] {name}: {st}")
    shutil.rmtree(dl, ignore_errors=True)
    return len(glob.glob(os.path.join(cache_dir, f"{pre}_*.npz")))


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
