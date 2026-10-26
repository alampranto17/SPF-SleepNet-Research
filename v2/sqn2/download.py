"""Download raw recordings on Kaggle (Internet ON) and convert them to the
compact per-recording cache one recording at a time (raw files are deleted
immediately after caching, so peak disk use stays small).

Sources
-------
* Sleep-EDF Expanded, sleep-cassette  : PhysioNet (official, open access)
* ISRUC-Sleep subgroup I / III        : official ISRUC server (.rar per subject)
* any Google-Drive folder/file        : gdown (folders limited to 50 items each)
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request

from . import data as D

PHYSIONET_SC = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
PHYSIONET_SUBJECTS = "https://physionet.org/files/sleep-edfx/1.0.0/SC-subjects.xls"
ISRUC_URL = "http://dataset.isr.uc.pt/ISRUC_Sleep/subgroup{grp}/{sid}.rar"


def _fetch(url, out, retries=4, timeout=120):
    for k in range(retries):
        try:
            tmp = out + ".part"
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            os.replace(tmp, out)
            return out
        except Exception as exc:
            if k == retries - 1:
                raise RuntimeError(f"download failed {url}: {exc}")
            time.sleep(5 * (k + 1))


# ---------------------------------------------------------------------------
# Sleep-EDF from PhysioNet
# ---------------------------------------------------------------------------
def sleep_edf_listing():
    with urllib.request.urlopen(PHYSIONET_SC, timeout=60) as r:
        html = r.read().decode("utf-8", "ignore")
    psg = sorted(set(re.findall(r'href="(SC\d{4}E\d-PSG\.edf)"', html)))
    hyp = sorted(set(re.findall(r'href="(SC\d{4}[A-Z]{2}-Hypnogram\.edf)"', html)))
    hyp_by = {h[:6]: h for h in hyp}
    pairs = []
    for p in psg:
        if p[:6] in hyp_by:
            pairs.append({"participant": p[:5], "night": int(p[5]), "psg": p,
                          "hyp": hyp_by[p[:6]]})
    pairs.sort(key=lambda r: (r["participant"], r["night"]))
    seen = set()
    for r in pairs:
        r["is_primary"] = r["participant"] not in seen
        seen.add(r["participant"])
    return pairs


def cache_sleep_edf_physionet(cache_dir, include_night2=True, max_recordings=None,
                              workers=4, log=print):
    os.makedirs(cache_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="sqn2_sedf_")
    pairs = sleep_edf_listing()
    if not include_night2:
        pairs = [p for p in pairs if p["is_primary"]]
    if max_recordings is not None:
        pairs = pairs[:max_recordings]
    log(f"[Sleep-EDF] {len(pairs)} PSG/hypnogram pairs listed on PhysioNet")
    # participant sheet (age / sex)
    try:
        _fetch(PHYSIONET_SUBJECTS, os.path.join(tmp, "SC-subjects.xls"))
        shutil.copy2(os.path.join(tmp, "SC-subjects.xls"),
                     os.path.join(cache_dir, "SC-subjects.xls"))
    except Exception as exc:
        log(f"[Sleep-EDF] SC-subjects.xls not downloaded: {exc}")

    def job(r):
        name = f"SEDF_{r['participant']}{r['night']}.npz"
        if os.path.exists(os.path.join(cache_dir, name)):
            return name, "cached"
        p = os.path.join(tmp, r["psg"])
        h = os.path.join(tmp, r["hyp"])
        try:
            _fetch(PHYSIONET_SC + r["psg"], p)
            _fetch(PHYSIONET_SC + r["hyp"], h)
            out = D.load_sleep_edf(p, h, is_primary=r["is_primary"])
            D.save_cache(os.path.join(cache_dir, name), *out)
            return name, "ok"
        except Exception as exc:
            return name, f"FAILED {exc}"
        finally:
            for f in (p, h):
                if os.path.exists(f):
                    os.remove(f)

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, (name, st) in enumerate(ex.map(job, pairs), 1):
            log(f"[Sleep-EDF {i}/{len(pairs)}] {name}: {st}")
    D.build_cache(cache_dir, sleepedf=cache_dir, max_recordings=0, log=lambda *a: None)
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# ISRUC (.rar) from the official server
# ---------------------------------------------------------------------------
def _ensure_unrar(log=print):
    for tool in ("unrar", "7z", "bsdtar"):
        if shutil.which(tool):
            return tool
    for cmd in (["apt-get", "install", "-y", "-qq", "unrar"],
                ["apt-get", "install", "-y", "-qq", "p7zip-full"]):
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        except Exception:
            try:
                subprocess.run(["apt-get", "update", "-qq"], capture_output=True, timeout=300)
                subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            except Exception:
                continue
        for tool in ("unrar", "7z"):
            if shutil.which(tool):
                return tool
    try:  # pure-python fallback with bundled libunrar
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
        subprocess.run(["7z", "x", "-y", f"-o{out_dir}", rar], check=True,
                       capture_output=True)
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


def cache_isruc_official(cache_dir, subgroup="I", subject_ids=None, workers=3, log=print):
    os.makedirs(cache_dir, exist_ok=True)
    grp_dir = {"I": "I", "III": "III"}[subgroup]
    cohort = "ISRUC-I" if subgroup == "I" else "ISRUC-III"
    prefix = "ISRUC1" if subgroup == "I" else "ISRUC3"
    ids = subject_ids or (range(1, 101) if subgroup == "I" else range(1, 11))
    tool = _ensure_unrar(log)
    log(f"[{cohort}] extractor: {tool}")
    tmp = tempfile.mkdtemp(prefix=f"sqn2_{prefix}_")

    def job(sid):
        name = f"{prefix}_{sid:03d}.npz"
        if os.path.exists(os.path.join(cache_dir, name)):
            return name, "cached"
        rar = os.path.join(tmp, f"{sid}.rar")
        sub = os.path.join(tmp, f"x{sid}")
        try:
            _fetch(ISRUC_URL.format(grp=grp_dir, sid=sid), rar, timeout=600)
            _extract_rar(rar, sub, tool)
            os.remove(rar)
            found = D.discover_isruc_dir(sub)
            if not found:
                return name, "FAILED no .rec/_1.txt in archive"
            _, rec, t1, t2 = found[0]
            out = D.load_isruc(rec, t1, t2, cohort, sid)
            D.save_cache(os.path.join(cache_dir, name), *out)
            for x in glob.glob(os.path.join(sub, "**", "*.xls*"), recursive=True):
                shutil.copy2(x, os.path.join(cache_dir, f"{prefix}_{sid:03d}_" +
                                             os.path.basename(x)))
            return name, f"ok ch={out[4]['channel_raw']}"
        except Exception as exc:
            return name, f"FAILED {exc}"
        finally:
            shutil.rmtree(sub, ignore_errors=True)
            if os.path.exists(rar):
                os.remove(rar)

    with cf.ThreadPoolExecutor(workers) as ex:
        for i, (name, st) in enumerate(ex.map(job, list(ids)), 1):
            log(f"[{cohort} {i}/{len(list(ids))}] {name}: {st}")
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Google Drive folder (e.g. the user's ISRUC-Sleep-III folder)
# ---------------------------------------------------------------------------
def cache_isruc_from_drive_folder(folder_link, cache_dir, subgroup="III", log=print):
    import gdown
    tmp = tempfile.mkdtemp(prefix="sqn2_drive_")
    gdown.download_folder(id=D.drive_id(folder_link), output=tmp, quiet=True,
                          remaining_ok=True)
    kw = {"isruc1": tmp} if subgroup == "I" else {"isruc3": tmp}
    rows = D.build_cache(cache_dir, log=log, **kw)
    shutil.rmtree(tmp, ignore_errors=True)
    return rows
