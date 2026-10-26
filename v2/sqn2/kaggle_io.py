"""Automatic private Kaggle Datasets for the cache and training runs.

Credentials are read from Kaggle *Secrets* (Add-ons -> Secrets) named
KAGGLE_USERNAME and KAGGLE_KEY (values from kaggle.json: kaggle.com ->
Settings -> API -> Create New Token).  They are never written into the notebook.
Every function fails softly: if anything is missing the notebook continues and
simply downloads / keeps outputs locally.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys


def setup_credentials(log=print):
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        try:
            from kaggle_secrets import UserSecretsClient
            s = UserSecretsClient()

            def _get(*names):
                for n in names:
                    try:
                        v = s.get_secret(n)
                    except Exception:
                        v = None
                    if v:
                        return v.strip()
                return None
            user = _get("KAGGLE_USERNAME", "kaggle_username")
            key = _get("KAGGLE_KEY", "kaggle_key")
        except Exception as exc:
            log(f"[kaggle] no API secrets found ({type(exc).__name__}); automatic datasets OFF. "
                f"Add secrets KAGGLE_USERNAME and KAGGLE_KEY to enable.")
            return None
    if not (user and key):
        log("[kaggle] secrets empty; automatic datasets OFF")
        return None
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = user, key
    # the preinstalled CLI (2.0.x) returns 404 for private-dataset downloads -> upgrade
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=False,
                   capture_output=True, timeout=300)
    log(f"[kaggle] API credentials loaded for user '{user}'")
    return user


def _kaggle(*args, timeout=7200):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def dataset_exists(user, slug):
    code, out = _kaggle("datasets", "status", f"{user}/{slug}", timeout=120)
    return code == 0 and "ready" in out.lower()


def _copy_tree(src, dst):
    """Copy every file below ``src`` (symlinks followed) into ``dst``, keeping relative
    paths and skipping files that already exist."""
    n = 0
    os.makedirs(dst, exist_ok=True)
    src = os.path.realpath(src)                       # Kaggle mounts inputs through symlinks
    files = set()
    for root, _, fs in os.walk(src, followlinks=True):
        files.update(os.path.join(root, f) for f in fs)
    files.update(f for f in glob.glob(os.path.join(src, "**", "*"), recursive=True) if os.path.isfile(f))
    for f in sorted(files):
        if os.path.basename(f) == "dataset-metadata.json":
            continue
        rel = os.path.relpath(f, src)
        d = os.path.join(dst, rel)
        if not os.path.exists(d):
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(f, d)
            n += 1
    return n


def _attached_inputs(slug, root="/kaggle/input", depth=5, alt_names=()):
    """Folders named ``slug`` (attached dataset) or one of ``alt_names`` (e.g. the
    ``sqn2_cache`` / ``runs`` folder inside an attached NOTEBOOK OUTPUT) under /kaggle/input."""
    names = {slug, *alt_names}
    out = []
    if not os.path.isdir(root):
        return out
    seen = set()
    for cur, dirs, files in os.walk(root, followlinks=True):
        real = os.path.realpath(cur)
        if real in seen or cur[len(root):].count(os.sep) >= depth:
            dirs[:] = []
            continue
        seen.add(real)
        if os.path.basename(cur) in names and _n_files(cur) > 0:
            out.append(cur)
            dirs[:] = []
    return out


def _n_files(path):
    return sum(len(f) for _, _, f in os.walk(os.path.realpath(path), followlinks=True))


def pull_dataset(user, slug, dest, log=print):
    """Restore a dataset into ``dest``: attached input (any mount path) AND the API
    (newest version; files already present are skipped)."""
    got = False
    for cand in _attached_inputs(slug, alt_names=(os.path.basename(dest.rstrip("/")),)):
        n = _copy_tree(cand, dest)
        for z in glob.glob(os.path.join(dest, "**", "*.zip"), recursive=True):
            try:
                shutil.unpack_archive(z, os.path.splitext(z)[0])
                os.remove(z)
            except Exception:
                pass
        log(f"[kaggle] restored {n} files from attached input {cand}")
        got = True
    if not user:
        if not got:
            log(f"[kaggle] no API credentials - {slug} not restored")
        return got
    tmp = dest.rstrip("/") + "_dl"
    src, how = None, None
    code, out = _kaggle("datasets", "download", "-d", f"{user}/{slug}", "-p", tmp, "--unzip")
    if code == 0 and os.path.isdir(tmp) and os.listdir(tmp):
        src, how = tmp, "kaggle CLI"
    else:
        msg = out.strip().splitlines()[-1][:200] if out.strip() else f"exit {code}"
        log(f"[kaggle] CLI download of {user}/{slug} failed ({msg}); trying kagglehub")
        try:
            try:
                import kagglehub
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"],
                               check=False, capture_output=True, timeout=300)
                import kagglehub
            src = kagglehub.dataset_download(f"{user}/{slug}")
            how = "kagglehub"
            log(f"[kaggle] kagglehub path {src} -> {os.path.realpath(src)} "
                f"({_n_files(src)} files)")
        except Exception as exc:
            log(f"[kaggle] kagglehub download failed too: {str(exc)[:200]}")
    if not src:
        if not got:
            log(f"[kaggle] no existing/accessible dataset {user}/{slug} - starting fresh "
                f"(or attach it: Add Input > Your Datasets > {slug})")
        return got
    n = _copy_tree(src, dest)
    for z in glob.glob(os.path.join(dest, "**", "*.zip"), recursive=True):   # dir-mode zips
        try:
            shutil.unpack_archive(z, os.path.splitext(z)[0])
            os.remove(z)
        except Exception:
            pass
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"[kaggle] restored {n} files from private dataset {user}/{slug} via {how}")
    return True


def push_dataset(src_dir, user, slug, title, message="update", log=print):
    """Create (private) or version the dataset from ``src_dir``."""
    if not user:
        return False
    meta = {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}
    with open(os.path.join(src_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f)
    if dataset_exists(user, slug):
        code, out = _kaggle("datasets", "version", "-p", src_dir, "-m", message, "--dir-mode", "zip")
        action = "new version of"
    else:
        code, out = _kaggle("datasets", "create", "-p", src_dir, "--dir-mode", "zip")
        action = "created private dataset"
    if code == 0:
        log(f"[kaggle] {action} {user}/{slug}")
        return True
    log(f"[kaggle] upload of {user}/{slug} failed (outputs are still in /kaggle/working): {out[-400:]}")
    return False
