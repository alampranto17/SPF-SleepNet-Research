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


def setup_credentials(log=print):
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (user and key):
        try:
            from kaggle_secrets import UserSecretsClient
            s = UserSecretsClient()
            user, key = s.get_secret("KAGGLE_USERNAME"), s.get_secret("KAGGLE_KEY")
        except Exception as exc:
            log(f"[kaggle] no API secrets found ({type(exc).__name__}); automatic datasets OFF. "
                f"Add secrets KAGGLE_USERNAME and KAGGLE_KEY to enable.")
            return None
    if not (user and key):
        log("[kaggle] secrets empty; automatic datasets OFF")
        return None
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = user, key
    if shutil.which("kaggle") is None:
        subprocess.run(["pip", "install", "-q", "kaggle"], check=False)
    log(f"[kaggle] API credentials loaded for user '{user}'")
    return user


def _kaggle(*args, timeout=7200):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def dataset_exists(user, slug):
    code, out = _kaggle("datasets", "status", f"{user}/{slug}", timeout=120)
    return code == 0 and "ready" in out.lower()


def _copy_tree(src, dst):
    n = 0
    os.makedirs(dst, exist_ok=True)
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        tgt = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(tgt, exist_ok=True)
        for f in files:
            if f == "dataset-metadata.json":
                continue
            d = os.path.join(tgt, f)
            if not os.path.exists(d):
                shutil.copy2(os.path.join(root, f), d)
                n += 1
    return n


def pull_dataset(user, slug, dest, log=print):
    """Restore a dataset into ``dest``: attached input first, then the API."""
    for cand in (f"/kaggle/input/{slug}", f"/kaggle/input/{slug}/{os.path.basename(dest)}"):
        if os.path.isdir(cand) and os.listdir(cand):
            n = _copy_tree(cand, dest)
            log(f"[kaggle] restored {n} files from attached input {cand}")
            return True
    if not user or not dataset_exists(user, slug):
        log(f"[kaggle] no existing dataset {user}/{slug} - starting fresh")
        return False
    tmp = dest.rstrip("/") + "_dl"
    code, out = _kaggle("datasets", "download", "-d", f"{user}/{slug}", "-p", tmp, "--unzip")
    if code != 0:
        log(f"[kaggle] download of {user}/{slug} failed: {out[-300:]}")
        return False
    for z in glob.glob(os.path.join(tmp, "**", "*.zip"), recursive=True):   # dir-mode zips
        shutil.unpack_archive(z, os.path.splitext(z)[0])
        os.remove(z)
    n = _copy_tree(tmp, dest)
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"[kaggle] restored {n} files from private dataset {user}/{slug}")
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
