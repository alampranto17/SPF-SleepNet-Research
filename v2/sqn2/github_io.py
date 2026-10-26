"""Push result tables / figures from Kaggle to GitHub.

The token is read from a Kaggle Secret named ``github_token`` (``GITHUB_TOKEN``
also accepted) or the environment variable of the same name.  Use a *fine-grained* personal access token limited
to the one repository with "Contents: Read and write".  The token is never
printed; any error text is scrubbed before it is shown.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
import threading
import time

RESULT_PATTERNS = ("*.csv", "*.json", "figures/*.png")
MAX_FILE_MB = 20


SECRET_NAMES = ("github_token", "GITHUB_TOKEN", "Github_Token", "GitHub_Token")


def get_token(log=print, names=SECRET_NAMES):
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
    except Exception:
        client = None
    if client is not None:
        for n in names:
            try:
                tok = client.get_secret(n)
            except Exception:
                tok = None
            if tok:
                log(f"[github] token loaded from Kaggle secret '{n}'")
                return tok.strip()
    log("[github] no GitHub token secret (github_token / GITHUB_TOKEN) - results stay in the "
        "Kaggle output only")
    return None


def _scrub(text, tok):
    return text.replace(tok, "***") if tok else text


_PUSH_LOCK = threading.Lock()


def push_results(out_dir, repo="alampranto17/SPF-SleepNet-Research", branch="v2-kaggle",
                 dest_root="v2/results", run_name=None, note="", token=None, log=print,
                 remote_url=None, extra_files=(), include_results=True, quiet=False):
    """Copy result files (and ``extra_files`` = [(src_path, rel_dest)]) to
    ``dest_root/run_name`` on ``branch`` and push.  Thread-safe; returns True on success."""
    with _PUSH_LOCK:
        return _push_results(out_dir, repo, branch, dest_root, run_name, note, token, log,
                             remote_url, extra_files, include_results, quiet)


def _push_results(out_dir, repo, branch, dest_root, run_name, note, token, log, remote_url,
                  extra_files, include_results, quiet):
    tok = token or get_token(log if not quiet else (lambda *a: None))
    if not tok and remote_url is None:
        return False
    run_name = run_name or time.strftime("%Y%m%d-%H%M")
    url = remote_url or f"https://x-access-token:{tok}@github.com/{repo}.git"
    work = tempfile.mkdtemp(prefix="sqn2_gh_")
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")

    def git(*args, cwd=work):
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env,
                           timeout=900)
        if r.returncode != 0:
            raise RuntimeError(_scrub((r.stderr or r.stdout)[-500:], tok))
        return r.stdout
    try:
        git("clone", "--depth", "1", "--branch", branch, "--single-branch", url, work, cwd=None)
        dest = os.path.join(work, dest_root, run_name)
        os.makedirs(dest, exist_ok=True)
        n = 0
        pairs = []
        if include_results and out_dir and os.path.isdir(out_dir):
            for pat in RESULT_PATTERNS:
                pairs += [(f, os.path.relpath(f, out_dir)) for f in glob.glob(os.path.join(out_dir, pat))]
        pairs += list(extra_files)
        for f, rel in pairs:
            if not os.path.isfile(f) or os.path.getsize(f) > MAX_FILE_MB * 1e6:
                continue
            tgt = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            shutil.copy2(f, tgt)
            n += 1
        with open(os.path.join(dest, "README.md"), "w") as fh:
            fh.write(f"# SleepQualityNet v2 results — {run_name}\n\n{note}\n\n"
                     "Generated automatically by the Kaggle notebook. Key file: "
                     "`research_question_results.csv` (pre-specified criteria, RQ1–RQ7). "
                     "Large files (OOF predictions, model weights) stay in the Kaggle output "
                     "and the private `sqn2-runs` dataset.\n")
        git("add", "-A")
        if not git("status", "--porcelain").strip():
            if not quiet:
                log("[github] nothing new to push")
            return True
        git("-c", "user.name=SQN2 Kaggle runner", "-c", "user.email=noreply@users.noreply.github.com",
            "commit", "-m", f"Kaggle {'progress' if quiet else 'results'} {run_name}")
        for k in range(3):                       # another push may have landed meanwhile
            try:
                git("push", "origin", branch)
                break
            except RuntimeError:
                if k == 2:
                    raise
                git("pull", "--rebase", "-q", "origin", branch)
        log(f"[github] pushed {n} files to {repo}@{branch}:{dest_root}/{run_name}")
        return True
    except Exception as exc:
        log(f"[github] push failed (results are still in the Kaggle output): {_scrub(str(exc), tok)}")
        return False
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Periodic progress sync (every N minutes, background thread)
# ---------------------------------------------------------------------------
STATUS = {"stage": "starting"}


class Tee:
    """Mirror everything printed to a log file (the notebook output is unchanged)."""

    def __init__(self, stream, path):
        self.stream, self.path = stream, path
        self.fh = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
        self._lock = threading.Lock()

    def write(self, text):
        with self._lock:
            try:
                self.fh.write(text)
            except Exception:
                pass
        return self.stream.write(text)

    def flush(self):
        try:
            self.fh.flush()
        except Exception:
            pass
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_log(work_dir):
    import sys
    path = os.path.join(work_dir, "progress.log")
    if not isinstance(sys.stdout, Tee):
        sys.stdout = Tee(sys.stdout, path)
    return path


def _inventory(cache_dir, path):
    rows = ["file,cohort,size_mb"]
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        b = os.path.basename(f)
        rows.append(f"{b},{b.split('_')[0]},{os.path.getsize(f) / 1e6:.2f}")
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")


def _status_file(work_dir, cache_dir, run_dir, t_start, path):
    import json
    from collections import Counter
    cached = Counter(os.path.basename(f).split("_")[0]
                     for f in glob.glob(os.path.join(cache_dir, "*.npz")))
    st = dict(STATUS)
    st.update(updated_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
              elapsed_min=round((time.time() - t_start) / 60, 1),
              cached_recordings=dict(cached),
              finished_training_runs=len(glob.glob(os.path.join(run_dir, "*.npz"))))
    try:
        du = shutil.disk_usage(work_dir)
        st["disk_free_gb"] = round(du.free / 1e9, 1)
    except Exception:
        pass
    with open(path, "w") as fh:
        json.dump(st, fh, indent=1)


class ProgressSync:
    """Push progress.log, status.json, cache inventory and (optionally) result files to
    GitHub every ``every_min`` minutes, from a daemon thread."""

    def __init__(self, work_dir, cache_dir, run_dir, out_dir, run_name, repo, branch,
                 every_min=20, include_results=True, t_start=None, log=print, remote_url=None):
        self.a = dict(work_dir=work_dir, cache_dir=cache_dir, run_dir=run_dir, out_dir=out_dir)
        self.run_name, self.repo, self.branch = run_name, repo, branch
        self.every = max(1, every_min) * 60
        self.include_results = include_results
        self.t_start = t_start or time.time()
        self.log = log
        self.remote_url = remote_url
        self.token = get_token(log) if remote_url is None else "local-test"
        self._stop = threading.Event()
        self.thread = None
        self.n_ok = 0

    def sync(self):
        w = self.a["work_dir"]
        pd_ = os.path.join(w, "_progress")
        os.makedirs(pd_, exist_ok=True)
        _status_file(w, self.a["cache_dir"], self.a["run_dir"], self.t_start,
                     os.path.join(pd_, "status.json"))
        _inventory(self.a["cache_dir"], os.path.join(pd_, "cache_inventory.csv"))
        logf = os.path.join(w, "progress.log")
        tail = os.path.join(pd_, "progress.log")
        if os.path.exists(logf):                  # keep the pushed log small (last ~3 MB)
            with open(logf, "rb") as fh:
                fh.seek(max(0, os.path.getsize(logf) - 3_000_000))
                data = fh.read()
            with open(tail, "wb") as fh:
                fh.write(data)
        extra = [(os.path.join(pd_, f), f"progress/{f}")
                 for f in ("status.json", "cache_inventory.csv", "progress.log")]
        extra += [(f, f"progress/runs/{os.path.basename(f)}")
                  for f in glob.glob(os.path.join(self.a["run_dir"], "*.json"))]
        ok = push_results(self.a["out_dir"], self.repo, self.branch, run_name=self.run_name,
                          note=f"Live progress (updated every {self.every // 60} min). Stage: "
                               f"{STATUS.get('stage')}.",
                          token=self.token, log=self.log, extra_files=extra,
                          include_results=self.include_results, quiet=True,
                          remote_url=self.remote_url)
        self.n_ok += bool(ok)
        return ok

    def _loop(self):
        while not self._stop.wait(self.every):
            try:
                self.sync()
            except Exception as exc:
                self.log(f"[github] progress sync failed: {_scrub(str(exc), self.token)}")

    def start(self):
        if not self.token:
            self.log("[github] no token - live progress sync OFF")
            return self
        try:
            self.sync()                            # first snapshot right away
        except Exception as exc:
            self.log(f"[github] first progress sync failed: {_scrub(str(exc), self.token)}")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.log(f"[github] live progress sync ON: every {self.every // 60} min -> "
                 f"{self.repo}@{self.branch}:v2/results/{self.run_name}/progress/")
        return self

    def stop(self, final=True):
        self._stop.set()
        if final and self.token:
            try:
                self.sync()
            except Exception:
                pass
