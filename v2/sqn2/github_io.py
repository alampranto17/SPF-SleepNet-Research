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


def push_results(out_dir, repo="alampranto17/SPF-SleepNet-Research", branch="v2-kaggle",
                 dest_root="v2/results", run_name=None, note="", token=None, log=print,
                 remote_url=None):
    tok = token or get_token(log)
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
        for pat in RESULT_PATTERNS:
            for f in glob.glob(os.path.join(out_dir, pat)):
                if os.path.getsize(f) > MAX_FILE_MB * 1e6:
                    continue
                tgt = os.path.join(dest, os.path.relpath(f, out_dir))
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
        git("-c", "user.name=SQN2 Kaggle runner", "-c", "user.email=noreply@users.noreply.github.com",
            "commit", "-m", f"Add Kaggle results {run_name}")
        git("push", "origin", branch)
        log(f"[github] pushed {n} result files to {repo}@{branch}:{dest_root}/{run_name}")
        return True
    except Exception as exc:
        log(f"[github] push failed (results are still in the Kaggle output): {_scrub(str(exc), tok)}")
        return False
    finally:
        shutil.rmtree(work, ignore_errors=True)
