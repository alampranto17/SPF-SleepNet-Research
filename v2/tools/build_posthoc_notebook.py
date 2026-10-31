"""Build SleepQualityNet_v2_posthoc_kaggle.ipynb: v2.2 post hoc analyses only (no training)."""
import glob, os
import nbformat as nbf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [md("""# SleepQualityNet v2.2: post hoc analyses (no training, about 20-30 min)
Runs only the analyses requested by the pre-submission review, on the saved runs of the reported run `20261030-1528`:
1. **Within-N3 comparators**: the detector's own N3 logit and calibrated probability, and a signal-only composite, evaluated with the same references as the factor scores (graded expert agreement, night-time decline, CAP R&K S4 vs S3).
2. **Calibration**: expected calibration error with and without positive-class weighting (`full_v2` vs `full_nopw`), and cross-fitted Platt scaling.
3. **ICC confidence intervals** (computed locally as well).

**How to run:** Add input `sqn2-cache` and `sqn2-runs` (latest). Accelerator: none or GPU (either works). Internet ON. Secret `github_token` ticked. Then *Save Version → Save & Run All*. Results are pushed to `v2/results/20261030-1528_posthoc/`.
These analyses are post hoc and never change a prespecified verdict.""")]
cells.append(code('''import os, sys, subprocess, importlib
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mne", "pyarrow", "h5py"], check=False)
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./work"
CACHE_DIR, RUN_DIR, OUT_DIR = f"{WORK}/sqn2_cache", f"{WORK}/runs", f"{WORK}/outputs_posthoc"
for d in (CACHE_DIR, RUN_DIR, OUT_DIR, "sqn2"):
    os.makedirs(d, exist_ok=True)
GITHUB_REPO, GITHUB_BRANCH, RUN_NAME = "alampranto17/SPF-SleepNet-Research", "v2-kaggle", "20261030-1528_posthoc"
'''))
for f in sorted(glob.glob(os.path.join(ROOT, "sqn2", "*.py"))):
    src = open(f).read()
    cells.append(code(f"%%writefile sqn2/{os.path.basename(f)}\n{src}", metadata={"jupyter": {"source_hidden": True}, "collapsed": True}))
cells.append(code('''for m in [m for m in list(sys.modules) if m == "sqn2" or m.startswith("sqn2.")]:
    del sys.modules[m]
importlib.invalidate_caches()
import torch, pandas as pd
from sqn2 import kaggle_io as KIO, train as T, posthoc as PH, github_io as GH
KUSER = KIO.setup_credentials()
KIO.pull_dataset(KUSER, "sqn2-cache", CACHE_DIR)
KIO.pull_dataset(KUSER, "sqn2-runs", RUN_DIR)
corpus = T.Corpus(CACHE_DIR, device="cuda" if torch.cuda.is_available() else "cpu")
PH.run_all(corpus, RUN_DIR, OUT_DIR)
pd.set_option("display.width", 220)
for f in ("posthoc_within_n3_comparators.csv", "posthoc_calibration.csv"):
    print(f); display(pd.read_csv(os.path.join(OUT_DIR, f)).round(4))
'''))
cells.append(code('''if os.path.isdir("/kaggle/working"):     # push only real Kaggle runs, never local tests
    GH.push_results(OUT_DIR, GITHUB_REPO, GITHUB_BRANCH, run_name=RUN_NAME,
                    note="v2.2 post hoc analyses on the saved runs of 20261030-1528 (no training)")
'''))
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "kaggle": {"isInternetEnabled": True}}
out = os.path.join(ROOT, "SleepQualityNet_v2_posthoc_kaggle.ipynb")
nbf.write(nb, out); print("wrote", out, len(cells), "cells")
