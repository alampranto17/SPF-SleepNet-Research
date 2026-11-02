"""Build SleepQualityNet_v2_posthoc4_kaggle.ipynb: v2.5 peer-review checks M1 and M7.

M7 trains one new variant (full_nodeltafeat) on the frozen folds of 20261030-1528; M1 is
analysis only. Everything else reuses the saved runs.
"""
import glob, os
import nbformat as nbf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [md("""# SleepQualityNet v2.5: peer-review checks M1 and M7 (about 1-3 h, GPU recommended)
Two post hoc checks requested by the simulated peer review. No prespecified verdict changes.

1. **M7 delta-feature control (trains one variant).** `full_nodeltafeat` is the primary model
   with the 9 handcrafted features that are *not* in the delta group (the 10 delta features are
   removed). It uses the frozen folds of `20261030-1528`. Compared with `full_v2` by pooled AUC
   and F1 per cohort, with a participant-level paired bootstrap of the AUC difference.
2. **M1 DOD alignment sensitivity (analysis only).** Re-scores the saved primary-model predictions
   against DOD labels shifted by offsets from -30 to +30 epochs around the label-chosen offset,
   and against the unshifted hypnogram (lag 0). This shows how far the DOD result depends on the
   offset. It is **not** a label-free alignment: that needs the EDF and hypnogram start times,
   which the cache does not store.

**How to run (fresh session: Save Version -> Save & Run All):** Add input `sqn2-cache` and `sqn2-runs`
(latest). Accelerator: GPU. Internet ON. Secret `github_token` ticked. Results go to
`v2/results/20261030-1528_posthoc4/`. Training is resumable: if the time budget ends, re-run the
notebook with the saved runs.""")]
cells.append(code('''import os, sys, subprocess, importlib, time
# Pin the preinstalled scientific stack so that installing packages cannot upgrade numpy/scipy/sklearn
from importlib.metadata import version, PackageNotFoundError
pins = []
for pkg in ("numpy", "scipy", "scikit-learn", "pandas", "numba", "matplotlib", "joblib", "torch", "pyarrow"):
    try:
        pins.append(f"{pkg}=={version(pkg)}")
    except PackageNotFoundError:
        pass
open("/tmp/sqn2_constraints.txt", "w").write("\\n".join(pins))
print("pinned:", pins)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", "/tmp/sqn2_constraints.txt",
                "mne", "h5py"], check=False)
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./work"
CACHE_DIR, RUN_DIR, OUT_DIR = f"{WORK}/sqn2_cache", f"{WORK}/runs", f"{WORK}/outputs_posthoc4"
for d in (CACHE_DIR, RUN_DIR, OUT_DIR, "sqn2"):
    os.makedirs(d, exist_ok=True)
GITHUB_REPO, GITHUB_BRANCH, RUN_NAME = "alampranto17/SPF-SleepNet-Research", "main", "20261030-1528_posthoc4"
T0 = time.time()
'''))
for f in sorted(glob.glob(os.path.join(ROOT, "sqn2", "*.py"))):
    src = open(f).read()
    cells.append(code(f"%%writefile sqn2/{os.path.basename(f)}\n{src}", metadata={"jupyter": {"source_hidden": True}, "collapsed": True}))
cells.append(code('''for m in [m for m in list(sys.modules) if m == "sqn2" or m.startswith("sqn2.")]:
    del sys.modules[m]
importlib.invalidate_caches()
import torch, pandas as pd
from sqn2 import kaggle_io as KIO, train as T, pipeline as P, posthoc4 as P4, github_io as GH
KUSER = KIO.setup_credentials()
KIO.pull_dataset(KUSER, "sqn2-cache", CACHE_DIR)
KIO.pull_dataset(KUSER, "sqn2-runs", RUN_DIR)
corpus = T.Corpus(CACHE_DIR, device="cuda" if torch.cuda.is_available() else "cpu")
'''))
cells.append(md("""## Step 1: train the delta-feature control (M7)
Folds are reloaded from `folds.json` in the saved runs, so the participant splits are identical to `full_v2`.
Runs that already exist are skipped. The primary model is listed first; it is skipped because its
files exist."""))
cells.append(code('''cfg = dict(P.DEFAULT_CFG)
cfg.update(variants=[T.PRIMARY_VARIANT, "full_nodeltafeat"], run_gbm=False, run_loco=False,
           time_budget_h=8.0, _t_start=T0)
P.train_all(corpus, cfg, RUN_DIR)
print("trained variants:", sorted({f.split("__")[0] for f in os.listdir(RUN_DIR) if f.endswith(".npz")}))
'''))
cells.append(md("## Step 2: analysis (M1 and M7)"))
cells.append(code('''P4.run_all(corpus, RUN_DIR, OUT_DIR)
pd.set_option("display.width", 220)
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".csv"):
        print(f); display(pd.read_csv(os.path.join(OUT_DIR, f)).round(4))
'''))
cells.append(code('''if os.path.isdir("/kaggle/working"):     # push only real Kaggle runs, never local tests
    GH.push_results(OUT_DIR, GITHUB_REPO, GITHUB_BRANCH, run_name=RUN_NAME,
                    note="v2.5 peer-review checks: M1 DOD alignment sensitivity, M7 delta-feature control")
'''))
nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "kaggle": {"isInternetEnabled": True}}
out = os.path.join(ROOT, "SleepQualityNet_v2_posthoc4_kaggle.ipynb")
nbf.write(nb, out); print("wrote", out, len(cells), "cells")
