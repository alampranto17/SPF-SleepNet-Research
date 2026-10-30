"""Assemble the self-contained Kaggle notebook from the sqn2 package sources."""
import os
import sys

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "sqn2")
MODULES = ["__init__", "data", "download", "features", "scores", "models", "baselines", "train",
           "stats", "analysis", "xai", "xai_ext", "sensitivity", "report", "figures", "pipeline",
           "kaggle_io", "github_io"]

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
MODULE_ROLE = {
    "__init__": "package marker and version",
    "data": "EDF/h5 readers, channel selection and re-referencing, filtering, resampling, DOD scorer alignment, compact cache",
    "download": "HMC, ISRUC (Drive/MEGA), CAP and DOD downloaders with mirrors, resume and quota handling",
    "features": "handcrafted EEG features (band powers, ratios, slow-wave detector)",
    "scores": "rule-derived factor scores, Overall N3 EEG Rating, episodes and overnight profile",
    "models": "SleepQualityNet v2 network (CNN, feature fusion, attention, dilated temporal context)",
    "baselines": "DeepSleepNet, TinySleepNet, AttnSleep, L-SeqSleepNet re-implemented under the same protocol",
    "train": "corpus, participant-grouped folds, trainer, temperature scaling, MC dropout, OOD distance",
    "stats": "bootstrap CIs, paired tests, Holm correction, ICC, agreement statistics",
    "analysis": "pre-specified criteria and RQ1-RQ7 analyses, exclusions",
    "xai": "SHAP surrogate, input-gradient saliency, attention and band-perturbation explanations",
    "xai_ext": "v2.1: explanations of errors and external cohorts, SHAP by outcome, explanation robustness",
    "sensitivity": "v2.1: sensitivity analyses for every RQ and the sensitivity matrix",
    "report": "data inventory, methods parameters, go/no-go table",
    "figures": "all figures",
    "pipeline": "training order, time budget, fold freezing, analysis orchestration",
    "kaggle_io": "save/restore private Kaggle datasets",
    "github_io": "live progress log and result push to GitHub",
}
MODULE_TABLE = ("## 2. Package source (`sqn2`)\n"
                "The cells in this section write the SleepQualityNet v2 package to `./sqn2/`. They are collapsed "
                "(click to expand) and identical to the `v2/sqn2/` folder of the GitHub repository.\n\n"
                "| module | role |\n|---|---|\n" +
                "\n".join(f"| `{m}.py` | {r} |" for m, r in MODULE_ROLE.items()))
HIDDEN = {"jupyter": {"source_hidden": True}, "collapsed": True}
cells = []

cells.append(md("""# SleepQualityNet v2: single-channel N3 detection with a rule-derived N3 EEG factor profile

**Execution notebook (Kaggle).** This notebook reproduces every number, table and figure of the v2 study from public data, end to end: data download → compact cache → participant-grouped training → pre-specified analyses RQ1–RQ7 → figures → archive. A separate, CPU-only notebook (`SleepQualityNet_v2_results.ipynb`) presents the saved results in manuscript order without re-running anything.

### Notebook map
| § | Section | What it does | Typical time (GPU T4 ×2) |
|:-:|---|---|---|
| 0 | Configuration | all user-settable options in one cell | – |
| 1 | Environment and reproducibility | installs packages, records versions, GPU and seeds (`environment.json`) | 1–2 min |
| 2 | Package source | writes the `sqn2` package (hidden cells; identical to the GitHub repository) | seconds |
| 3 | Data acquisition | HMC, ISRUC-I/II/III, CAP, DOD-H/O → compact 100 Hz cache; channel audit | restored from `sqn2-cache` in minutes (first build: hours) |
| 4 | Cohort inventory | corpus, Table-1 style inventory, channel audit | 2–5 min |
| 5 | Model training | 5-fold participant CV: primary model, ablations, published baselines, GBM, leave-one-cohort-out | skipped when runs are restored from `sqn2-runs` |
| 6 | Pre-specified analyses | RQ1–RQ7 + RQ1b (baselines) + RQ2b (within-N3 grade), evaluated mechanically against fixed criteria; v2.1 sensitivity analyses, extended ablations and extended XAI reported separately | 2–2.5 h |
| 7 | Figures | all manuscript and supplementary figures, grouped by research question | 2–5 min |
| 8 | Export | zip archive, GitHub push of tables and figures (no EEG data) | 1–2 min |
| 9 | Reporting rules and availability statements | wording rules, data/code availability, reproducibility checklist | – |

### Study design in one paragraph
Training cohorts **ISRUC-Sleep I** (100 participants) and **HMC** (151 clinical PSGs, AASM). External cohorts, never used for training or model selection: **ISRUC-II** (8 participants × 2 sessions; test–retest), **ISRUC-III** (10 healthy), **Dreem DOD-H** (25 healthy) and **DOD-O** (55 OSA), each DOD night scored by 5 experts, and **CAP** (R&K scoring; used only as the S3/S4 reference for the within-N3 grade). One verified central EEG derivation (C4-M1/A1 or C3-M2/A2), 0.3–35 Hz zero-phase band-pass, 100 Hz. Folds are participant-grouped and frozen at the first training session; early stopping, temperature scaling and the decision threshold use an inner validation split only.

### How to run (Kaggle, free tier)
1. *Settings → Accelerator → **GPU T4 ×2***, *Internet → **On***.
2. *Add input*: your private datasets **`sqn2-cache`** and **`sqn2-runs`** (latest versions). Inputs cannot be attached during a Commit run, so attach them first.
3. *Add-ons → Secrets* (one time): `KAGGLE_USERNAME`, `KAGGLE_KEY` and optionally `github_token` (fine-grained, this repository only, *Contents: read and write*). Never paste a key into a cell.
4. *Save Version → Save & Run All (Commit)*. Finished training runs are skipped automatically; progress is pushed to GitHub every 20 min.

`QUICK_TEST = True` runs a 30-min pipeline check on a small subset. Its numbers are never reportable and are kept in a separate `sqn2-runs-quicktest` dataset.

### Scope of claims
All factor scores, the N3 EEG Rating and the within-N3 grade are **rule-derived research-prototype descriptors of the EEG signal**. They are not clinical or restorative sleep quality, are not validated against expert quality ratings (none exist in these datasets) and are not diagnostic. No synthetic data is used anywhere in this notebook.
"""))

cells.append(md("## 0. Configuration\nEvery option that changes what the notebook does is in this cell. The defaults reproduce the reported study."))

cells.append(code('''# ============================ CONFIGURATION ============================
QUICK_TEST = False          # True = small subset for a fast end-to-end check (results NOT reportable)
TIME_BUDGET_H = 10.0        # stop launching new training runs after this many hours (Kaggle limit 12 h)

RUN_STAGES = ["data", "train", "analyse"]   # normally leave all three
AUTO_KAGGLE_DATASETS = True    # save/restore private datasets via Kaggle Secrets (see top)
CACHE_DATASET = "sqn2-cache"   # private dataset holding the compact cache (shared by quick/full)
RUNS_DATASET = "sqn2-runs"     # private dataset holding finished training runs
AUTO_GITHUB_PUSH = True        # push result tables/figures to GitHub (needs Kaggle secret github_token)
GITHUB_REPO = "alampranto17/SPF-SleepNet-Research"
GITHUB_BRANCH = "v2-kaggle"
PUSH_QUICK_TEST_RESULTS = False   # quick-test numbers are a pipeline check, not results
GITHUB_SYNC_EVERY_MIN = 20        # live progress (log, status, cache list, results so far) -> GitHub; 0 = off

# ---- data sources (Sleep-EDF is no longer used) ----
USE_HMC_PHYSIONET = True              # training cohort (PhysioNet, CC BY 4.0)
# ISRUC: the old server dataset.isr.uc.pt is retired. Order tried: Google Drive -> official MEGA.
# Drive folders must be shared "Anyone with the link" (same files as the thesis data folders).
ISRUC1_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1KiEl4xk0gRTNG9Oeg-gCfzKGKU0gnjGH"
ISRUC2_DRIVE_FOLDER = None            # not on Drive -> official MEGA folder (2.2 GB)
ISRUC3_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1b4Df79NxyiOiYjG83bk2c9tGKXVDpZ5F"
USE_ISRUC_MEGA = True                 # official MEGA links (free quota ~5 GB / 6 h per IP)
DRIVE_RETRY_ROUNDS = 3                # extra Drive passes after a quota stop (full run only) ...
DRIVE_COOLDOWN_MIN = 20               # ... with this pause before each
ISRUC_KNOWN_UNUSABLE = {}             # (subject 40 is now re-referenced C4 - A1)
ISRUC1_MIN_TO_TRAIN = 90              # start training once >= this many ISRUC-I subjects are cached
                                      # (after all Drive passes of this session); folds then freeze
USE_ISRUC_OFFICIAL = False            # old server, retired (kept only as a last resort)
USE_DOD_ZENODO = True                 # Dreem DOD-H / DOD-O, 5 scorers (external)
USE_CAP_PHYSIONET = True              # CAP Sleep Database: expert R&K S3/S4 -> reference for the Low/Moderate/High N3 grade
RELABEL_DOD = True                    # align DOD scorer hypnograms to the (cropped) Zenodo signals via the h5 hypnogram
DOD_ALLOW_FULL_ZIP_DOWNLOAD = False   # only if range streaming fails AND disk has ~40 GB free
# Optional: raw folders already attached as Kaggle datasets (leave None to download)
RAW_ISRUC1_DIR = None
RAW_ISRUC2_DIR = None
RAW_ISRUC3_DIR = None
RAW_HMC_DIR = None
RAW_DODH_DIR = None
RAW_DODO_DIR = None
RAW_DOD_SCORERS_DIR = None   # local copy of dreem-learning-evaluation/scorers
# Optional manual resume (not needed when AUTO_KAGGLE_DATASETS works)
PREVIOUS_OUTPUT_DIR = None

WORK = "/kaggle/working" if __import__("os").path.isdir("/kaggle/working") else "./work"
CACHE_DIR = f"{WORK}/sqn2_cache"
RUN_DIR = f"{WORK}/runs"
OUT_DIR = f"{WORK}/outputs"
if QUICK_TEST:
    RUNS_DATASET = RUNS_DATASET + "-quicktest"   # never mix quick-test runs with real runs
import time as _time; T_START = _time.time()
RUN_NAME = _time.strftime("%Y%m%d-%H%M") + ("_quicktest" if QUICK_TEST else "")   # GitHub folder v2/results/<RUN_NAME>
print("WORK =", WORK)
'''))

cells.append(md("## 1. Environment and reproducibility\nInstalls the few packages Kaggle lacks and records the exact software and hardware environment in `outputs/environment.json`, which is pushed with the results."))
cells.append(code('''# ============================ SETUP ============================
import subprocess, sys, importlib, os
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
for mod, pkg in [("mne", "mne"), ("gdown", "gdown"), ("shap", "shap"), ("xlrd", "xlrd"),
                 ("openpyxl", "openpyxl"), ("pyarrow", "pyarrow"), ("h5py", "h5py"),
                 ("remotezip", "remotezip")]:
    if importlib.util.find_spec(mod) is None:
        _pip(pkg)
_pip("-U", "gdown>=6", "kaggle")   # gdown 6: no 50-file folder limit; new kaggle CLI: private-dataset download works
import torch, numpy as np, pandas as pd
os.makedirs("sqn2", exist_ok=True)
for d in (CACHE_DIR, RUN_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device", DEVICE,
      "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "NO GPU - enable GPU T4 x2")
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 160)
'''))

cells.append(code('''# ---- record the environment (software versions, hardware, seeds) ----
import platform, json, importlib.metadata as _md
def _ver(pkg):
    try:
        return _md.version(pkg)
    except Exception:
        return None
ENV = {
    "run_name": RUN_NAME, "quick_test": QUICK_TEST, "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": {k: _ver(k) for k in ("torch", "numpy", "scipy", "pandas", "scikit-learn", "lightgbm",
                                      "mne", "shap", "h5py", "pyarrow", "matplotlib", "statsmodels")},
    "cuda": torch.version.cuda if DEVICE == "cuda" else None,
    "cudnn": torch.backends.cudnn.version() if DEVICE == "cuda" else None,
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if DEVICE == "cuda" else [],
    "global_seed": 42, "seed_note": "fold seed = 42*1000 + (sum of ord(fold id) mod 997); folds frozen in runs/folds.json",
}
with open(os.path.join(OUT_DIR, "environment.json"), "w") as f:
    json.dump(ENV, f, indent=2)
print(json.dumps(ENV, indent=2))
'''))

cells.append(md(MODULE_TABLE))
for m in MODULES:
    src = open(os.path.join(PKG, f"{m}.py")).read()
    cells.append(code(f"%%writefile sqn2/{m}.py\n{src}", metadata=dict(HIDDEN)))

cells.append(md("""## 3. Data acquisition and compact cache
Each recording is processed and deleted immediately after caching, so disk use stays small. Finished recordings are skipped on re-runs.
The cache log printed below is the **channel audit**: every recording must show a central EEG derivation (C4-A1/C4-M1 or C3-A2/C3-M2); recordings without one are skipped (never an EOG fallback). DOD records are streamed from the remote zip one at a time (HTTP range requests)."""))

cells.append(code('''import importlib, shutil, glob, json, sys
# Re-running in the same Kaggle session? Drop any previously imported sqn2 modules so the
# freshly written package files above are the ones actually used.
for _m in [m for m in list(sys.modules) if m == "sqn2" or m.startswith("sqn2.")]:
    del sys.modules[_m]
importlib.invalidate_caches()
import sqn2
from sqn2 import data as D, download as DL, kaggle_io as KIO
assert hasattr(DL, "cache_isruc_from_mega"), "stale sqn2 package - restart the kernel (Run > Restart)"
from sqn2 import github_io as GH
GH.start_log(WORK)                                # everything printed is also kept in progress.log
if AUTO_GITHUB_PUSH and GITHUB_SYNC_EVERY_MIN:
    if "SYNC" in globals():
        SYNC.stop(final=False)
    GH.STATUS["stage"] = "A: data download"
    SYNC = GH.ProgressSync(WORK, CACHE_DIR, RUN_DIR, OUT_DIR, RUN_NAME, GITHUB_REPO, GITHUB_BRANCH,
                           every_min=GITHUB_SYNC_EVERY_MIN, t_start=T_START,
                           include_results=PUSH_QUICK_TEST_RESULTS or not QUICK_TEST).start()
KUSER = KIO.setup_credentials() if AUTO_KAGGLE_DATASETS else None
if AUTO_KAGGLE_DATASETS:
    KIO.pull_dataset(KUSER, CACHE_DATASET, CACHE_DIR)
    KIO.pull_dataset(KUSER, RUNS_DATASET, RUN_DIR)
n_cached_before = len(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
if PREVIOUS_OUTPUT_DIR:
    for sub, dst in (("sqn2_cache", CACHE_DIR), ("runs", RUN_DIR)):
        src = os.path.join(PREVIOUS_OUTPUT_DIR, sub)
        if os.path.isdir(src):
            for f in glob.glob(os.path.join(src, "*")):
                if not os.path.exists(os.path.join(dst, os.path.basename(f))):
                    shutil.copy2(f, dst)
            print("resumed", sub, "from", src)
if "data" in RUN_STAGES:
    q = QUICK_TEST
    if any([RAW_ISRUC1_DIR, RAW_ISRUC2_DIR, RAW_ISRUC3_DIR, RAW_HMC_DIR, RAW_DODH_DIR, RAW_DODO_DIR]):
        D.build_cache(CACHE_DIR, RAW_ISRUC1_DIR, RAW_ISRUC2_DIR, RAW_ISRUC3_DIR, RAW_HMC_DIR,
                      dodh=RAW_DODH_DIR, dodo=RAW_DODO_DIR, dod_scorers=RAW_DOD_SCORERS_DIR)
    if USE_HMC_PHYSIONET and not RAW_HMC_DIR:
        try:
            DL.cache_hmc_physionet(CACHE_DIR, max_recordings=6 if q else None)
        except Exception as exc:
            print("HMC download failed:", exc)
    ISRUC_N = {"I": 100, "II": 8, "III": 10}
    ISRUC_Q = {"I": range(1, 9), "II": range(1, 3), "III": range(1, 3)}
    def _n_cached(grp):
        pre = {"I": "ISRUC1", "II": "ISRUC2", "III": "ISRUC3"}[grp]
        return len({os.path.basename(f).split("_")[1] for f in glob.glob(os.path.join(CACHE_DIR, pre + "_*.npz"))})
    SRC = {"I": (RAW_ISRUC1_DIR, ISRUC1_DRIVE_FOLDER), "II": (RAW_ISRUC2_DIR, ISRUC2_DRIVE_FOLDER),
           "III": (RAW_ISRUC3_DIR, ISRUC3_DRIVE_FOLDER)}
    def _need(grp):
        return len(ISRUC_Q[grp]) if q else ISRUC_N[grp]
    # 1) Google Drive for every subgroup that has a Drive copy (I, III)
    for grp, (raw_dir, drive) in SRC.items():
        if raw_dir or not drive or _n_cached(grp) >= _need(grp):
            continue
        try:
            DL.cache_isruc_from_drive_folder(drive, CACHE_DIR, grp, subject_ids=list(ISRUC_Q[grp]) if q else None)
        except Exception as exc:
            print(f"ISRUC-{grp} Drive download failed:", exc)
    # 2) MEGA (free quota ~5 GB / 6 h): subgroups without a Drive copy first (II, 2.2 GB) ...
    if USE_ISRUC_MEGA:
        for grp, (raw_dir, drive) in SRC.items():
            if not raw_dir and not drive and _n_cached(grp) < _need(grp):
                try:
                    DL.cache_isruc_from_mega(CACHE_DIR, grp)
                except Exception as exc:
                    print(f"ISRUC-{grp} MEGA download failed:", exc)
        # ... then MEGA as a fallback. MEGA downloads the whole folder from subject 1 onwards,
        # so for ISRUC-I (14 GB > quota) it is only useful when Drive delivered (almost) nothing.
        MEGA_FALLBACK_BELOW = {"I": 0.10, "III": 0.90}
        for grp, (raw_dir, drive) in SRC.items():
            if not raw_dir and drive and _n_cached(grp) < MEGA_FALLBACK_BELOW.get(grp, 0.9) * _need(grp):
                try:
                    DL.cache_isruc_from_mega(CACHE_DIR, grp)
                except Exception as exc:
                    print(f"ISRUC-{grp} MEGA download failed:", exc)
    for grp, (raw_dir, drive) in SRC.items():
        if USE_ISRUC_OFFICIAL and not raw_dir and _n_cached(grp) < _need(grp):
            try:
                DL.cache_isruc_official(CACHE_DIR, grp, list(ISRUC_Q[grp]) if q else None)
            except Exception as exc:
                print(f"ISRUC-{grp} official download failed:", exc)
        print(f"ISRUC-{grp}: {_n_cached(grp)}/{_need(grp)} subjects cached")
    DL.cache_isruc_details(CACHE_DIR)
    if USE_CAP_PHYSIONET:
        try:
            DL.cache_cap_physionet(CACHE_DIR, max_recordings=3 if q else None)
        except Exception as exc:
            print("CAP download failed:", exc)
    if USE_DOD_ZENODO and not (RAW_DODH_DIR or RAW_DODO_DIR):
        try:
            DL.cache_dod(CACHE_DIR, max_recordings=3 if q else None,
                         allow_full_download=DOD_ALLOW_FULL_ZIP_DOWNLOAD)
        except Exception as exc:
            print("DOD download failed:", exc)
    N_RELABELLED = 0
    if RELABEL_DOD:
        try:
            N_RELABELLED = len(DL.relabel_dod(CACHE_DIR, max_recordings=3 if q else None) or [])
        except Exception as exc:
            print("DOD relabel failed:", exc)
    # further Drive passes: the anonymous Drive quota (~5 GB per machine) recovers with time.
    # Training cannot start before ISRUC-I is complete, so waiting here costs nothing extra.
    for rnd in range(1 + (0 if q else DRIVE_RETRY_ROUNDS)):
        todo = [g for g, (raw_dir, drive) in SRC.items()
                if not raw_dir and drive and _n_cached(g) < _need(g) - ISRUC_KNOWN_UNUSABLE.get(g, 0)]
        if not todo:
            break
        if rnd > 0:
            print(f"Drive quota: waiting {DRIVE_COOLDOWN_MIN} min before Drive pass {rnd + 2} ...")
            _time.sleep(DRIVE_COOLDOWN_MIN * 60)
        for grp in todo:
            print(f"ISRUC-{grp}: Drive pass {rnd + 2} for the {_need(grp) - _n_cached(grp)} missing subjects")
            try:
                DL.cache_isruc_from_drive_folder(SRC[grp][1], CACHE_DIR, grp, subject_ids=list(ISRUC_Q[grp]) if q else None)
            except Exception as exc:
                print(f"ISRUC-{grp} Drive pass failed:", exc)
            print(f"ISRUC-{grp}: {_n_cached(grp)}/{_need(grp)} subjects cached")
files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
by = pd.Series([os.path.basename(f).split("_")[0] for f in files]).value_counts()
print("\\ncached recordings:", by.to_dict(), f"| total {len(files)}",
      f"| cache size {sum(os.path.getsize(f) for f in files)/1e9:.2f} GB",
      f"| elapsed {(_time.time()-T_START)/60:.1f} min")
assert len(files) > 0, "No recordings cached - check Internet ON and the download messages above."
for coh in ("ISRUC1", "HMC"):
    if not any(os.path.basename(f).startswith(coh + "_") for f in files):
        print(f"WARNING: no {coh} recordings cached - training cohort missing")
if AUTO_KAGGLE_DATASETS and KUSER and (len(files) > n_cached_before or globals().get("N_RELABELLED", 0) > 0):
    KIO.push_dataset(CACHE_DIR, KUSER, CACHE_DATASET, "SleepQualityNet v2 compact EEG cache",
                     f"{len(files)} recordings")
'''))

cells.append(md("""## 4. Cohort inventory and channel audit
Builds the training/evaluation corpus from the cache and writes `channel_audit.csv` (EEG derivation and native sampling rate of every recording) and `data_inventory.csv` (participants, recordings, epochs, N3 share and available metadata per cohort; the basis of Table 1)."""))
cells.append(code('''from sqn2 import train as T, report as RP
GH.STATUS["stage"] = "B: corpus and data inventory"
corpus = T.Corpus(CACHE_DIR, device=DEVICE)
audit = corpus.rec[["cohort", "participant", "recording", "night", "is_primary", "channel_raw",
                    "derivation", "native_fs", "n"]].reset_index(drop=True)
audit.to_csv(os.path.join(OUT_DIR, "channel_audit.csv"), index=False)
print(audit.groupby(["cohort", "channel_raw"]).size().rename("recordings").to_string())
summ = corpus.ep.groupby("cohort").agg(participants=("participant", "nunique"), epochs=("rec", "size"),
        scored=("label", lambda s: int((s >= 0).sum())), n3=("label", lambda s: int((s == 3).sum())),
        qc_flagged=("flag_any_v1", "mean"))
summ["n3_pct_of_scored"] = 100 * summ.n3 / summ.scored
display(summ)
display(RP.data_inventory(corpus, OUT_DIR).head(40))
'''))

cells.append(md("""## 5. Model training (5-fold participant-grouped CV, ablations, baselines, GBM, leave-one-cohort-out)
Order: the primary model (`full_v2`) on all folds first, then the ablation variants, the features-only GBM and the LOCO models. Each finished run is saved to `runs/` and skipped on re-runs.

| variant | features | pos-weight | attention | context (±15 epochs) |
|---|:-:|:-:|:-:|:-:|
| cnn | – | – | ✓ | – |
| cnn_feat | ✓ | – | ✓ | – |
| cnn_pw | – | ✓ | ✓ | – |
| cnn_feat_pw (v1 design) | ✓ | ✓ | ✓ | – |
| cnn_feat_pw_noattn | ✓ | ✓ | mean pool | – |
| **full_v2** | ✓ | ✓ | ✓ | ✓ |

**v2.1 ablations (added after run 20261028-0834; reported separately, not part of the pre-specified RQ4 verdict).** Leave-one-component-out from the full model, a smaller context, delta-only features and two seed repeats of the primary model (≈ 1.6 min per fold each on T4 ×2):

| variant | features | pos-weight | attention | context |
|---|:-:|:-:|:-:|:-:|
| full_nofeat | – | ✓ | ✓ | ±15 |
| full_nopw | ✓ | – | ✓ | ±15 |
| full_noattn | ✓ | ✓ | mean pool | ±15 |
| full_ctx_r3 | ✓ | ✓ | ✓ | ±3 |
| full_deltafeat | delta group (10/19) | ✓ | ✓ | ±15 |
| full_v2_seed1 / _seed2 | ✓ | ✓ | ✓ | ±15 (training seed +1 / +2) |

**Same-protocol published baselines** (raw EEG only, identical folds, selection, threshold, loss and class weight): DeepSleepNet (2017), TinySleepNet (2020), AttnSleep (2021), L-SeqSleepNet (2023). Training order: full_v2 → ablations → baselines → GBM → leave-one-cohort-out; anything not finished within the time budget continues on the next commit.

Training cohorts: ISRUC-I + HMC. External (never trained on): ISRUC-II, ISRUC-III, DOD-H, DOD-O; CAP (R&K S3/S4 expert reference for the within-N3 grade only).

**Within-N3 grade (RQ2b).** Every detected N3 epoch gets a Low / Moderate / High *N3 slow-wave grade* from the fraction of the epoch covered by slow waves (≥ 75 µV, 0.5–2 Hz): < 20 % / 20–50 % (R&K stage 3) / > 50 % (R&K stage 4), plus the secondary N3 EEG Rating band (0–33 / 33–67 / 67–100). It is validated against expert R&K S3 vs S4 (CAP), graded multi-expert agreement (DOD, ISRUC), within-night decline and ISRUC-II night-to-night stability.
"""))
cells.append(code('''from sqn2 import pipeline as P
CFG = dict(P.DEFAULT_CFG)
CFG.update(time_budget_h=TIME_BUDGET_H, _t_start=T_START, verbose=False)
if QUICK_TEST:
    CFG.update(n_folds=3, max_passes=3, patience=2, mc_passes=3, n_boot=100,
               variants=["full_v2", "cnn_feat_pw", "tinysleepnet", "lseqsleepnet"], xai_epochs_per_fold=20)
_n_isruc1 = len({os.path.basename(f).split("_")[1] for f in glob.glob(os.path.join(CACHE_DIR, "ISRUC1_*.npz"))})
_n_hmc = len(glob.glob(os.path.join(CACHE_DIR, "HMC_*.npz")))
_n_runs_done = len(glob.glob(os.path.join(RUN_DIR, "*.npz")))
# enough ISRUC-I subjects before the folds are frozen by the first training run
if not QUICK_TEST and "train" in RUN_STAGES and _n_runs_done == 0 and \
        (_n_isruc1 < ISRUC1_MIN_TO_TRAIN or _n_hmc < 150):
    RUN_STAGES = [s for s in RUN_STAGES if s not in ("train", "analyse")]
    print(f"Training cohorts incomplete (ISRUC-I {_n_isruc1}/100, need >= {ISRUC1_MIN_TO_TRAIN}; HMC {_n_hmc}/151): training "
          f"is NOT started so no GPU hours are wasted. The cache is already saved to "
          f"'{CACHE_DATASET}'; run the notebook again (e.g. after the MEGA/Drive quota resets) "
          f"and it continues downloading from where it stopped.")
if "train" in RUN_STAGES:
    GH.STATUS["stage"] = "C: training"
    n_runs_before = len(glob.glob(os.path.join(RUN_DIR, "*.npz")))
    try:
        folds = P.train_all(corpus, CFG, RUN_DIR)
    finally:   # save finished runs even if training stops early
        n_runs = len(glob.glob(os.path.join(RUN_DIR, "*.npz")))
        if AUTO_KAGGLE_DATASETS and KUSER and n_runs > n_runs_before:
            KIO.push_dataset(RUN_DIR, KUSER, RUNS_DATASET, "SleepQualityNet v2 training runs",
                             f"{n_runs} runs")
    print(f"training stage finished, elapsed {(_time.time()-T_START)/3600:.2f} h")
'''))

cells.append(md("""## 6. Pre-specified analyses (RQ1–RQ7)
Every research question has a success criterion fixed before the results were seen (`sqn2/analysis.py::CRITERIA`). The analysis evaluates each criterion mechanically and writes the verdict (SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED / ANSWERED) to `research_question_results.csv`. Recordings or epochs removed from evaluation (non-finite signal, DOD label alignment below 0.6 agreement) are listed with the reason in `evaluation_exclusions.csv`; any analysis that fails is logged in `analysis_errors.txt` instead of stopping the notebook.

| RQ | Question |
|:-:|---|
| RQ1 | Participant-independent binary N3 detection from one EEG channel, internally and in external cohorts |
| RQ1b | Same-protocol comparison with DeepSleepNet, TinySleepNet, AttnSleep and L-SeqSleepNet |
| RQ2 | Physiological validity of the factor scores (early vs late night, N3 vs N2, graded multi-expert agreement) |
| RQ2b | Low / Moderate / High within-N3 slow-wave grade against expert R&K S3 vs S4 (CAP) |
| RQ3 | Stability of the Overall N3 EEG Rating (weights, participants, cohorts, test–retest) |
| RQ4 | Contribution of feature fusion, class weighting, attention and temporal context |
| RQ5 | Agreement of explanation methods with the physiology of N3 |
| RQ6 | Reliability score for rejecting untrustworthy epochs |
| RQ7 | N3 episodes and the Overnight N3 EEG Profile against expert hypnograms |
"""))
cells.append(code('''ANALYSED = False
if "analyse" not in RUN_STAGES:
    GH.STATUS["stage"] = "stopped after data stage (training cohorts incomplete or analysis not requested)"
    print("Analysis skipped: training cohorts are not complete yet (see the message above). "
          "This is not an error - the downloaded data is saved; run the notebook again to continue.")
elif not glob.glob(os.path.join(RUN_DIR, "full_v2__fold*.npz")):
    print("Analysis skipped: no trained full_v2 runs yet.")
else:
    GH.STATUS["stage"] = "D: RQ1-RQ7 analyses"
    res, summary = P.analyse_all(corpus, CFG, RUN_DIR, OUT_DIR)
    ANALYSED = True
    if QUICK_TEST:
        print("QUICK_TEST run: numbers below are a pipeline check only - do not report them.")
    display(summary)
    for f in ("go_no_go_decisions.csv", "rq1_detection_metrics.csv", "rq1_sota_comparison.csv",
              "rq1_interrater_benchmark.csv", "rq2_rk_grade_validation.csv", "rq2_n3_grade_validity.csv",
              "rq4_component_effects.csv", "rq2_factor_validity.csv", "rq6_reliability_components.csv",
              "rq7_profile_agreement.csv", "rq4_extended_component_effects.csv", "rq4_seed_variability.csv",
              "rq5x_band_importance_by_outcome.csv", "rq5x_sensitivity.csv", "sensitivity_matrix.csv"):
        fp = os.path.join(OUT_DIR, f)
        if os.path.exists(fp):
            print(f); display(pd.read_csv(fp))
'''))

cells.append(md("""## 7. Figures
All figures are written to `outputs/figures/` (300 dpi PNG) and displayed below, grouped by research question. A figure whose inputs are missing is skipped with a message; the others are still produced."""))
cells.append(code('''if ANALYSED:
    from sqn2 import figures as FG
    from IPython.display import Image, Markdown
    FIGS = FG.make_all(OUT_DIR)
    GROUPS = [("rq1_", "RQ1 / RQ1b - N3 detection and same-protocol baselines"),
              ("rq2_", "RQ2 - factor-score validity"), ("rq2b_", "RQ2b - within-N3 slow-wave grade"),
              ("rq4_", "RQ4 - component contributions"), ("rq4x_", "RQ4 (v2.1) - leave-one-component-out"),
              ("rq5_", "RQ5 - explanations"), ("rq5x_", "RQ5 (v2.1) - errors, external cohorts, robustness"),
              ("sensitivity_", "Sensitivity analyses (v2.1)"),
              ("rq6_", "RQ6 - reliability"), ("rq7_", "RQ7 - episodes and overnight profiles"),
              ("example_", "Illustrative night")]
    shown = set()
    for key, title in GROUPS:
        group = [p for p in FIGS if os.path.basename(p).startswith("fig_" + key) and p not in shown]
        if group:
            display(Markdown(f"### {title}"))
        for p in group:
            shown.add(p)
            display(Markdown(f"`{os.path.basename(p)}`")); display(Image(p, width=900))
    for p in FIGS:
        if p not in shown:
            display(Markdown(f"`{os.path.basename(p)}`")); display(Image(p, width=900))
'''))

cells.append(md("## 8. Export\nWrites `sqn2_v2_outputs.zip` (tables, figures, out-of-fold predictions, run metadata, frozen folds) to the notebook output and pushes the tables and figures to `v2/results/<RUN_NAME>/` on GitHub. EEG signals and per-epoch predictions are **not** pushed to the public repository (ISRUC licence: research use only)."))
cells.append(code('''# package outputs (tables, figures, OOF predictions, run metadata)
import zipfile
zp = os.path.join(WORK, "sqn2_v2_outputs.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(OUT_DIR):
        for f in fs:
            z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), WORK))
    for f in glob.glob(os.path.join(RUN_DIR, "*.npz")) + [os.path.join(RUN_DIR, "folds.json")]:
        if os.path.exists(f) and os.path.getsize(f) < 50e6:
            z.write(f, os.path.relpath(f, WORK))
print("wrote", zp, f"{os.path.getsize(zp)/1e6:.1f} MB", f"| total elapsed {(_time.time()-T_START)/3600:.2f} h")

from sqn2 import github_io as GH
if AUTO_GITHUB_PUSH and (PUSH_QUICK_TEST_RESULTS or not QUICK_TEST):
    n_runs = len(glob.glob(os.path.join(RUN_DIR, "*.npz")))
    note = (f"{'QUICK TEST - pipeline check only, not reportable. ' if QUICK_TEST else ''}"
            f"Cohorts: {corpus.rec.cohort.value_counts().to_dict()}; training runs available: {n_runs}.")
    GH.push_results(OUT_DIR, GITHUB_REPO, GITHUB_BRANCH, run_name=RUN_NAME, note=note)
GH.STATUS["stage"] = "finished"
if "SYNC" in globals():
    SYNC.stop(final=True)                         # last progress snapshot
'''))

cells.append(md("""## 9. Reporting rules and availability statements

### Reporting rules
* Report `research_question_results.csv` **as it comes out**, including criteria that are not met. Settings are never changed after looking at test results.
* Primary metrics use all scored epochs of primary recordings (QC-flagged epochs included); clean-only metrics are a sensitivity analysis.
* Every exclusion in `evaluation_exclusions.csv` is reported in the Methods with its reason.
* Allowed wording: *N3 EEG Rating, factor-level EEG score, Delta Intensity / Slow-Wave Morphology / Spectral Dominance / N3 Continuity / Signal Reliability Score, Overnight N3 EEG Profile, N3 slow-wave grade, rule-derived research prototype*.
* Not allowed: *clinical or restorative sleep quality, validated N3 quality, diagnostic or medical-grade score, personalised clinical recommendation*.
* Attention and feature fusion are reported as components that did **not** significantly change detection accuracy (RQ4), not as improvements.
* Exact parameter values for the Methods section are in `methods_parameters.csv`; software versions in `environment.json`.

### Data availability (draft statement)
All recordings are public: HMC and the CAP Sleep Database (PhysioNet), ISRUC-Sleep subgroups I–III (research-use licence) and the Dreem Open Datasets DOD-H/DOD-O (Zenodo; scorer hypnograms from the Dreem `dreem-learning-evaluation` repository). The notebook downloads them from these sources; no data are redistributed.

### Code availability (draft statement)
The complete code (`sqn2` package, this notebook and the results notebook) is in the GitHub repository `alampranto17/SPF-SleepNet-Research` (folder `v2/`), together with the result tables and figures of every full run.

### Reproducibility checklist
| item | where |
|---|---|
| participant-grouped, frozen folds | `runs/folds.json` (in the output archive) |
| no test-fold model selection | inner split for early stopping, temperature and threshold (§5) |
| fixed seeds | `environment.json`, `methods_parameters.csv` |
| pre-specified criteria | `sqn2/analysis.py::CRITERIA`, `research_question_results.csv` |
| exclusions with reasons | `evaluation_exclusions.csv` |
| software and hardware | `environment.json` |
| same protocol for every baseline | §5, `sqn2/baselines.py::BASELINE_NOTES` |
"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"},
                  "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                             "isGpuEnabled": True}}
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "SleepQualityNet_v2_kaggle.ipynb")
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
