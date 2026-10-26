"""Assemble the self-contained Kaggle notebook from the sqn2 package sources."""
import os
import sys

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "sqn2")
MODULES = ["__init__", "data", "download", "features", "scores", "models", "baselines", "train",
           "stats", "analysis", "xai", "report", "figures", "pipeline", "kaggle_io", "github_io"]

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# SleepQualityNet v2 — N3 detection and rule-derived N3 EEG factor profile (Kaggle)

**What this notebook does (end to end, no manual data upload needed):**
1. Downloads **HMC** (PhysioNet, ~150 clinical PSGs, AASM v2.4), **ISRUC-Sleep subgroups I/II/III** (official server; II = 8 people × 2 sessions) and **Dreem DOD-H / DOD-O** (Zenodo; 80 recordings scored by 5 experts each), and converts each recording to a compact cache (verified central EEG channel only, 100 Hz, *all* epochs kept in chronological order, every expert's hypnogram, v1 QC rules stored as flags).
2. Trains the detection model and all ablation variants with **5-fold participant-grouped CV on ISRUC-I + HMC** (inner-validation early stopping and threshold selection — the outer test fold is never used for selection), leave-one-cohort-out models, and evaluates ISRUC-II, ISRUC-III, DOD-H and DOD-O as **external** cohorts.
3. Computes the **rule-derived factor scores** (delta intensity, slow-wave morphology, spectral dominance, N3 continuity), the **Overall N3 EEG Rating**, a cross-fitted **reliability score**, N3 episodes and the **Overnight N3 EEG Profile**.
4. Answers RQ1–RQ7 against **pre-specified criteria** (`research_question_results.csv`) and writes figures and tables to `/kaggle/working/outputs`.

**How to run on Kaggle (free tier) — fully automatic**
* *Settings → Accelerator → **GPU T4 ×2*** and *Settings → **Internet ON*** (phone-verified account).
* **One-time:** kaggle.com → *Settings → API → Create New Token* downloads `kaggle.json` (`{"username": ..., "key": ...}`). In this notebook: *Add-ons → Secrets → Add* two secrets, **`KAGGLE_USERNAME`** and **`KAGGLE_KEY`**, with those values, and tick them for this notebook. (Never paste the key into a cell.)
* Then just **Save Version → Save & Run All (Commit)**, every time:
  - first run: downloads the data, **automatically saves the cache as your private dataset `sqn2-cache`**, then trains and analyses;
  - every later run: **restores `sqn2-cache` (and finished training runs from `sqn2-runs`) automatically** — no re-download, no re-training of finished runs.
* Optional first check: `QUICK_TEST = True` (≈30 min; uses separate `sqn2-runs-quicktest`, so it never mixes with the real runs).
* **Results to GitHub (optional):** add a secret named **`github_token`** (`GITHUB_TOKEN` also works): a *fine-grained* token for `SPF-SleepNet-Research` only, with *Contents: Read and write*. At the end of each full run the tables, figures and `research_question_results.csv` are pushed to `v2/results/<date>/` on the `v2-kaggle` branch.
* Without the secrets everything still works, but outputs stay only in the notebook output.

**Scope of claims.** All scores are rule-derived research-prototype descriptors of the EEG signal. They are *not* clinical sleep quality, not validated against expert quality ratings (none exist in these datasets) and not diagnostic. Synthetic data is never used: if a dataset fails to download the notebook stops or reports it.
"""))

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
ISRUC_KNOWN_UNUSABLE = {"I": 1}       # ISRUC-I subject 40: no referenced central derivation in the file
USE_ISRUC_OFFICIAL = False            # old server, retired (kept only as a last resort)
USE_DOD_ZENODO = True                 # Dreem DOD-H / DOD-O, 5 scorers (external)
USE_CAP_PHYSIONET = True              # CAP Sleep Database: expert R&K S3/S4 -> reference for the Low/Moderate/High N3 grade
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

cells.append(md("## Package source (written to `./sqn2/`)\nThe code below is the full SleepQualityNet v2 package; it is also kept in the GitHub repository."))
for m in MODULES:
    src = open(os.path.join(PKG, f"{m}.py")).read()
    cells.append(code(f"%%writefile sqn2/{m}.py\n{src}"))

cells.append(md("""## Stage A — data acquisition and compact cache
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
if AUTO_KAGGLE_DATASETS and KUSER and len(files) > n_cached_before:
    KIO.push_dataset(CACHE_DIR, KUSER, CACHE_DATASET, "SleepQualityNet v2 compact EEG cache",
                     f"{len(files)} recordings")
'''))

cells.append(md("## Stage B — corpus, data inventory and channel audit"))
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

cells.append(md("""## Stage C — training (5-fold participant CV, ablations, GBM baseline, leave-one-cohort-out)
Order: the primary model (`full_v2`) on all folds first, then the ablation variants, the features-only GBM and the LOCO models. Each finished run is saved to `runs/` and skipped on re-runs.

| variant | features | pos-weight | attention | context (±15 epochs) |
|---|:-:|:-:|:-:|:-:|
| cnn | – | – | ✓ | – |
| cnn_feat | ✓ | – | ✓ | – |
| cnn_pw | – | ✓ | ✓ | – |
| cnn_feat_pw (v1 design) | ✓ | ✓ | ✓ | – |
| cnn_feat_pw_noattn | ✓ | ✓ | mean pool | – |
| **full_v2** | ✓ | ✓ | ✓ | ✓ |

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
# all usable ISRUC-I subjects (99/100) before the folds are frozen by the first training run
if not QUICK_TEST and "train" in RUN_STAGES and _n_runs_done == 0 and (_n_isruc1 < 99 or _n_hmc < 150):
    RUN_STAGES = [s for s in RUN_STAGES if s not in ("train", "analyse")]
    print(f"Training cohorts incomplete (ISRUC-I {_n_isruc1}/99 usable, HMC {_n_hmc}/151): training "
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

cells.append(md("## Stage D — RQ1–RQ7 analyses against pre-specified criteria"))
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
              "rq7_profile_agreement.csv"):
        fp = os.path.join(OUT_DIR, f)
        if os.path.exists(fp):
            print(f); display(pd.read_csv(fp))
'''))

cells.append(code('''if ANALYSED:
    from sqn2 import figures as FG
    from IPython.display import Image
    for p in FG.make_all(OUT_DIR):
        print(os.path.basename(p)); display(Image(p, width=900))
'''))

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

cells.append(md("""## Reporting rules for the manuscript
* Report `research_question_results.csv` **as it comes out**, including criteria that are not met; do not re-run with changed settings selected on these test results.
* Primary metrics = all scored epochs of primary recordings (QC-flagged epochs included); clean-only metrics are a sensitivity analysis.
* Allowed wording: *N3 EEG Rating, factor-level EEG score, Delta Intensity / Slow-Wave Morphology / Spectral Dominance / N3 Continuity / Signal Reliability Score, Overnight N3 EEG Profile, rule-derived research prototype*.
* Not allowed: *clinical/restorative sleep quality, validated N3 quality, diagnostic or medical-grade score*.
* Exact parameter values for the Methods section are in `methods_parameters.csv`.
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
