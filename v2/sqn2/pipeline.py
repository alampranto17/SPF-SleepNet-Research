"""Stage runner used by the Kaggle notebook.  Every stage is resumable: finished
runs/files are detected and skipped, so a timed-out session can be restarted."""
from __future__ import annotations

import json
import os
import time

import torch

import numpy as np
import pandas as pd

from . import analysis as A
from . import report as RP
from . import train as T
from . import xai as X

DEFAULT_CFG = dict(
    lr=2e-3, wd=1e-2, warmup_steps=200, chunk_len=64, chunks_per_batch=16,
    max_passes=30, patience=5, mc_passes=10, seed=42, verbose=True, n_folds=5,
    variants=list(T.VARIANTS), run_gbm=True, run_loco=True, n_boot=1000,
    xai_epochs_per_fold=120, time_budget_h=10.0,
)


def load_or_make_folds(corpus, cfg, run_dir, log=print):
    """Folds are FROZEN at the first training session (runs/folds.json) so that runs
    resumed in later sessions always use identical participant splits, even if more
    recordings were cached in the meantime.  Training-cohort participants that are not
    in the frozen folds are reported and left unused."""
    path = os.path.join(run_dir, "folds.json")
    have_runs = any(f.endswith(".npz") for f in os.listdir(run_dir))
    if os.path.exists(path) and have_runs:
        with open(path) as f:
            saved = json.load(f)
        if len(saved) == cfg["n_folds"]:
            recs = set(corpus.rec.index)
            ext = list(corpus.rec[~corpus.rec.cohort.isin(T.TRAIN_COHORTS)].index)
            folds = []
            for fo in saved:
                fo = dict(fo)
                fo["fold"] = int(fo["fold"]) if str(fo["fold"]).isdigit() else fo["fold"]
                for k in ("train_recs", "inner_recs", "test_recs"):
                    missing = [r for r in fo[k] if r not in recs]
                    if missing:
                        raise RuntimeError(f"frozen fold {fo['fold']} needs recordings missing from the "
                                           f"cache: {missing[:5]} - restore the full sqn2-cache first")
                fo["external_recs"] = ext
                folds.append(fo)
            used = {r for fo in folds for k in ("train_recs", "inner_recs", "test_recs") for r in fo[k]}
            unused = corpus.rec[corpus.rec.cohort.isin(T.TRAIN_COHORTS) & ~corpus.rec.index.isin(used)]
            log(f"[train] reusing frozen folds from {path}"
                + (f"; {len(unused)} training-cohort recordings cached later are NOT used: "
                   f"{list(unused.index)[:10]}" if len(unused) else ""))
            return folds
    folds = T.make_folds(corpus, k=cfg["n_folds"], seed=cfg["seed"])
    with open(path, "w") as f:
        json.dump([{k: list(map(str, v)) if isinstance(v, list) else v for k, v in fo.items()}
                   for fo in folds], f, indent=1)
    return folds


def train_all(corpus, cfg, run_dir, log=print):
    os.makedirs(run_dir, exist_ok=True)
    folds = load_or_make_folds(corpus, cfg, run_dir, log)
    tr = T.Trainer(corpus, cfg)
    order = [T.PRIMARY_VARIANT] + [v for v in cfg["variants"] if v != T.PRIMARY_VARIANT]
    t0 = time.time()
    budget = cfg.get("time_budget_h", 1e9) * 3600
    t_start = cfg.get("_t_start", t0)              # notebook start time if provided

    def out_of_time():
        return time.time() - t_start > budget

    for v in order:                          # primary model first: RQ1/2/3/5/6/7 need it
        for fo in folds:
            if out_of_time():
                log(f"[train] TIME BUDGET reached - skipping {v} fold {fo['fold']}; "
                    f"re-run the notebook with the saved runs to resume")
                continue
            try:
                tr.run(v, fo, run_dir, seed=cfg["seed"])
                log(f"[train] {v} fold {fo['fold']} done  (elapsed {(time.time() - t0) / 60:.1f} min)")
            except Exception as exc:          # one failing model must not stop the others
                import traceback
                log(f"[train] {v} fold {fo['fold']} FAILED: {type(exc).__name__}: {exc}")
                log(traceback.format_exc()[-1500:])
                if v == T.PRIMARY_VARIANT:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break                         # skip remaining folds of this variant
    if cfg.get("run_gbm") and not out_of_time():
        for fo in folds:
            try:
                tr.run("feat_gbm", fo, run_dir, seed=cfg["seed"], gbm=True)
            except Exception as exc:
                log(f"[train] feat_gbm fold {fo['fold']} FAILED: {exc}")
    if cfg.get("run_loco"):
        cohorts = sorted(corpus.rec[corpus.rec.cohort.isin(list(T.TRAIN_COHORTS))].cohort.unique())
        for coh in cohorts:
            if out_of_time():
                break
            try:
                tr.run(T.PRIMARY_VARIANT, T.loco_split(corpus, coh, cfg["seed"]), run_dir, seed=cfg["seed"])
            except Exception as exc:
                log(f"[train] LOCO {coh} FAILED: {exc}")
    return folds


def _safe(name, fn, log, default=None):
    """Run one analysis block; on failure log the full traceback and continue."""
    import traceback
    try:
        return fn()
    except Exception as exc:
        log(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
        log(traceback.format_exc()[-2500:])
        with open(os.path.join(_safe.out_dir, "analysis_errors.txt"), "a") as f:
            f.write(f"=== {name}\n{traceback.format_exc()}\n")
        return default


def analyse_all(corpus, cfg, run_dir, out_dir, log=print):
    os.makedirs(out_dir, exist_ok=True)
    _safe.out_dir = out_dir
    A.EXCLUSIONS.clear()
    if os.path.exists(os.path.join(out_dir, "analysis_errors.txt")):
        os.remove(os.path.join(out_dir, "analysis_errors.txt"))
    _safe("inventory", lambda: RP.data_inventory(corpus, out_dir), log)
    _safe("methods", lambda: RP.methods_parameters(cfg, out_dir), log)
    res = {}
    log("[RQ1] detection")
    r1, ir, oof, ext = A.rq1(corpus, run_dir, out_dir, n_boot=cfg["n_boot"])   # required
    res["r1"] = r1
    log("[RQ2/3] factor scores")
    S = A.compute_fold_scores(corpus, oof, run_dir)
    Sx = _safe("external scores", lambda: A.compute_external_scores(corpus, ext), log) \
        if ext is not None and len(ext) else None
    cols = ["idx", "rec", "epoch_idx", "participant", "cohort", "label", "pred", "n3_vote_frac"]

    def _save_scores():
        fs = [S.join(oof[cols]).assign(split="OOF")]
        if Sx is not None:
            fs.append(Sx.join(ext[cols]).assign(split="external"))
        pd.concat(fs, ignore_index=True).to_parquet(os.path.join(out_dir, "factor_scores.parquet"))
    _safe("factor_scores.parquet", _save_scores, log)
    res["r2"] = _safe("RQ2", lambda: A.rq2(oof, S, corpus, out_dir, ext, Sx), log)
    log("[RQ2b] within-N3 grade (Low / Moderate / High)")
    res["r2b"] = _safe("RQ2b", lambda: A.rq2_bands(oof, S, out_dir, ext, Sx, corpus=corpus), log)
    res["r3"] = _safe("RQ3", lambda: A.rq3(oof, S, out_dir, ext, Sx), log)
    log("[RQ4] component analysis")
    res["r4"] = _safe("RQ4", lambda: A.rq4(corpus, run_dir, out_dir), log)
    log("[RQ1b] same-protocol comparison with published models")
    res["r1b"] = _safe("RQ1b", lambda: A.sota_comparison(corpus, run_dir, out_dir, n_boot=cfg["n_boot"]), log)
    log("[RQ6] reliability")
    r6 = _safe("RQ6", lambda: A.rq6(oof, out_dir, ext), log)
    res["r6"] = r6
    log("[RQ7] episodes / overnight profiles")
    res["r7"] = _safe("RQ7", lambda: A.rq7(oof, S, r6[3] if r6 else None, out_dir, ext, Sx,
                                            r6[4] if r6 else None), log)
    log("[RQ5] explainability")
    res["r5"] = _safe("RQ5", lambda: X.rq5(corpus, run_dir, out_dir, oof, S,
                                            n_per_fold=cfg["xai_epochs_per_fold"]), log,
                      default={"observed": "FAILED (see analysis_errors.txt)", "status": "NOT_RUN"})
    summ = _safe("summary", lambda: A.rq_summary(out_dir, res["r1"], res["r2"], res["r3"], res["r4"],
                                                 res["r5"], res["r6"], res["r7"], r2b=res.get("r2b"),
                                                 r1b=res.get("r1b")), log)
    if summ is not None:
        _safe("go/no-go", lambda: RP.go_no_go(summ, out_dir), log)
    ex = _safe("exclusions", lambda: A.write_exclusions(out_dir), log)
    if ex is not None and len(ex):
        log(f"[analysis] excluded from evaluation: "
            f"{ex.drop_duplicates(['rec']).groupby('cohort').size().to_dict()} recordings "
            f"(see evaluation_exclusions.csv)")
    return res, summ
