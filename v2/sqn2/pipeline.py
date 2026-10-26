"""Stage runner used by the Kaggle notebook.  Every stage is resumable: finished
runs/files are detected and skipped, so a timed-out session can be restarted."""
from __future__ import annotations

import json
import os
import time

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


def train_all(corpus, cfg, run_dir, log=print):
    os.makedirs(run_dir, exist_ok=True)
    folds = T.make_folds(corpus, k=cfg["n_folds"], seed=cfg["seed"])
    with open(os.path.join(run_dir, "folds.json"), "w") as f:
        json.dump([{k: list(map(str, v)) if isinstance(v, list) else v for k, v in fo.items()}
                   for fo in folds], f, indent=1)
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
            tr.run(v, fo, run_dir, seed=cfg["seed"])
            log(f"[train] {v} fold {fo['fold']} done  (elapsed {(time.time() - t0) / 60:.1f} min)")
    if cfg.get("run_gbm") and not out_of_time():
        for fo in folds:
            tr.run("feat_gbm", fo, run_dir, seed=cfg["seed"], gbm=True)
    if cfg.get("run_loco"):
        cohorts = sorted(corpus.rec[corpus.rec.cohort.isin(list(T.TRAIN_COHORTS))].cohort.unique())
        for coh in cohorts:
            if out_of_time():
                break
            tr.run(T.PRIMARY_VARIANT, T.loco_split(corpus, coh, cfg["seed"]), run_dir, seed=cfg["seed"])
    return folds


def analyse_all(corpus, cfg, run_dir, out_dir, log=print):
    os.makedirs(out_dir, exist_ok=True)
    RP.data_inventory(corpus, out_dir)
    RP.methods_parameters(cfg, out_dir)
    res = {}
    log("[RQ1] detection")
    r1, ir, oof, ext = A.rq1(corpus, run_dir, out_dir, n_boot=cfg["n_boot"])
    res["r1"] = r1
    log("[RQ2/3] factor scores")
    S = A.compute_fold_scores(corpus, oof, run_dir)
    Sx = A.compute_external_scores(corpus, ext) if ext is not None and len(ext) else None
    cols = ["idx", "rec", "epoch_idx", "participant", "cohort", "label", "pred", "n3_vote_frac"]
    fs = [S.join(oof[cols]).assign(split="OOF")]
    if Sx is not None:
        fs.append(Sx.join(ext[cols]).assign(split="external"))
    pd.concat(fs, ignore_index=True).to_parquet(os.path.join(out_dir, "factor_scores.parquet"))
    res["r2"] = A.rq2(oof, S, corpus, out_dir, ext, Sx)
    res["r3"] = A.rq3(oof, S, out_dir, ext, Sx)
    log("[RQ4] component analysis")
    res["r4"] = A.rq4(corpus, run_dir, out_dir)
    log("[RQ6] reliability")
    r6 = A.rq6(oof, out_dir, ext)
    res["r6"] = r6
    log("[RQ7] episodes / overnight profiles")
    res["r7"] = A.rq7(oof, S, r6[3], out_dir, ext, Sx, r6[4])
    log("[RQ5] explainability")
    try:
        res["r5"] = X.rq5(corpus, run_dir, out_dir, oof, S, n_per_fold=cfg["xai_epochs_per_fold"])
    except Exception as exc:  # keep the rest of the report if XAI fails
        log(f"[RQ5] failed: {exc}")
        res["r5"] = {"observed": f"FAILED: {exc}", "status": "NOT_RUN"}
    summ = A.rq_summary(out_dir, res["r1"], res["r2"], res["r3"], res["r4"], res["r5"],
                        res["r6"], res["r7"])
    RP.go_no_go(summ, out_dir)
    return res, summ
