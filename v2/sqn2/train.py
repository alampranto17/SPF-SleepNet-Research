"""Corpus loading, participant-grouped folds and the GPU-resident trainer.

Protocol (fixes v1 selection leakage):
  * outer 5-fold split by PARTICIPANT, stratified by cohort;
  * inside each training set, 15 % of participants (per cohort) form an inner
    validation set used for early stopping, threshold selection and
    temperature scaling -- the outer test fold is never looked at;
  * identical folds, seeds and chunk sampling for every ablation variant.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, cohen_kappa_score,
                             matthews_corrcoef, balanced_accuracy_score)

from . import data as D
from . import features as FT
from .models import SleepQualityNetV2, count_params
from .scores import montage_family

VARIANTS = {
    # name                 use_feat pos_weight attention context
    "cnn":                 dict(use_feat=False, pos_weight=False, use_attention=True, context=False),
    "cnn_feat":            dict(use_feat=True, pos_weight=False, use_attention=True, context=False),
    "cnn_pw":              dict(use_feat=False, pos_weight=True, use_attention=True, context=False),
    "cnn_feat_pw":         dict(use_feat=True, pos_weight=True, use_attention=True, context=False),
    "cnn_feat_pw_noattn":  dict(use_feat=True, pos_weight=True, use_attention=False, context=False),
    "full_v2":             dict(use_feat=True, pos_weight=True, use_attention=True, context=True),
    # same-protocol published baselines (raw EEG only; see baselines.py)
    "deepsleepnet":        dict(arch="deepsleepnet", use_feat=False, pos_weight=True),
    "tinysleepnet":        dict(arch="tinysleepnet", use_feat=False, pos_weight=True),
    "attnsleep":           dict(arch="attnsleep", use_feat=False, pos_weight=True),
    "lseqsleepnet":        dict(arch="lseqsleepnet", use_feat=False, pos_weight=True),
}
PRIMARY_VARIANT = "full_v2"
ABLATION_VARIANTS = [k for k, v in VARIANTS.items() if "arch" not in v]
BASELINE_VARIANTS = [k for k, v in VARIANTS.items() if "arch" in v]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
class Corpus:
    """All cached recordings concatenated in chronological order."""

    def __init__(self, cache_dir, feat_dir=None, device="cpu", cohorts=None, log=print,
                 max_recordings=None):
        files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        if max_recordings:
            files = files[:max_recordings]
        feat_dir = feat_dir or os.path.join(cache_dir, "_features")
        os.makedirs(feat_dir, exist_ok=True)
        X, rows, recs, feats, qcs = [], [], [], [], []
        self.labels_multi = {}
        off = 0
        for i, p in enumerate(files):
            arr, meta = D.load_cache(p)
            if cohorts and meta["cohort"] not in cohorts:
                continue
            ep = arr["epochs"].astype(np.float32)
            # float16 cache: |x| > 65504 uV (amplifier saturation / broken channel) became inf.
            # Such recordings are flagged (signal_valid=False) and excluded from every evaluation;
            # the signal is sanitised so no model ever sees non-finite input.
            fin_ep = np.isfinite(ep).all(axis=1)
            n_nonfinite = int((~fin_ep).sum())
            if n_nonfinite:
                ep = np.nan_to_num(ep, nan=0.0, posinf=0.0, neginf=0.0)
            fpath = os.path.join(feat_dir, os.path.basename(p).replace(".npz", ".parquet"))
            if os.path.exists(fpath):
                ft = pd.read_parquet(fpath)
            else:
                ft = pd.DataFrame(FT.recording_features(ep))
                ft.to_parquet(fpath)
            n = len(ep)
            sd = ep.std(axis=1)
            scale = float(np.median(sd[sd > 1e-3])) if (sd > 1e-3).any() else 1.0
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            xs = np.clip(ep / scale, -60000.0, 60000.0)        # keep float16 finite
            X.append(xs.astype(np.float16))
            qc = {k: arr[k] for k in arr if k.startswith(("qc_", "flag_", "rk_"))}
            qcs.append(pd.DataFrame(qc))
            feats.append(ft)
            meta.update({"offset": off, "n": n, "raw_scale": scale,
                         "montage": montage_family(meta["derivation"]),
                         "n_nonfinite_epochs": n_nonfinite, "signal_valid": n_nonfinite == 0})
            if n_nonfinite:
                log(f"[corpus] {meta['recording']}: {n_nonfinite} epochs with non-finite signal "
                    f"(float16 overflow) -> recording excluded from evaluation")
            recs.append(meta)
            lm = arr["labels_multi"].astype(int)
            scored = lm >= 0
            n_r = scored.sum(1)
            frac = np.where(n_r > 0, ((lm == 3) & scored).sum(1) / np.maximum(n_r, 1), np.nan)
            rows.append(pd.DataFrame({
                "rec": meta["recording"], "participant": meta["participant"],
                "cohort": meta["cohort"], "montage": meta["montage"],
                "is_primary": meta["is_primary"], "epoch_idx": arr["epoch_index"],
                "label": arr["labels"].astype(int),
                "label2": lm[:, 1] if lm.shape[1] > 1 else np.full(n, -2),
                "n_raters_epoch": n_r, "n3_vote_frac": frac,
                "all_votes_n2n3": (scored.any(1) & np.all(~scored | np.isin(lm, [2, 3]), 1)),
                "signal_valid": np.full(n, n_nonfinite == 0)}))
            self.labels_multi[meta["recording"]] = lm
            off += n
            if (i + 1) % 25 == 0:
                log(f"[corpus] {i + 1}/{len(files)} recordings loaded")
        self.rec = pd.DataFrame(recs).set_index("recording", drop=False)
        self.ep = pd.concat(rows, ignore_index=True)
        self.feat = pd.concat(feats, ignore_index=True)
        self.qc = pd.concat(qcs, ignore_index=True)
        self.ep = pd.concat([self.ep, self.qc], axis=1)
        self.Xcpu = np.concatenate(X)
        self.device = torch.device(device)
        self.X = torch.from_numpy(self.Xcpu).to(self.device)
        self.Fraw = FT.detection_matrix(self.feat)
        self.y = torch.from_numpy(self.ep["label"].to_numpy().copy()).to(self.device)
        self.participant_meta = {}
        pm = os.path.join(cache_dir, "participant_metadata.json")
        if os.path.exists(pm):
            with open(pm) as f:
                self.participant_meta = json.load(f)
        log(f"[corpus] {len(self.rec)} recordings, {len(self.ep):,} epochs, "
            f"{self.ep.participant.nunique()} participants; X on {self.device} "
            f"({self.X.element_size() * self.X.nelement() / 1e9:.2f} GB)")

    def idx_of(self, recs):
        return np.concatenate([np.arange(self.rec.loc[r, "offset"],
                                         self.rec.loc[r, "offset"] + self.rec.loc[r, "n"])
                               for r in recs]) if len(recs) else np.zeros(0, int)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------
TRAIN_COHORTS = ("ISRUC-I", "HMC")
EXTERNAL_COHORTS = ("ISRUC-II", "ISRUC-III", "DOD-H", "DOD-O")   # pre-specified RQ1 externals
GRADING_COHORTS = ("CAP",)    # external, R&K S3/S4 expert reference for the within-N3 grade only


def make_folds(corpus, k=5, seed=42, train_cohorts=TRAIN_COHORTS, inner_frac=0.15):
    """Participant-level folds stratified by cohort; returns list of dicts."""
    rng = np.random.default_rng(seed)
    rec = corpus.rec
    parts = rec[rec.cohort.isin(train_cohorts)].groupby("participant")["cohort"].first()
    fold_of = {}
    for coh, g in parts.groupby(parts):
        p = np.array(sorted(g.index))
        rng.shuffle(p)
        for i, pid in enumerate(p):
            fold_of[pid] = i % k
    folds = []
    for f in range(k):
        test_p = sorted([p for p, ff in fold_of.items() if ff == f])
        train_all = [p for p, ff in fold_of.items() if ff != f]
        inner = []
        for coh in train_cohorts:
            cp = sorted([p for p in train_all if parts[p] == coh])
            rng2 = np.random.default_rng(seed + 100 + f)
            rng2.shuffle(cp)
            inner += cp[: max(1, int(round(inner_frac * len(cp))))] if cp else []
        train_p = sorted(set(train_all) - set(inner))
        prim = rec[rec.is_primary]
        folds.append({
            "fold": f,
            "train_recs": list(prim[prim.participant.isin(train_p)].index),
            "inner_recs": list(prim[prim.participant.isin(inner)].index),
            "test_recs": list(rec[rec.participant.isin(test_p)].index),
            "external_recs": list(rec[~rec.cohort.isin(train_cohorts)].index),
            "train_participants": train_p, "inner_participants": sorted(inner),
            "test_participants": test_p,
        })
    return folds


def loco_split(corpus, train_cohort, seed=42, inner_frac=0.15):
    rec = corpus.rec
    prim = rec[rec.is_primary]
    p = sorted(prim[prim.cohort == train_cohort].participant.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(p)
    inner = p[: max(1, int(round(inner_frac * len(p))))]
    train = p[len(inner):]
    other = rec[(rec.cohort != train_cohort) & rec.is_primary]
    return {"fold": f"LOCO_train_{train_cohort}",
            "train_recs": list(prim[prim.participant.isin(train)].index),
            "inner_recs": list(prim[prim.participant.isin(inner)].index),
            "test_recs": list(other.index), "external_recs": [],
            "train_participants": train, "inner_participants": inner,
            "test_participants": sorted(other.participant.unique())}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def binary_metrics(y, prob, thr):
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, float)
    pred = (prob >= thr).astype(int)
    out = {"n": int(len(y)), "prevalence": float(y.mean()) if len(y) else np.nan,
           "threshold": float(thr)}
    if len(np.unique(y)) < 2:
        out.update({k: np.nan for k in ("pr_auc", "roc_auc")})
    else:
        out["pr_auc"] = float(average_precision_score(y, prob))
        out["roc_auc"] = float(roc_auc_score(y, prob))
    out.update({
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 else np.nan,
        "balanced_acc": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
    })
    return out


def best_f1_threshold(y, prob):
    y = np.asarray(y).astype(int)
    qs = np.unique(np.quantile(prob, np.linspace(0.5, 0.995, 200)))
    grid = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), qs]))
    f1s = [f1_score(y, prob >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def fit_temperature(logits, y, iters=200):
    lg = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        T_ = logT.clamp(np.log(0.05), np.log(20.0)).exp()
        loss = Fnn.binary_cross_entropy_with_logits(lg / T_, yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(np.clip(logT.exp().item(), 0.05, 20.0))


def ece(y, prob, bins=15):
    y, prob = np.asarray(y), np.asarray(prob)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - prob[m].mean())
    return float(e)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self, corpus, cfg):
        self.c = corpus
        self.cfg = cfg
        self.dev = corpus.device

    # -- helpers --------------------------------------------------------
    def _chunks(self, recs, L, rng):
        starts = []
        for r in recs:
            o, n = int(self.c.rec.loc[r, "offset"]), int(self.c.rec.loc[r, "n"])
            if n < L:
                continue
            sh = int(rng.integers(0, L))
            starts.extend(range(o + sh, o + n - L + 1, L))
        return np.array(starts, dtype=np.int64)

    def _predict_recs(self, model, recs, F, mc=0):
        model.eval()
        out = {"idx": [], "logit": [], "emb": [], "mc_p": []}
        with torch.no_grad(), torch.autocast(self.dev.type, enabled=self.dev.type == "cuda"):
            for r in recs:
                o, n = int(self.c.rec.loc[r, "offset"]), int(self.c.rec.loc[r, "n"])
                x, f = self.c.X[o:o + n], F[o:o + n]
                lg, e = model.predict_sequence(x, f, return_emb=True)
                out["idx"].append(np.arange(o, o + n))
                out["logit"].append(lg.float().cpu().numpy())
                out["emb"].append(e.float().cpu().numpy())
                if mc:
                    for m in model.modules():
                        if isinstance(m, torch.nn.Dropout):
                            m.train()
                    ps = [torch.sigmoid(model.predict_sequence(x, f).float()).cpu().numpy()
                          for _ in range(mc)]
                    out["mc_p"].append(np.stack(ps, 1))
                    model.eval()
        res = {k: (np.concatenate(v) if v else None) for k, v in out.items()}
        return res

    # -- main -----------------------------------------------------------
    def run(self, variant, split, out_dir, seed=42, gbm=False):
        os.makedirs(out_dir, exist_ok=True)
        tag = f"{variant}__fold{split['fold']}"
        path = os.path.join(out_dir, tag + ".npz")
        if os.path.exists(path):
            return path
        cfg, c = self.cfg, self.c
        t0 = time.time()
        tr_idx = c.idx_of(split["train_recs"])
        lab_tr = tr_idx[c.ep.label.to_numpy()[tr_idx] >= 0]
        scaler = FT.RobustScaler().fit(c.Fraw[lab_tr])
        Fs = torch.from_numpy(scaler.transform(c.Fraw)).to(self.dev)
        z_tr = (c.Fraw[lab_tr] - scaler.med) / scaler.iqr
        feat_clip_frac = float(np.mean(np.abs(np.nan_to_num(z_tr)) >= 10))  # reported in Methods
        y_np = c.ep.label.to_numpy()
        n3_tr = int((y_np[lab_tr] == 3).sum())
        pw = (len(lab_tr) - n3_tr) / max(n3_tr, 1)

        if gbm:
            return self._run_gbm(split, path, scaler, tag, t0)

        v = VARIANTS[variant]
        fs = _fold_seed(seed, split["fold"])
        torch.manual_seed(fs)
        rng = np.random.default_rng(fs)
        if "arch" in v:
            from . import baselines as BL
            model = BL.build(v["arch"]).to(self.dev)
        else:
            model = SleepQualityNetV2(n_feat=c.Fraw.shape[1], use_feat=v["use_feat"],
                                      use_attention=v["use_attention"],
                                      context=v["context"]).to(self.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        amp = self.dev.type == "cuda"
        gscaler = torch.amp.GradScaler("cuda", enabled=amp)
        pos_w = torch.tensor(pw if v["pos_weight"] else 1.0, device=self.dev)
        L, B = cfg["chunk_len"], cfg["chunks_per_batch"]
        inner_y = (y_np[c.idx_of(split["inner_recs"])] == 3).astype(int)
        inner_mask = y_np[c.idx_of(split["inner_recs"])] >= 0
        best, best_state, bad, hist = -1, None, 0, []
        n_steps_total = cfg["max_passes"] * max(1, len(self._chunks(split["train_recs"], L, rng)) // B)
        step = 0
        warm = max(1, min(cfg["warmup_steps"], n_steps_total // 10))   # <=10 % of steps
        for ep in range(cfg["max_passes"]):
            model.train()
            starts = self._chunks(split["train_recs"], L, rng)
            rng.shuffle(starts)
            tl, nb = 0.0, 0
            for b in range(0, len(starts) - B + 1, B):
                lr = cfg["lr"] * min(1.0, (step + 1) / warm) * \
                    0.5 * (1 + math.cos(math.pi * min(step / max(n_steps_total, 1), 1.0)))
                for g in opt.param_groups:
                    g["lr"] = lr
                s = torch.from_numpy(starts[b:b + B]).to(self.dev)
                idx = (s[:, None] + torch.arange(L, device=self.dev)[None, :])
                x = c.X[idx].float()
                f = Fs[idx]
                yy = c.y[idx]
                if cfg.get("augment", True):
                    x = x * (0.9 + 0.2 * torch.rand(B, 1, 1, device=self.dev))
                    x = x + 0.02 * torch.randn_like(x)
                m = yy >= 0
                with torch.autocast(self.dev.type, enabled=amp):
                    logit = model(x, f)
                loss = Fnn.binary_cross_entropy_with_logits(
                    logit.float()[m], (yy[m] == 3).float(), pos_weight=pos_w)
                opt.zero_grad(set_to_none=True)
                gscaler.scale(loss).backward()
                gscaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                gscaler.step(opt)
                gscaler.update()
                tl += float(loss.detach())
                nb += 1
                step += 1
            pr = self._predict_recs(model, split["inner_recs"], Fs)
            if not np.isfinite(pr["logit"]).all():
                raise FloatingPointError(f"{tag}: non-finite logits after pass {ep + 1} "
                                         f"(numerical overflow) - variant skipped")
            p_in = _sigmoid(pr["logit"])
            ap = average_precision_score(inner_y[inner_mask], p_in[inner_mask]) \
                if inner_y[inner_mask].any() else 0.0
            hist.append({"pass": ep + 1, "train_loss": tl / max(nb, 1), "inner_pr_auc": ap})
            if cfg.get("verbose", True):
                print(f"  {tag} pass {ep + 1:02d} loss={tl / max(nb, 1):.4f} "
                      f"innerPR-AUC={ap:.4f} ({time.time() - t0:.0f}s)")
            if ap > best + 1e-4:
                best, bad = ap, 0
                best_state = {k: v_.detach().clone() for k, v_ in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
        model.load_state_dict(best_state)
        # inner-val calibration and threshold (outer test fold untouched)
        pr = self._predict_recs(model, split["inner_recs"], Fs)
        lg_in = pr["logit"][inner_mask]
        T = fit_temperature(lg_in, inner_y[inner_mask])
        p_in = _sigmoid(lg_in)
        thr = best_f1_threshold(inner_y[inner_mask], p_in)
        # training-embedding statistics for Mahalanobis OOD distance
        sub = rng.choice(lab_tr, size=min(len(lab_tr), 20000), replace=False)
        tr_recs_sub = split["train_recs"]
        ptr = self._predict_recs(model, tr_recs_sub, Fs)
        pos = {i: k for k, i in enumerate(ptr["idx"])}
        sel = np.array([pos[i] for i in sub])
        E = ptr["emb"][sel]
        Y = (y_np[sub] == 3).astype(int)
        mus = [E[Y == k].mean(0) for k in (0, 1)]
        cov = np.cov(np.concatenate([E[Y == k] - mus[k] for k in (0, 1)]).T) + \
            1e-3 * np.eye(E.shape[1])
        icov = np.linalg.inv(cov)
        eval_recs = list(split["test_recs"]) + list(split["external_recs"])
        pt = self._predict_recs(model, eval_recs, Fs, mc=cfg["mc_passes"])
        d = np.stack([np.einsum("ij,jk,ik->i", pt["emb"] - mu, icov, pt["emb"] - mu)
                      for mu in mus], 1)
        maha = np.sqrt(np.min(d, 1))
        mc = pt["mc_p"]
        pm = mc.mean(1)
        H = -(pm * np.log(pm + 1e-8) + (1 - pm) * np.log(1 - pm + 1e-8))
        EH = -(mc * np.log(mc + 1e-8) + (1 - mc) * np.log(1 - mc + 1e-8)).mean(1)
        np.savez_compressed(
            path, idx=pt["idx"], logit=pt["logit"].astype(np.float32),
            prob=_sigmoid(pt["logit"]).astype(np.float32),
            prob_cal=_sigmoid(pt["logit"] / T).astype(np.float32),
            mc_mean=pm.astype(np.float32), mc_entropy=H.astype(np.float32),
            mc_mi=(H - EH).astype(np.float32), mc_std=mc.std(1).astype(np.float32),
            maha=maha.astype(np.float32), threshold=thr, temperature=T,
            inner_pr_auc=best, n_params=count_params(model), pos_weight=float(pos_w),
            seconds=time.time() - t0, history=json.dumps(hist), feat_clip_frac=feat_clip_frac,
            meta=json.dumps({k: split[k] for k in split if k != "fold"} |
                            {"fold": str(split["fold"]), "variant": variant}))
        torch.save(model.state_dict(), path.replace(".npz", ".pt"))
        print(f"  {tag}: done in {time.time() - t0:.0f}s  thr={thr:.3f} T={T:.2f} "
              f"innerPR-AUC={best:.4f} params={count_params(model):,}")
        return path

    def _run_gbm(self, split, path, scaler, tag, t0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        c = self.c
        y = c.ep.label.to_numpy()
        tr = c.idx_of(split["train_recs"])
        tr = tr[y[tr] >= 0]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             random_state=0)
        clf.fit(c.Fraw[tr], (y[tr] == 3).astype(int))
        inn = c.idx_of(split["inner_recs"])
        inn = inn[y[inn] >= 0]
        thr = best_f1_threshold((y[inn] == 3).astype(int), clf.predict_proba(c.Fraw[inn])[:, 1])
        ev = c.idx_of(list(split["test_recs"]) + list(split["external_recs"]))
        p = clf.predict_proba(c.Fraw[ev])[:, 1]
        np.savez_compressed(path, idx=ev, prob=p.astype(np.float32), threshold=thr,
                            seconds=time.time() - t0,
                            meta=json.dumps({k: split[k] for k in split if k != "fold"} |
                                            {"fold": str(split["fold"]), "variant": "feat_gbm"}),
                            n_params=0)
        print(f"  {tag}: GBM done thr={thr:.3f}")
        return path


def _sigmoid(x):
    from scipy.special import expit       # numerically stable (no overflow warnings)
    return expit(np.asarray(x, dtype=np.float64))


def _fold_seed(seed, fold):
    """Deterministic per-fold seed (Python's hash() is randomised per process)."""
    return int(seed) * 1000 + sum(ord(ch) for ch in str(fold)) % 997


def load_run(path):
    z = np.load(path, allow_pickle=False)
    return {k: (z[k] if z[k].shape else z[k].item()) for k in z.files}
