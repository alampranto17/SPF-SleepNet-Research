"""Build the reader-facing results notebook (SleepQualityNet_v2_results.ipynb).

The notebook loads the tables and figures that the execution notebook saved
(Kaggle output, an attached Kaggle dataset, a local folder or the GitHub results
folder) and presents them in manuscript order.  It trains nothing, needs no EEG
data and contains no hard-coded result values: every number is read from the
saved files.
"""
import os
import sys

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# SleepQualityNet v2: results in manuscript order

**Results notebook.** It reads the tables and figures written by the execution notebook (`SleepQualityNet_v2_kaggle.ipynb`) and presents them in the order of the manuscript: study design → pre-specified outcomes → RQ1 to RQ7 → decisions and limitations. Nothing is trained and no EEG data is needed; it runs on CPU in about a minute. Every number shown is read from the saved result files, none is typed into this notebook.

### Contents
1. Where the results come from
2. Study design: cohorts, channels, exclusions (Table 1)
3. Pre-specified criteria and outcomes
4. RQ1: N3 detection (Table 2) · RQ1b: same-protocol baselines (Table 3)
5. RQ2: validity of the factor scores · RQ2b: within-N3 slow-wave grade
6. RQ3: stability of the Overall N3 EEG Rating
7. RQ4: component contributions
8. RQ5: explanations
9. RQ6: reliability score
10. RQ7: N3 episodes and the Overnight N3 EEG Profile
11. Go / no-go decisions, Methods parameters and environment
12. Claims: what the results support and what they do not

### Scope
The factor scores, the Overall N3 EEG Rating and the within-N3 grade are rule-derived research-prototype descriptors of the EEG signal. They are not clinical sleep quality and are not validated against expert quality ratings.
"""))

cells.append(md("""## 1. Where the results come from
Set `RESULTS_DIR` to a folder that contains `research_question_results.csv` (for example the `outputs/` folder of a finished Kaggle run, or an unzipped `sqn2_v2_outputs.zip`). Leave it as `None` to search automatically, in this order:
1. `/kaggle/working/outputs` (this notebook run right after the execution notebook);
2. any attached Kaggle input (e.g. the output of the execution notebook added with *Add input → Notebook output*);
3. the latest full run in the GitHub repository (`v2/results/<run>/`), downloaded once as a zip into `./results_cache/`.

The GitHub copy contains tables and figures only. Figures that need per-epoch predictions are included as PNG files; to re-draw them, use the Kaggle output instead."""))

cells.append(code('''RESULTS_DIR = None                     # e.g. "/kaggle/working/outputs" or "./outputs"; None = automatic
GITHUB_REPO = "alampranto17/SPF-SleepNet-Research"
GITHUB_BRANCH = "v2-kaggle"
GITHUB_RUN = "latest"                  # or a run folder name such as "20261027-1633"

import glob, json, os, re, urllib.request
import numpy as np, pandas as pd
from IPython.display import Image, Markdown, display
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 120); pd.set_option("display.max_columns", 40)

def _github_results(repo, branch, run, dest="./results_cache"):
    """Download the branch once as a zip (a single request) and extract one results folder."""
    import io, zipfile
    url = f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
    zf = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=300).read()))
    names = zf.namelist()
    runs = sorted({m.group(1) for n in names
                   for m in [re.match(r"[^/]+/v2/results/([^/]+)/research_question_results\\.csv$", n)] if m}, reverse=True)
    runs = [r for r in runs if "quicktest" not in r]
    if not runs:
        raise FileNotFoundError("no full run with research_question_results.csv on GitHub yet")
    run = runs[0] if run == "latest" else run
    out = os.path.join(dest, run)
    for n in names:
        m = re.match(r"[^/]+/v2/results/" + re.escape(run) + r"/(.+)$", n)
        if m and not n.endswith("/") and not m.group(1).startswith("progress/"):
            tgt = os.path.join(out, m.group(1))
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            with open(tgt, "wb") as f:
                f.write(zf.read(n))
    return out

def find_results():
    if RESULTS_DIR:
        return RESULTS_DIR
    for cand in ["/kaggle/working/outputs"] + sorted(glob.glob("/kaggle/input/**/research_question_results.csv", recursive=True)):
        d = cand if os.path.isdir(cand) else os.path.dirname(cand)
        if os.path.exists(os.path.join(d, "research_question_results.csv")):
            return d
    return _github_results(GITHUB_REPO, GITHUB_BRANCH, GITHUB_RUN)

R = find_results()
FIG = os.path.join(R, "figures")
print("results folder:", os.path.abspath(R))
print(len(glob.glob(os.path.join(R, "*.csv"))), "tables,", len(glob.glob(os.path.join(FIG, "*.png"))), "figures")
if os.path.exists(os.path.join(R, "environment.json")):
    print("run:", json.load(open(os.path.join(R, "environment.json"))).get("run_name"))
'''))

cells.append(code('''# ---- helpers: read a table / show a figure / show a pre-specified verdict ----
def T(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        display(Markdown(f"*`{name}` is not in this results folder (produced by a newer version of the execution notebook).*"))
        return None
    return pd.read_csv(p)

def J(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None

def F(name, width=850):
    p = os.path.join(FIG, name)
    if os.path.exists(p):
        display(Image(p, width=width))
    else:
        display(Markdown(f"*Figure `{name}` is not in this results folder; it is produced by the current execution notebook.*"))

RQ = pd.read_csv(os.path.join(R, "research_question_results.csv"))
def verdict(rq):
    row = RQ[RQ.research_question == rq]
    if row.empty:
        display(Markdown(f"*{rq}: no pre-specified verdict in this run.*")); return
    r = row.iloc[0]
    display(Markdown(f"**Pre-specified criterion ({rq}).** {r.prespecified_criterion}\\n\\n"
                     f"**Observed.** {r.observed}\\n\\n**Verdict: `{r.status}`**"))

def ci(s, d=3):
    """'(0.83, 0.86)' -> '0.830-0.860'"""
    try:
        a, b = [float(x) for x in str(s).strip("()[] ").split(",")]
        return f"{a:.{d}f}-{b:.{d}f}"
    except Exception:
        return ""

def est_ci(v, s, d=3):
    if pd.isna(v):
        return ""
    c = ci(s, d)
    return f"{v:.{d}f} ({c})" if c else f"{v:.{d}f}"

def show(df, caption=None, precision=3):
    """Display a table without the row index, floats rounded."""
    if df is None:
        return
    if caption:
        display(Markdown(caption))
    pcols = {c: "{:.2g}" for c in df.columns if re.search(r"(^p$|^p_|_p$|_p_)", str(c)) and df[c].dtype.kind == "f"}
    st = df.style.hide(axis="index").format(precision=precision, na_rep="")
    if pcols:
        st = st.format(pcols, na_rep="")
    display(st)
'''))

# ---------------------------------------------------------------- design
cells.append(md("""## 2. Study design: cohorts, channels and exclusions
**Table 1** lists every cohort with its role, participants, recordings, scored epochs and available labels and metadata, exactly as found in the cache (fields that do not exist in a dataset are reported as unavailable, not imputed). The channel audit confirms that every recording uses a central EEG derivation and shows the native sampling rate before resampling to 100 Hz. Recordings removed from evaluation are listed with the reason."""))
cells.append(code('''ROLE = {"ISRUC-I": "training", "HMC": "training", "ISRUC-II": "external (test-retest)", "ISRUC-III": "external",
        "DOD-H": "external (5 scorers)", "DOD-O": "external (5 scorers)", "CAP": "R&K S3/S4 reference (grade only)"}
inv = T("data_inventory.csv")
if inv is not None:
    wide = inv.pivot_table(index="dataset", columns="field", values="detail", aggfunc="first")
    stat = inv.pivot_table(index="dataset", columns="field", values="status", aggfunc="first")
    keep = [c for c in ("participants", "recordings", "nights", "EEG channel", "sampling frequency",
                        "sleep-stage labels (scorer 1)", "independent expert scorers", "age / sex",
                        "clinical diagnoses") if c in wide.columns]
    def _cell(v, st):
        v = re.sub(r"np\\.float64\\(([0-9.]+)\\)", r"\\1", str(v)) if isinstance(v, str) else v
        return v if st == "AVAILABLE" else st.replace("_", " ").lower()
    t1 = pd.DataFrame({c: [_cell(v, st) for v, st in zip(wide[c], stat[c])] for c in keep}, index=wide.index)
    order = [c for c in ROLE if c in t1.index] + [c for c in t1.index if c not in ROLE]
    t1 = t1.loc[order]
    t1.insert(0, "role", [ROLE.get(i, "") for i in t1.index])
    show(t1.reset_index(), "**Table 1. Cohorts.**")
aud = T("channel_audit.csv")
if aud is not None:
    display(Markdown("**Channel audit** (recordings per cohort, raw channel and native sampling rate)"))
    ca = aud.groupby(["cohort", "derivation"]).agg(recordings=("recording", "nunique"),
            raw_channel_names=("channel_raw", lambda x: ", ".join(f"{k} ({v})" for k, v in x.value_counts().items())),
            native_fs_Hz=("native_fs", lambda x: ", ".join(f"{v:g}" for v in sorted(x.unique())))).reset_index()
    show(ca)
exc = T("evaluation_exclusions.csv")
if exc is not None:
    display(Markdown("**Evaluation exclusions** (primary model; every reason is reported in the Methods)"))
    e = exc[exc.variant == exc.variant.iloc[0]] if len(exc) else exc
    show(e.groupby(["cohort", "reason"]).agg(recordings=("rec", "nunique"), epochs=("n_epochs", "sum")).reset_index())
'''))

cells.append(md("""## 3. Pre-specified criteria and outcomes
Each research question had a success criterion fixed before any result was seen. The verdicts below are computed mechanically by the execution notebook and are reported unchanged, including criteria that were not met."""))
cells.append(code('''display(RQ.style.hide(axis="index").set_properties(**{"text-align": "left", "white-space": "pre-wrap"})
        .apply(lambda s: ["background-color:#e8f3e8" if v in ("SUPPORTED", "ANSWERED") else
                          "background-color:#fff4dc" if v == "PARTIALLY_SUPPORTED" else
                          "background-color:#fbe4e4" if v == "NOT_SUPPORTED" else "" for v in s], subset=["status"]))
'''))

# ---------------------------------------------------------------- RQ1
cells.append(md("""## 4. RQ1: participant-independent N3 detection
*Can one central EEG channel support binary N3 detection that generalises across participants and cohorts?*

Out-of-fold (OOF) predictions come from 5-fold participant-grouped cross-validation on ISRUC-I + HMC; external cohorts were never used for training, early stopping, temperature scaling or threshold selection. Confidence intervals are participant-level bootstrap intervals. **Table 2** gives the primary metrics per evaluation set."""))
cells.append(code('''verdict("RQ1")
m = T("rq1_detection_metrics.csv")
if m is not None:
    def _name(x):
        x = str(x)
        if x == "pooled_all_cohorts": return "Cross-validation, pooled (ISRUC-I + HMC)"
        if x.startswith("cohort="): return "Cross-validation, " + x[7:]
        if x.startswith("external="): return x[9:].replace(" (5-fold ensemble)", "") + " (external, 5-fold ensemble)"
        if x.startswith("pooled_clean"): return "Cross-validation, QC-clean epochs only (sensitivity)"
        if x.startswith("fold="): return "Cross-validation, fold " + x[5:]
        return x.replace("LOCO_train_", "Leave-one-cohort-out: train ").replace(" -> ", " -> ")
    t2 = pd.DataFrame({
        "evaluation set": [_name(x) for x in m.subset], "participants": m.n_participants.astype("Int64"),
        "epochs": m.n.astype("Int64"), "N3 prevalence": m.prevalence,
        "F1 (95% CI)": [est_ci(v, s) for v, s in zip(m.f1, m.f1_ci95)],
        "PR-AUC (95% CI)": [est_ci(v, s) for v, s in zip(m.pr_auc, m.pr_auc_ci95)],
        "ROC-AUC": m.roc_auc, "precision": m.precision, "recall": m.recall, "Cohen kappa": m.kappa})
    main = m.subset.str.match(r"pooled_all|cohort=|external=")
    show(t2[main.values], "**Table 2. N3 detection, primary model (participant-level bootstrap 95% CI).**")
    show(t2[~main.values], "**Supplementary Table S1. Fold-wise, QC-clean sensitivity and leave-one-cohort-out results.**")
'''))
cells.append(md("""**Error structure.** The confusion matrices (row-normalised) and the share of each expert stage predicted as N3 show where the errors are; for N3 detection the expected confusion is with N2. ROC and precision-recall curves, calibration after temperature scaling and the per-participant F1 distribution complete the picture."""))
cells.append(code('''for f in ("fig_rq1_confusion_matrices.png", "fig_rq1_n3_by_expert_stage.png", "fig_rq1_pr_curves.png",
          "fig_rq1_roc_curves.png", "fig_rq1_calibration.png", "fig_rq1_participant_f1.png"):
    F(f)
'''))
cells.append(md("""**Human benchmark.** For cohorts with several scorers, agreement of the model with each individual expert is compared with the agreement between experts. A model inside the expert-to-expert range performs at the level of human scorers for this task."""))
cells.append(code('''ir = T("rq1_interrater_benchmark.csv")
if ir is not None:
    show(ir[["cohort", "set", "n_raters", "n_participants", "expert_vs_expert_f1_mean", "model_vs_expert_f1_mean",
                "expert_vs_expert_kappa_mean", "model_vs_expert_kappa_mean", "model_within_expert_range"]])
F("fig_rq1_interrater.png")
'''))
cells.append(md("""### RQ1b: same-protocol comparison with published models
DeepSleepNet (2017), TinySleepNet (2020), AttnSleep (2021) and L-SeqSleepNet (2023) were re-implemented and trained on **identical** folds, inputs (raw single-channel EEG), model selection, temperature scaling, threshold rule, loss and class weighting. Differences are paired per participant (Wilcoxon signed-rank, Holm correction). Deviations from each original recipe are listed in `sqn2/baselines.py::BASELINE_NOTES`. **Table 3** reports the cross-validation set and each external cohort."""))
cells.append(code('''verdict("RQ1b")
s = T("rq1_sota_comparison.csv")
if s is not None:
    t3 = pd.DataFrame({"model": s.model, "evaluation set": s.set, "parameters": s.n_params.astype("Int64"),
                       "F1 (95% CI)": [est_ci(v, c) for v, c in zip(s.f1, s.f1_ci95)],
                       "PR-AUC (95% CI)": [est_ci(v, c) for v, c in zip(s.pr_auc, s.pr_auc_ci95)],
                       "ROC-AUC": s.roc_auc.round(3), "kappa": s.kappa.round(3),
                       "train min / fold": s.train_minutes_per_fold.round(1)})
    show(t3, "**Table 3. Same-protocol comparison with published models.**")
pt = T("rq1_sota_paired_tests.csv")
if pt is not None:
    show(caption="**Paired participant-level differences (SleepQualityNet v2 minus baseline)**", df=pt[["comparison", "metric", "n", "median_participant_diff", "ci95_low", "ci95_high", "p", "p_holm"]])
F("fig_rq1_sota.png"); F("fig_rq1_sota_forest.png")
'''))

# ---------------------------------------------------------------- RQ2
cells.append(md("""## 5. RQ2: validity of the factor scores
*Do the rule-derived factor scores (Delta Intensity, Slow-Wave Morphology, Spectral Dominance, N3 Continuity) and the Overall N3 EEG Rating behave as N3 physiology predicts?*

Three independent checks, none of which uses the score definitions as a target: (i) slow-wave activity declines across the night (early vs late detected N3, participant-level); (ii) expert N3 epochs score higher than expert N2 epochs; (iii) in multi-scorer cohorts, the rating increases with the fraction of experts who scored the epoch as N3 (graded agreement)."""))
cells.append(code('''verdict("RQ2")
fv = T("rq2_factor_validity.csv")
if fv is not None:
    show(fv[fv.cohort == "all"][["test", "factor", "n", "median_diff", "ci95_low", "ci95_high", "p"]])
for f in ("fig_rq2_early_vs_late.png", "fig_rq2_factors_by_stage.png", "fig_rq2_graded_expert_agreement.png"):
    F(f)
'''))
cells.append(md("""### RQ2b: Low / Moderate / High within-N3 slow-wave grade
Every detected N3 epoch receives a grade from the fraction of the epoch covered by slow waves (≥ 75 µV, 0.5–2 Hz): **Low** < 20 %, **Moderate** 20–50 % (R&K stage 3 range), **High** > 50 % (R&K stage 4 range). The only expert reference for depth within N3 in public data is R&K S3 vs S4 scoring, so the grade is validated against CAP (R&K), plus within-night decline (G1), graded multi-expert agreement (G2) and ISRUC-II night-to-night stability (G4). The criterion is reported as pre-specified even where it is narrowly missed."""))
cells.append(code('''verdict("RQ2b")
rk = T("rq2_rk_grade_validation.csv")
if rk is not None:
    display(Markdown("**Grade vs expert R&K S3/S4 (CAP)**")); show(rk)
for f in ("rq2_rk_crosstab_detected.csv", "rq2_sw_grade_distribution.csv", "rq2_rating_band_distribution.csv"):
    t = T(f)
    if t is not None:
        display(Markdown(f"`{f}`")); show(t)
gv = T("rq2_n3_grade_validity.csv")
if gv is not None:
    display(Markdown("**G1-G4 validity tests**")); show(gv)
F("fig_rq2b_n3_grade.png")
'''))

# ---------------------------------------------------------------- RQ3
cells.append(md("""## 6. RQ3: stability of the Overall N3 EEG Rating
*Is the combined rating stable under reasonable changes of the factor weights, across participants and between nights?*

The default weights (0.35 / 0.30 / 0.20 / 0.15) are design parameters, not clinically established values. Stability is checked with Dirichlet-perturbed weights around the prior, equal and PCA-derived weights, leaving one factor out, split-half reliability and ISRUC-II test–retest (two recordings of the same person). Because the rating is a weighted sum, its decomposition into factor contributions (weight × score) is exact."""))
cells.append(code('''verdict("RQ3")
st = J("rq3_overall_rating_stability.json")
if st:
    rows = []
    for k, v in st.get("weight_sensitivity", {}).items():
        rows.append(("weight perturbation " + k, "p05 rho", v.get("p05_rho")))
    for k, v in st.get("alternative_weightings", {}).items():
        if not isinstance(v, dict):
            rows.append(("alternative weighting", k, v))
    for k, v in st.get("missing_factor_rho", {}).items():
        rows.append(("leave one factor out", k, v))
    rows.append(("split-half", "Spearman-Brown", st.get("split_half_spearman_brown")))
    for k, v in st.get("test_retest", {}).items():
        if isinstance(v, dict):
            rows.append(("test-retest (ISRUC-II)", k, v.get("icc21")))
    show(pd.DataFrame(rows, columns=["analysis", "quantity", "value"]))
red = T("rq3_factor_redundancy_spearman.csv")
if red is not None:
    display(Markdown("**Factor redundancy (Spearman rho between factors)**")); show(red)
'''))

# ---------------------------------------------------------------- RQ4
cells.append(md("""## 7. RQ4: component contributions
*What does each design component add to N3 detection?* Each comparison changes one component and keeps folds, seeds, preprocessing, targets and metrics identical; differences are paired per participant with Holm correction. A non-significant difference is reported as no measurable effect, not as an improvement."""))
cells.append(code('''verdict("RQ4")
ce = T("rq4_component_effects.csv")
if ce is not None:
    show(ce[["component", "with", "without", "metric", "pooled_with", "pooled_without", "median_diff",
                "ci95_low", "ci95_high", "p_holm", "params_with", "params_without"]])
vm = T("rq4_variant_pooled_metrics.csv")
if vm is not None:
    vm = vm.rename(columns={vm.columns[0]: "variant"})
    show(vm[["variant", "n_params", "f1", "pr_auc", "roc_auc", "kappa", "n3_minutes_icc21"]])
F("fig_rq4_component_forest.png")
'''))

# ---------------------------------------------------------------- RQ5
cells.append(md("""## 8. RQ5: explanations
*Do independent explanation methods point to the physiology of N3?* Frequency-band perturbation and integrated gradients are computed on the network itself, a SHAP surrogate on handcrafted features (with its fidelity R²), and attention is compared with detected slow waves. Agreement is quantified explicitly (top-band agreement, rank correlation per participant). The factor decomposition of the rating is exact and is not a neural-network explanation."""))
cells.append(code('''verdict("RQ5")
x = J("rq5_summary.json")
if x:
    display(pd.Series({k: v for k, v in x.items() if k not in ("observed", "status")}, name="value").to_frame())
sh = T("rq5_shap_surrogate.csv")
if sh is not None:
    display(Markdown("**SHAP surrogate per fold (fidelity and band-group importance)**"))
    show(sh[[c for c in sh.columns if c == "fold" or c == "fidelity_r2" or (c.startswith("shap_"))]])
F("fig_rq5_band_explanations.png")
'''))

# ---------------------------------------------------------------- RQ6
cells.append(md("""## 9. RQ6: reliability score
*Can a reliability score flag epochs whose N3 decision should not be trusted?* Signal quality, model confidence (MC dropout) and out-of-distribution distance are kept as separate components and combined by a cross-fitted logistic model whose target is prediction error. Reported as a 0–100 score, low / medium / high categories and a rejection flag; the risk–coverage curve shows F1 when the least reliable epochs are withheld."""))
cells.append(code('''verdict("RQ6")
for f in ("rq6_reliability_components.csv", "rq6_reliability_categories.csv", "rq6_risk_coverage.csv"):
    t = T(f)
    if t is not None:
        display(Markdown(f"`{f}`")); show(t)
F("fig_rq6_risk_coverage.png")
'''))

# ---------------------------------------------------------------- RQ7
cells.append(md("""## 10. RQ7: N3 episodes and the Overnight N3 EEG Profile
*Can epoch-level outputs be aggregated into episode and whole-night summaries that agree with the expert hypnogram?* An N3 episode is two or more consecutive N3 epochs. Agreement of predicted and expert overnight quantities is given as ICC(2,1), Spearman correlation and Bland–Altman bias with 95 % limits of agreement. The profile summarises N3 in the EEG; it is not an overall sleep-quality measure."""))
cells.append(code('''verdict("RQ7")
pa = T("rq7_profile_agreement.csv")
if pa is not None:
    show(pa)
for f in ("fig_rq7_bland_altman.png", "fig_rq7_profiles.png", "fig_example_overnight_profile.png"):
    F(f)
'''))

# ---------------------------------------------------------------- decisions
cells.append(md("""## 11. Go / no-go decisions, Methods parameters and environment
The go / no-go table maps each verdict to a development decision. `methods_parameters.csv` holds every parameter value quoted in the Methods; `environment.json` the software and hardware of the run."""))
cells.append(code('''for f in ("go_no_go_decisions.csv", "methods_parameters.csv"):
    t = T(f)
    if t is not None:
        display(Markdown(f"`{f}`")); display(t)
env = J("environment.json")
if env:
    display(Markdown("`environment.json`")); display(pd.json_normalize(env).T.rename(columns={0: "value"}))
err = os.path.join(R, "analysis_errors.txt")
if os.path.exists(err):
    display(Markdown("**Analyses that failed in this run** (must be resolved or reported)")); print(open(err).read())
'''))

cells.append(md("""## 12. Claims: what the results support and what they do not
Use the verdict table in §3 as the single source for every claim.

**May be claimed when the corresponding verdict supports it**
* Participant-independent single-channel N3 detection with external validation in independent cohorts, reported per cohort including the cohorts where performance drops.
* Performance relative to published models trained under the same protocol, with paired participant-level tests and parameter counts.
* Agreement with individual experts relative to the expert-to-expert range.
* Physiological plausibility of the rule-derived factor scores (night-time decline, N3 > N2, graded multi-expert agreement).
* Stability of the Overall N3 EEG Rating under weight changes and between nights, with exact factor decomposition.
* A reliability score that identifies error-prone epochs, and overnight N3 quantities that agree with the expert hypnogram.

**Must not be claimed**
* Clinical, restorative or general sleep quality; diagnostic or medical-grade use; personalised clinical recommendations.
* That the factor scores, rating or within-N3 grade are validated against expert quality ratings (no such ratings exist in these datasets).
* That attention or feature fusion improves detection when RQ4 shows no significant effect.
* That a criterion was met when the verdict says otherwise, or results from settings chosen after seeing test results.
"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"},
                  "kaggle": {"accelerator": "none", "isInternetEnabled": True, "isGpuEnabled": False}}
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "SleepQualityNet_v2_results.ipynb")
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
