# Recommended development path

| Path | Scientific strength | Implementation cost | Annotation cost | Retraining | Publication value | Reviewer risk | Clinical interpretability |
|---|---|---|---|---|---|---|---|
| A: finish v1 paper unchanged | low (channel bug, circular target) | none | none | yes (results invalid) | low | **very high** | low |
| **B: v2 transparent multi-factor prototype** | good (independent references, pre-registered criteria) | done (this repository) | none | yes (Kaggle run) | good (JBHI / Sensors / Physiol. Meas. level) | moderate | moderate |
| C: collect expert ratings first | high | moderate | high | no | high | low | high |
| D: within-N3 pairwise ranking | moderate | moderate | moderate (pairwise judgements) | yes | moderate | moderate | moderate |
| **E: hybrid = B now + expert validation subset** | **highest** | done + small study | moderate (600 epochs × 3 raters) | no extra | **highest** | low | high |

**Primary path: B → E.** Publish v2 (Path B) with honest limitations, and use the expert-annotation protocol as the concrete proposal for an international collaboration (Path E). That second study is what turns the rule-derived ratings into validated ones.
**Fallback:** if RQ2/RQ3 criteria are not met on real data, publish detection, reliability and night profile (RQ1, RQ6, RQ7), report the factor results as negative or partial, and move to Path C.
