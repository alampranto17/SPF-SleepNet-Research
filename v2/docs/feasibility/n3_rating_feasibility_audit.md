# N3 rating feasibility audit (summary)

* **Data.** Training: ISRUC-I (100) + HMC (~150). External: ISRUC-II (8 × 2 sessions), ISRUC-III (10), DOD-H (25) and DOD-O (55). Single central EEG channel, scored hypnograms, 2 scorers (ISRUC) or 5 scorers (DOD), age, sex and diagnosis for ISRUC from the details sheets. No arousal, respiratory, subjective-quality, clinical-outcome or expert N3-intensity labels (`missing_data_inventory.csv`).
* **v1 code** could not support RQ2-continuity or RQ7 (epoch order lost), and its N3 results are invalid: 83/100 ISRUC subjects were read from an EOG channel, and checkpoints were selected on the test fold (`missing_code_inventory.csv`, `../02_v1_audit_and_supervisor_comments.md`).
* **v2 code** implements every missing component. With current data: RQ1, RQ4, RQ5, RQ6 and RQ7 are FEASIBLE_WITH_CURRENT_DATA; RQ2 and RQ3 are PARTIALLY_FEASIBLE, because the scores are rule-derived and have no expert target (`research_question_feasibility.csv`).
* **Circularity.** Factor scores are never evaluated against the variables used to build them. The references are the homeostatic decline, expert N3 vs N2 labels, the scorer-disagreement continuum, night-to-night reproducibility and the expert-hypnogram profile.
* **Claims.** Rule-derived research prototype only; NO_GO for clinical sleep-quality claims (`go_no_go_decisions_planned.csv`).
* **Path.** B now, then E (`recommended_development_path.md`, `expert_annotation_requirements.md`).
