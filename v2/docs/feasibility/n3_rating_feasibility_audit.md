# N3 rating feasibility audit (summary)

* **Data.** ISRUC-I (100), Sleep-EDF SC (78, two nights for most), ISRUC-III (10, external). Single-channel EEG, scored hypnograms, two scorers for ISRUC, age and sex for Sleep-EDF. No arousal, respiratory, subjective-quality, clinical-outcome or expert N3-intensity labels (`missing_data_inventory.csv`).
* **v1 code** could not support RQ2-continuity or RQ7 (epoch order lost), and its N3 results are invalid: 83/100 ISRUC subjects were read from an EOG channel, and checkpoints were selected on the test fold (`missing_code_inventory.csv`, `../02_v1_audit_and_supervisor_comments.md`).
* **v2 code** implements every missing component. With current data: RQ1, RQ4, RQ5, RQ6 and RQ7 are FEASIBLE_WITH_CURRENT_DATA; RQ2 and RQ3 are PARTIALLY_FEASIBLE, because the scores are rule-derived and have no expert target (`research_question_feasibility.csv`).
* **Circularity.** Factor scores are never evaluated against the variables used to build them. The references are the homeostatic decline, expert N3 vs N2 labels, the scorer-disagreement continuum, night-to-night reproducibility and the expert-hypnogram profile.
* **Claims.** Rule-derived research prototype only; NO_GO for clinical sleep-quality claims (`go_no_go_decisions_planned.csv`).
* **Path.** B now, then E (`recommended_development_path.md`, `expert_annotation_requirements.md`).
