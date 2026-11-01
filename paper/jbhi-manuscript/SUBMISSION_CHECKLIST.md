# JBHI submission checklist (SleepQualityNet v2.1, run 20261030-1528; post hoc runs _posthoc and _posthoc2)

Title (changed after the second internal review): Single-Channel EEG Detection of N3 Sleep with Interpretable Slow-Wave Factor Scores: A Multi-Cohort Study with Multi-Scorer References

Requirements checked against the JBHI "Prepare and submit your manuscript" page (embs.org/jbhi), 31 Oct 2026.

## Already done in this package
| Requirement | Status |
|---|---|
| IEEE two-column template (IEEEtran journal), figures and tables embedded, single PDF | done: `build/main.pdf` |
| Abstract <= 250 words, no references, no equations; objective, methods, results, conclusion, one significance sentence | done (248 words; "F1" and "N3" kept as metric/stage names) |
| Page limit 14 pages incl. supplementary | 10 main + 4 supplementary = 14 |
| IEEE reference style, numbered | done; all 50 cited references checked against PubMed or the publisher/proceedings page (Davidson 2025 author list corrected) |
| Public datasets cited with source | done (Data and code availability section) |
| Code and results availability | done; GitHub `v2/` folder, run `20261030-1528` (v2.1 package pushed today) |
| Cover letter text (portal text box) | done: `cover_letter.txt` (fill the bracketed parts) |

## You must do (cannot be done by Claude)
0. Post hoc Kaggle runs: done (`20261030-1528_posthoc`, `_posthoc2`, `_posthoc3`); all VALUE placeholders filled.
0a. **AI-use disclosure (IEEE policy):** the Acknowledgment now discloses that Claude (Anthropic) assisted in drafting text, code and figures. Confirm or adjust the wording, keep it in the cover letter, and make sure every author has read and verified the manuscript.
0b. **Recommended:** release the five trained fold models (96k parameters each, from the private `sqn2-runs` dataset) on GitHub, and add one sentence to the Data and code availability section. Reviewer comment m17 asks for this.
1. **Authors**: names, affiliations, **institutional or IEEE e-mail for every author** (gmail is not accepted), **ORCID for every author**, corresponding author. Fill the `\author{}` block in `main.tex`.
2. **Funding / acknowledgments / competing interests**: the first-page footnote ("Manuscript received ..." and the funding statement) was removed until the supervisor review. After it, add back in `main.tex` inside `\author{...}`: `\thanks{This work received no external funding.}` or the funder and grant number. The received date is filled by the journal.
3. **Ethics sentence**: public de-identified data; ask your supervisor or institution whether a reference number is needed. JBHI requires an IRB number only for non-public human data.
4. **Author Consent Form** with handwritten signatures of all authors (multi-author papers), uploaded as a supporting document.
5. **Four suggested reviewers**, independent of the work and from institutions different from all authors. Candidates from the cited literature (check independence and get institutional e-mails yourself): Huy Phan, Kaare Mikkelsen, Elisabeth Heremans, Antoine Guillot, Shenda Hong, Diego Alvarez-Estevez, Emadeldeen Eldele, Magdy Younes.
6. **Supervisor read-through** of the full PDF before submission.
7. **Overlength charges**: regular papers over 8 pages pay a mandatory charge (USD 250 per page for pages 9-10). Main text is 10 pages -> about USD 500 if accepted (supplementary counts toward the 14-page limit; check with the editorial office whether supplementary pages are charged). Table V, the risk-coverage figure and the example night are already in the supplementary; reaching 8 pages would need Table IV and Fig. 6 moved as well.
8. **Preprint decision**: if you post to arXiv/medRxiv first, say so in the cover letter.
9. Create the account and submit in the **IEEE Author Portal**; upload `main.pdf`, `supplementary.pdf`, source zip if asked, consent form.

## Decisions worth considering before submission
- **Model name**: the paper avoids calling the method "SleepQualityNet", because the paper explicitly disclaims sleep-quality claims. The repository name can stay.
- **ISRUC-III amplitude scale** (Limitations): amplitude-dependent factors are near the floor in ISRUC-III. A 5-minute check of the source-file units would let you either fix it or keep the disclosure as written.
- **Calibration**: the post hoc notebook compares temperature scaling, the no-class-weighting model and cross-fitted Platt scaling; reported as post hoc in the Discussion.


## Page budget
Main text is exactly 10 pages with about ten free lines at the end of page 10. Filling the author block uses part of that space; if it spills to page 11, ask Claude to trim (page 11 costs USD 350 and breaks the 14-page total).
