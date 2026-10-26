# Literature check and defensible novelty (checked 26 Oct 2026)

## 1. Prior work that limits what can be claimed

| Area | Key prior work | What it does | Implication for this paper |
|---|---|---|---|
| Continuous sleep depth | **SDI**: Zhou, Song, Sun, …, Westover, Hong, *npj Digital Medicine* 2025 (arXiv 2407.04753) | Transformer on 4-channel PSG (C4, EOG, EMG, ECG) trained with pairwise ranking over W<N1<N2<N3; >10k PSGs; biomarkers linked to mortality. Within-N3 variation is not analysed. | Do **not** claim "first continuous depth measure". Ours is N3-specific, single-channel, factor-decomposed and rule-derived. |
| Continuous depth (ECG) | **ECG-SDI**: Xie, Zhao, Zhou, …, Penzel, Hong, arXiv 2610.23727 (Oct 2026) | Same idea from single-lead ECG; reports intra-stage heterogeneity and CPAP response. | Same as above. Cite as concurrent work. |
| Continuous depth (spectral) | **ORP**: Younes et al., *Sleep* 2015; review Younes, *Front. Neurol.* 2023 | 3-s spectral-pattern lookup scaled 0–2.5 (wake-likeness). The review notes most depth change happens before N3 is scored. | ORP measures wake-likeness, not N3 slow-wave expression. Cite. |
| N3 scoring criteria | **Davidson, Sharman, Kyle, Tarassenko**, *Sleep* 2025 (already ref 30) | The 75 µV / 20 % rules lack an empirical basis and bias SWS by age and sex. Proposes amplitude- or power-threshold automatic SWS labelling. | Motivates threshold-free, montage-referenced morphology factors. |
| SWS beyond staging | Sun et al., *Sleep Med.* 2026 (EEMD SWA); Mullins et al., *JSR* 2024; Qin et al., *Sleep* 2025 | SWA metrics track OSA better than N3%. SWA and slow-wave measures are more night-to-night reproducible than N3%. | Supports the expectation tested in RQ3/RQ7 (factor ratings more reproducible than N3 minutes). |
| Slow-wave morphology | e.g. Deshaies-Rugama et al., *SLEEP* 2024 abstract; many hd-EEG studies | Amplitude, slope and density as cortical-synchrony markers. | Morphology features are not new; integrating them into an automated single-channel N3 profile is. |
| Uncertainty / selective staging | **U-PASS**, Heremans et al. 2023 (arXiv 2306.04663); bioRxiv 2025 ensembles + MC dropout (2-channel); **COVER**, arXiv 2610.03911 (2026) | Uncertainty, referral of uncertain epochs, risk-coverage. | A reliability score alone is **not** novel. What is new here: separating signal-quality, model and OOD components for an N3 rating, and validating against **inter-scorer disagreement**. |
| Single-channel staging SOTA | L-SeqSleepNet (Phan et al. 2023): Sleep-EDF Fpz-Cz N3 F1 0.862, SHHS 0.849; TinySleepNet 0.883; XSleepNet ≈ 0.87; SleepTransformer SHHS 0.852 | 5-class models with long sequence context. | Benchmark for RQ1. Context modelling matters, so v2 adds a ±15-epoch dilated temporal CNN. Comparisons are indicative only (5-class vs binary, different splits). |
| Sleep "quality" indices | De Fazio et al., *Sensors* 2026 (staging-derived index fitted to PSQI); Shin et al., *IEEE TNSRE* 2025 (delta–beta PAC vs PSQI) | Whole-night or subjective-quality indices. | Not N3-specific. Our construct is an EEG descriptor, not subjective quality. |

## 2. Novelty claims that survive the check (use "to our knowledge" once)
1. **An N3-specific, factor-decomposed EEG profile from one channel.** It produces delta intensity, slow-wave morphology (threshold-free amplitude and slope plus AASM coverage), spectral dominance and continuity scores, plus an exactly additive overall rating, only for detected N3 epochs. The literature has general depth scales (SDI, ORP) and single biomarkers (SWA, density), but not this decomposition integrated with detection.
2. **Validation against references independent of the score's construction:**
   - within-night homeostatic decline;
   - **graded expert agreement**: scores should rise with the fraction of experts (ISRUC 2, DOD 5) who scored the epoch N3;
   - **test–retest reproducibility** (ISRUC-II, two sessions) against conventional N3 minutes;
   - agreement of the predicted overnight profile with the expert-hypnogram profile.
3. **A reliability score validated against human disagreement**, with signal quality, model confidence and OOD distance reported separately rather than merged without justification.
4. **Multi-site, multi-device external evaluation.** Training on ISRUC-I (Portugal) + HMC (Netherlands); testing on ISRUC-II/III and Dreem DOD-H/O (France, different PSG devices), with leave-one-cohort-out models and a channel-verified pipeline.
5. **Graded expert-agreement validation.** On DOD, five experts score every epoch. The factor scores and the reliability score are tested against the *fraction* of experts voting N3, a graded human reference that two-scorer datasets cannot provide.

## 3. Claims to avoid
- "First continuous sleep-depth / sleep-quality score" (SDI, ORP, ECG-SDI exist).
- "Clinically validated", "restorative", "diagnostic", "sleep quality".
- "Outperforms state of the art" unless an identical protocol is used. Report the published N3 F1 values as context only.
- "Attention explains the decision" or "causal evidence". Use "perturbation sensitivity" and "attribution".

## Datasets
- HMC sleep staging database (PhysioNet) — https://physionet.org/content/hmc-sleep-staging/
- Dreem Open Datasets, Guillot et al. 2020 — https://arxiv.org/pdf/1911.03221 ; Zenodo record — https://zenodo.org/records/15900394
- ISRUC-Sleep — https://sleeptight.isr.uc.pt/

## Sources
- Zhou et al., Continuous Sleep Depth Index Annotation with Deep Learning… — https://arxiv.org/html/2407.04753v2 ; https://www.nature.com/articles/s41746-025-01607-0
- Xie et al., ECG-SDI — https://www.alphaxiv.org/abs/2610.23727
- Younes 2023, New insights… ORP — https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2023.1273623/pdf
- Davidson et al. 2025, Is it time to revisit the scoring of slow wave (N3) sleep? — https://academic.oup.com/sleep/article/48/10/zsaf063/8074201
- Sun et al. 2026, EEMD-derived SWA metric — https://consensus.app/papers/details/3befc9143b0d570d9a0c6d6227ed428d/
- Mullins et al. 2024, Stability of SWS across two nights — https://consensus.app/papers/details/cf61afbda76a529a9cd1986f52a8934f/
- Qin et al. 2025, SWA from multi-night home EEG — https://consensus.app/papers/details/ea4efd5781305c6a81419c3298ca65e9/
- Heremans et al. 2023, U-PASS — https://arxiv.org/pdf/2306.04663
- Trustworthy Sleep Staging (MC dropout, ensembles), bioRxiv 2025 — https://www.biorxiv.org/content/10.1101/2025.08.23.671918.full.pdf
- COVER: Learning to Accept More in Selective Sleep Staging — https://www.alphaxiv.org/abs/2610.03911
- L-SeqSleepNet — https://arxiv.org/pdf/2301.03441
- De Fazio et al. 2026, Sensors 26(13):4091 — https://www.mdpi.com/1424-8220/26/13/4091
- Shin 2026 dissertation / TNSRE 2025 (delta–beta PAC) — https://www.arxiv.org/pdf/2511.15012
- Pan et al. 2026, Front. Neurosci. (sleep staging) — https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1944816/full
