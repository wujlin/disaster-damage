# Evaluation protocol: predicting when the satellite damage judgment is unreliable

This is a pre-registered evaluation protocol. It is fixed before any statistics are seen. Thresholds, label rules, splits, baselines, metrics, and the fallback cutoff below are part of that registration. They are not to be chosen, tuned, or dropped after results are known.

## Task

For buildings (open-source footprints) without ground photos at inference time, predict from satellite imagery and the footprint alone whether the satellite damage judgment for that building is unreliable.

Supervision comes from buildings that have an independent, non-satellite damage reference (field survey or UAV). xBD/xView2 satellite labels must never be used to define a conflict. Cross-view continued pretraining is used only so that satellite features can hint that a facade may be damaged. Pretraining can only better extract facade-damage correlates already visible in the satellite image and shared with the ground view. It cannot move ground-private evidence (interior waterline, wall damage under eaves, charring invisible from above) into the satellite image. An invariance term that is too strong may erase satellite-private cues such as roof texture that still predict \(F\) or \(U\).

## Scorable test set

The scorable test set contains only footprints that have an independent non-satellite damage reference AND sufficient satellite evidence. Sufficient satellite evidence is the frozen rule: a post-event image intersects the footprint and clear, valid pixels cover at least 80% of the footprint area. That is the satellite+ground and satellite-only coverage classes. The reference, and any ground photo, is hidden at inference. The model sees only satellite imagery and the footprint.

Ground-only and insufficient-evidence footprints are reported in the inventory and descriptive statistics only. They are never scored. The label needs a satellite judgment; otherwise cloud or no-data would create free "unreliable" labels.

Photo-less buildings without an independent reference are deployment targets only. They are never used for evaluation.

## 1. Label

**Satellite judgment unreliable** means either of the following:

- the satellite-predicted damage grade differs from the independent reference by at least one grade, or
- the satellite predicts no-damage or minor-damage while the reference shows facade damage.

These thresholds are frozen here, before any statistics are computed.

For event \(e\), \(\hat{Y}^{s}\) comes only from the ViT-B seed-1 fold head that never saw \(e\). "Frozen seed 1" means a frozen recipe, seed, and that fold's weight sha256, not one shared head for every event. The official recipe that defines \(U\) sets both `backbone.image_size` and `data.image_size` to 518 (`configs/xbd_dinov2_vitb.yaml`, which `train-xbd --fold` loads). 224 remains a short-side probe only. The weight loader does not interpolate positional embeddings and still rejects a size mismatch. This image size is part of the recipe that enters the definition of \(U\). It is not a backbone swap and not a loader fix. The backbone stays DINOv2 ViT-B/14. Write that sha256 in the change log before any \(U\) for \(e\) is computed. Official xBD labels are never \(Y\).

Let \(\hat{p}\) be the full softmax of that same fold head. The official class order is no-damage, minor-damage, major-damage, destroyed. \(\hat{Y}^{s}\) is the class of greatest probability in \(\hat{p}\). When two or more classes share that maximum within absolute tolerance 1e-12, choose the class with the smallest index in the official class order and mark `tied=True`. The frozen thresholds above are split and reported as follows.

\(U_E = \mathbf{1}\{\hat{Y}^{s} \neq Y\}\).

\(U_F = \mathbf{1}\{\hat{Y}^{s} \in \{\mathrm{no},\mathrm{minor}\} \land F = 1\}\).

\(U = U_E \lor U_F\).

\(U_{F,\mathrm{only}} = \mathbf{1}\{U_F = 1 \land U_E = 0\}\).

\(U_E\), \(U_F\), \(U\), and \(U_{F,\mathrm{only}}\) are all reported.

\(U\) is defined relative to the frozen classifier and the mapping, not an intrinsic property of the building. \(U_F\) is a policy override (a correct minor prediction is still labeled unreliable if a first-floor interior waterline is present). Changing the head changes \(U\).

### Frozen definitions (to fill before first statistics run)

The definitions in this section must be filled in and committed before the first statistics run. Grade mapping, facade damage, the footprint-matching rule, and the insufficient-evidence rules below are frozen. Later changes must be logged in the change log with the date and the reason.

- Damage-grade scale mapping between reference sources and the xBD 4-class scale. CAL FIRE DINS wildfire damage inspection records map as: No Damage / Unaffected → xBD no-damage; Affected → xBD no-damage; Minor → minor-damage; Major → major-damage; Destroyed → destroyed. FEMA damage assessment records map as: No Damage/Unaffected → xBD no-damage; Affected and Minor → minor-damage; Major → major-damage; Destroyed → destroyed. UAV references are labeled by humans to the xBD 4-class guide; no separate grade system.
- Four-class eligibility. The official four classes are no-damage, minor-damage, major-damage, and destroyed. The labels unknown, unclassified, ds1, and -1 are not four-class eligible. The Palu mapping remains unfrozen.
- Exact definition of "facade damage" in the reference: structural wall damage visible in the ground image — through-cracks; partial or full wall missing/collapsed; wall fire charring; OR flood waterline reaching first-floor interior height. Broken windows alone or surrounding debris alone do NOT count. \(F\) is coded from component fields, never from the ordinal grade.
- Footprint-to-reference matching rule (IoU / distance threshold): open-source footprint matched to an xBD polygon only if IoU ≥ 0.5. A match is ambiguous, and is dropped, only when two or more candidates have IoU ≥ 0.3, or the best/second-best match score ratio is < 1.2. The 30 m value is only the upper bound on localization error: a ground photo is dropped when that error is greater than 30 m. Counts of dropped footprints go in the statistics table.
- Predefined insufficient-evidence rules. These rules assign coverage categories and must never use model uncertainty, entropy, or confidence.

  Satellite evidence is insufficient when no post-event image intersects the footprint, or clear-sky valid pixels on the footprint (not cloud, not cloud shadow, not no-data) are below 80%. Satellite evidence is sufficient only when both of the following hold: a post-event satellite image exists whose intersection with the footprint is nonempty, and clear, valid pixels cover at least 80% of the footprint area. Cloud masks use the imagery's own QA band first. If none exists, use a pre-trained cloud detector whose weights and decision threshold are frozen before any conflict labels are inspected. Do not retune the detector or the 80% cutoff on damage or conflict labels.

  Ground evidence is unavailable when no photo is uniquely attached to the building, localization error is greater than 30 m, the match is ambiguous, or the visual matcher fails the precision frozen on a calibration set (building-ID precision ≥ 0.90). A match is ambiguous only when two or more footprint candidates have IoU ≥ 0.3, or the best/second-best match score ratio is < 1.2. The 30 m value is only the upper bound on localization error. Ground evidence is usable only when all of the following hold: a ground or UAV photo is attached to the building; localization error is ≤ 30 m; the match is not ambiguous under the rule above; and, if a visual matcher is used, its operating point is frozen on a held-out calibration set to building-ID precision ≥ 0.90, with best/second-best score ratio ≥ 1.2. Do not retune on the test set. Counts of footprints dropped for ambiguity or for localization error above 30 m go in the statistics table.

  Coverage has exactly four classes, assigned only by these rules and never by model entropy:

  - satellite+ground: sufficient satellite evidence and usable ground evidence
  - satellite-only: sufficient satellite evidence and unavailable ground evidence
  - ground-only: insufficient satellite evidence and usable ground evidence
  - insufficient evidence: insufficient satellite evidence and unavailable ground evidence

  Ground-only buildings are not counted as insufficient evidence. Coverage classes are descriptive inventory statistics only. They are not a stratification of the unreliability metrics.

### CAL FIRE DINS point-to-footprint matching (frozen)

DINS records are points. A point that lies inside exactly one xBD footprint is matched to that building, even if other footprints lie within 10 m. The 10 m rule applies only when the point lies outside every footprint: it then matches if and only if exactly one footprint lies within 10 m. A point inside two or more footprints is dropped as ambiguous. A point outside every footprint, with two or more candidate footprints within 10 m, is dropped as ambiguous. Distance is the shortest point-to-polygon distance, not the distance to the centroid. In dense blocks, neighbouring footprints are routinely within 10 m. The 10 m radius is frozen now and must not be tuned after seeing match rates.

The source must be the official CAL FIRE DINS release. The download date and the file sha256 are recorded in the change log. For Nuns and Tubbs fire points, keep only points inside the xBD `santa-rosa-wildfire` tile extents. Points outside those extents are a coverage issue and are reported separately. They are not counted as matching drops.

The first audit reports only matched counts and drop reasons. Conflict is not defined with xBD official labels. It is computed only after the frozen ViT-B seed 1 head exists.

## 2. Split

- **Primary:** event-level holdout.
- **Secondary:** spatial block holdout within an event.

Buildings used to train the conflict signal, and their neighboring spatial blocks, never enter the test set.

### Leave-one-reference-event-out damage heads

The satellite damage head is DINOv2 ViT-B/14, seed 1, image size 518. Conflict labels are computed later from that head. 224 is a short-side probe only and is not the recipe that defines \(U\). A reference event is an event with an independent ground reference. Each fold holds that event out of one damage head entirely.

For a fold head, every image whose xBD `metadata.disaster` is the held-out event is removed from the official train, test, and hold splits used by that head. Membership is counted from those files by the exact disaster string. The count in each of those three splits is asserted to be 0. The removed images are a separate eval set for that head. They are not used as train or val, and they are not part of that head's in-fold test or hold. In this repository the hold split is named `holdout`. Some releases store it in a directory named `hold`; `data.split_dirs.holdout` points at that directory. There is no separate official val split.

Folds are an appendable list in `configs/folds/event_folds.yaml`. Adding a later fold, such as `minus-hurricane-florence`, is a new list entry. Out-of-fold scoring reads that list. It does not hardcode fold names or a fixed number of reference folds.

A building is scored only by a head that never saw its event. A Michael building (`hurricane-michael`) is routed to the fold that holds out Michael. A Palu building (`palu-tsunami`) is routed to the fold that holds out Palu. A head that saw the event does not produce \(\hat{Y}^{s}\) for that event. Eligible heads are not averaged. Events that no enabled reference fold holds out are not scored.

### Historical source eligibility

\(S\) is the satellite damage class \(\hat{Y}^{s}\). A pair is an eligible historical source for a fold only when every one of the following holds: the event is not this fold's holdout; \(S\) and \(Y\) are in the official four classes; \(R = 1\) and \(V = 1\); the match was accepted; and \(S\) was produced by a head that held out that event. \(R = 1\) means an independent reference is available. \(V = 1\) means sufficient satellite evidence. In-sample reports are recorded and flagged. They do not authorize a historically trained \(Q\).

Active reference folds, seed 1:

- `minus-hurricane-michael` is the main fold. It holds out `hurricane-michael`. It is enabled now. xBD already contains those tiles.
- `minus-palu-tsunami` holds out `palu-tsunami`. It is enabled, and the fold name trains and evaluates end to end.

Pipeline checks, not Claim-2 evidence:

- `minus-santa-rosa` holds out `santa-rosa-wildfire`. It is not the seed-1 main fold.
- `minus-socal` holds out `socal-fire` (Woolsey/socal). Santa Rosa and Woolsey/socal are a wildfire pipeline check, not Claim-2 evidence.

Fold name, held-out event, seed, manifest sha256, and checkpoint sha256 are recorded in `docs/folds.md`. The manifest sha256 is written when the fold is applied to a data root. Checkpoint sha256 stays TBD until that head is trained. No-fold training still uses the full official train split.

Torch is not unified across hosts. Each train, score, and conflict-head evaluation writes `run_env.json` (Python, hostname, torch, CUDA, GPU names, NVIDIA driver). One fold's training and scoring must match that fingerprint. `--cross-env-check` writes a separate directory and is only a comparison. Agreement of \(\hat{Y}^{s}\) is judged per event. `no_effect` is true only when every event is at or above 99%. If any event is below 99%, Palu-fold results are annotated with the version difference. `minus-hurricane-michael-seed2` is a stability repeat and does not define \(U\).

## 3. Baselines

Under the same hidden-photo setting (satellite imagery and footprint only; reference and any ground photo hidden), compare against:

- satellite damage head entropy
- satellite damage head max probability (`1 - max prob`)
- a capacity-controlled baseline \(g_0(\hat{p})\) that sees only the full softmax (and any shared deployment covariates \(W\))
- random selection

The method must beat entropy, \(1 - \max p\), and \(g_0\) under that setting. The P4 main test compares only entropy and \(1 - \max p\). \(g_0\) belongs only to P5.

Public DINOv3 ViT-L/16 SAT-493M weights (timm `vit_large_patch16_dinov3.sat493m`) are reported as a strong satellite-only reference row, evaluated as frozen features with the same probe as the other rows. This row is not the matched control for Claim 1. That control remains continued pretraining on the same SAGIN satellite images, with the same steps and augmentations, and only the cross-view term removed. The SAT-493M weight-file sha256 is TBD and is recorded when the file is copied. The conflict-label head stays the frozen ViT-B seed 1 head. SAT-493M is only a compared method and is never used to define labels.

## 4. Metrics

AP is primary. Compute AP, PR-AUC, precision@K, and ECE on every scorable held-out footprint, never on the conflict subset alone. Photos are hidden at inference. Coverage classes are reported as descriptive statistics, not as a metric stratification. Report these metrics for \(U_E\), \(U_F\), \(U\), and \(U_{F,\mathrm{only}}\).

- AP (primary)
- PR-AUC
- precision@K for K = 1%, 5%, and 10%
- ECE (expected calibration error, the gap between confidence and actual accuracy)

Compare methods only as within-event paired differences. The falsifiers below are one-sided 95% lower bounds from a spatial-block paired bootstrap.

Cancel precision@1% when \(N_e = 30\), or when fewer than one item is expected. \(N_e\) is the number of scorable held-out footprints in event \(e\).

## Fallback

N is frozen as follows: at least 300 conflict footprints with independent references in the test set, AND at least 3 events each with ≥30. If either fails, report descriptive results plus small-scale tests only; do not train a dedicated prediction head.

This cutoff was filled and committed before the first statistics run. Later changes must be logged in the change log with the date and the reason.

## Pre-registered claims and theoretical motivation

Training may use satellite, UAV, and ground views. Deployment and evaluation inference use only the post-event satellite image plus the footprint. Fusion accuracy is outside this protocol. The frozen fallback is unchanged: at least 300 conflict footprints, at least 3 events, and at least 30 conflicts per event.

### Notation

\(X^{s}\) is the satellite image together with the footprint. \(X^{g}\) is the ground or UAV view. \(Y\) is the independent non-satellite damage grade. Official xBD labels are never \(Y\). \(F\) is the frozen facade-damage indicator, coded from component fields, never from the ordinal grade. For event \(e\), \(\hat{Y}^{s}\) and the full softmax \(\hat{p}\) come only from the ViT-B seed-1 fold head that never saw \(e\). \(H\) is the entropy of \(\hat{p}\), computed with the natural logarithm and reported in nats. Ranking is invariant to log base. \(U_E\), \(U_F\), \(U\), and \(U_{F,\mathrm{only}}\) are the labels in the Label section. The target on the scorable set is \(\eta(x^{s}) = P(U = 1 \mid X^{s} = x^{s})\). \(R = 1\) means an independent reference is available. \(V = 1\) means sufficient satellite evidence. \(Q = \{R = 1, V = 1\}\). \(R = 1\) is not the same event as "has a ground photo". \(U\) is observable only on \(Q\). The deployment set is \(D\): sufficient satellite evidence and no \(X^{g}\). \(W\) is any shared deployment covariates. \(S\) is the satellite damage class \(\hat{Y}^{s}\). \(Z\) is currently an open pre-disaster covariate mapping. The protocol does not yet freeze a field table for \(Z\). \(g(X^{s})\) is a predictor of \(U\) read from \(X^{s}\). \(g_0(\hat{p})\) is the capacity-controlled baseline that sees only the full softmax (and any shared deployment covariates \(W\)). \(\phi(X^{s})\) is a satellite representation. \(\phi_{\mathrm{cross}}\) is the representation after cross-view pretraining. \(\phi_{\mathrm{sat}}\) is the satellite representation compared with \(\phi_{\mathrm{cross}}\) in P2 and P3. The SAT-493M row in Baselines is a frozen-feature reference row only. It is not \(\phi_{\mathrm{sat}}\), and it is never used to define labels. \(m_0\) is defined under P1.

### Assumptions

Assumption A is conditioned on the full softmax \(\hat{p}\). It is split into A-data and A-repr. A-repr does not imply beating entropy.

- **A-data.** \(I(X^{s}; U \mid \hat{p}) > 0\).
- **A-repr.** \(I(\phi(X^{s}); U \mid \hat{p}) > 0\). A-repr does not imply beating entropy.

Assumption B is the Q–D overlap form. Claim 2 is restricted to the overlapping support of \(Q\) and \(D\). Covariate shift is allowed. Concept shift is not identifiable from unlabeled target images.

- Sub-point. \(P(U \mid X^{s}, \tau)\) is assumed invariant only within disaster type \(\tau\).

Assumption C is the transport form, with P6 diagnostics. Selection ignorability is not a default. It is replaced by a transport assumption plus truncated IPW and Gamma sensitivity bounds (e.g. 1, 2, 4).

Assumption D is the testable form, plus a shuffled-pair control. A DINO-style identity alignment may erase damage state as a nuisance. Identity and geometry features therefore stay separate from a damage residual pathway, and the invariance term must not be too strong. The testable check is that cross-view distance does not drop while the facade probe worsens. If pretraining starts, add a shuffled-pair control: the satellite–ground pairing is shuffled, so the control does not keep true cross-view pairs.

### Claim 1

Claim 1 is better extraction of satellite-visible facade-damage correlates shared with the ground view.

Under the hidden-photo protocol, the satellite encoder after SAGIN multi-view continued pretraining predicts \(U\) better than the same architecture from the off-the-shelf DINOv2 initialization. The gain is attributable to the cross-view term only relative to the required satellite-only continued-pretraining control (same SAGIN satellite images, same steps/compute, same augmentations, cross-view term removed).

The information-processing inequality is \(I(\phi(X^{s}); F) \le I(X^{s}; F)\).

If pretraining starts, add a shuffled-pair control.

The matched control named in the SAT-493M baseline paragraph is that satellite-only continued-pretraining control. SAT-493M stays a frozen-feature reference row only.

### Claim 2

Claim 2 is restricted to the overlapping support of \(Q\) and \(D\). Covariate shift is allowed. Concept shift is not identifiable from unlabeled target images. The main Claim 2 text is restricted to that overlap. Any statement about \(D\) uses a transport assumption plus truncated IPW and Gamma sensitivity bounds (e.g. 1, 2, 4). The event-level holdout and the within-event spatial-block holdout in the Split section do not extend Claim 2 off that overlap.

The primary test is a within-disaster-type event holdout. The secondary, harder test is a cross-disaster-type holdout. If only the within-type test holds, the reported claim is type-conditional transfer. The Assumption B sub-point applies: \(P(U \mid X^{s}, \tau)\) is assumed invariant only within disaster type \(\tau\).

### Failure modes

- A fails if there is no win over entropy, \(1 - \max p\) and \(g_0\) (facade-subset AP near random).
- A-data fails when \(I(X^{s}; U \mid \hat{p})\) is not greater than 0.
- A-repr does not imply beating entropy.
- B fails as an event shortcut: event holdout drops to the baselines while the within-event spatial-block holdout stays good.
- Assumption B fails when a Claim 2 statement leaves the overlapping support of \(Q\) and \(D\), or when concept shift is treated as identified from unlabeled target images.
- Assumption C fails when selection ignorability is used as a default, or when a statement about \(D\) omits truncated IPW and Gamma sensitivity bounds (e.g. 1, 2, 4). P6 is the overlap diagnostics.
- Assumption D fails when cross-view distance drops while the facade probe worsens. If pretraining starts, a missing shuffled-pair control means the pretraining comparison is not interpretable under D.

### Pre-registered predictions

P1–P6 are replaced by P1–P7. These predictions are fixed before the first statistics run. They are not to be changed after results are seen. The falsifiers are one-sided 95% lower bounds from a spatial-block paired bootstrap. For a prediction that is a positive within-event paired difference, the lower bound not greater than 0 fails to support that prediction.

- **P1.** Freeze the median entropy of non-conflict samples (\(U = 0\)) from the training events. That frozen median is \(m_0\). \(q = P(H < m_0 \mid U = 1, Q)\). The one-sided 95% lower bound on \(q\) not greater than 0.5 fails to support the overconfident-conflict narrative.
- **P2.** AP difference of \(\phi_{\mathrm{cross}}\) versus \(\phi_{\mathrm{sat}}\) on \(F\), restricted to no/minor (\(\hat{Y}^{s} \in \{\mathrm{no},\mathrm{minor}\}\)). The slice is scorable footprints in that grade band, never the conflict subset alone.
- **P3.** AP difference of \(\phi_{\mathrm{cross}}\) versus \(\phi_{\mathrm{sat}}\) on \(U\), on every scorable held-out footprint.
- **P4.** The primary test is pooled across qualified events. A qualified event is a held-out event with at least 30 conflict footprints, the per-event bar already frozen in Fallback. Within each qualified event \(e\), compute the paired AP difference \(\Delta_e = \mathrm{AP}_e(g) - \mathrm{AP}_e(b)\) for each baseline \(b\) in {entropy, \(1 - \max p\)}. The pooled statistic is the equal-weight mean \(\bar{\Delta} = \frac{1}{K}\sum_e \Delta_e\) over the \(K\) qualified events, so no single event dominates by size. Uncertainty comes from a spatial-block bootstrap that resamples blocks within each event and recomputes every \(\Delta_e\). P4 passes only if the one-sided 95% lower bound of that equal-weight mean is above 0 versus entropy and versus \(1 - \max p\); both must pass. The capacity-controlled baseline \(g_0(\hat{p})\) is not part of the P4 pass criterion. The P4 main test compares only entropy and \(1 - \max p\). \(g_0\) belongs only to P5. As a secondary, non-decision report alongside that primary mean, also report the conflict-count-weighted mean of those within-event paired AP differences, with the same within-event spatial-block bootstrap. The equal-weight mean remains primary. The pooled test runs only when the frozen Fallback holds (at least 300 conflicts in total and at least 3 qualified events). Each \(\Delta_e\) is reported with its bootstrap interval and sign as direction only. No per-event significance claim is made, and the number of events with \(\Delta_e > 0\) out of \(K\) is reported. A pooled win whose sign is driven by one event (leave-one-event-out mean \(\le 0\)) is reported as such.
- **P5.** \(g\) versus \(g_0\), as a within-event paired AP difference, using the same pooled estimator as P4 (equal-weight mean of within-event paired AP differences, spatial-block bootstrap within events, one-sided 95% lower bound above 0) as its own falsifier. The method must beat \(g_0\).
- **P6.** Overlap diagnostics for the transport form of Assumption C. P6 does not add a numeric overlap cutoff.
- **P7.** Frozen versus conservative mapping sensitivity. Report P2–P5 under the frozen mapping and under the conservative mappings in Source mapping.

## Source mapping

The frozen CAL FIRE DINS and FEMA mapping sentences in the Label section stay in force. FEMA mapping stays frozen. DINS Affected → no stays frozen (no rollback). Mappings in this section that are not those frozen sentences are the approved StEER package, the Palu non-mapping, or a pre-registered sensitivity. They are not a rollback.

### StEER wind

StEER wind is frozen as follows. Code \(F\) from component fields, never from the grade.

- 0 → no
- 1 → minor (does not satisfy \(F\))
- 2 → minor, never major
- 3 → major only with recorded partial roof or wall structural failure, otherwise minor or unknown
- 4 is not automatically destroyed. StEER wind 4 maps to major by default, and the building stays in \(Q\). Wind 4 is promoted to destroyed only when a component record shows the building as a whole is no longer an intact structure, in the sense of xBD destroyed ("completely collapsed ... or otherwise no longer present") and StEER surge 6 ("no intact structure"). That requires at least one of: (1) an explicit complete-collapse, building-no-longer-present, or no-intact-structure field is recorded; or (2) wall structure failure in the living area (Table 4 note [2]; ground-floor breakaway walls of elevated buildings do not count) and roof structure recorded as failed or removed so that no intact roof structure remains. Roof structure > 15% alone does not qualify. The Table 4 shaded indicators for grade 4 (window or door failure > 50%, roof or wall substrate > 25%, roof structure > 15%, wall structure failure) define grade 4 and are not used to promote it. If the fields are missing or do not meet the rule, the building stays major and is listed as "wind 4 not promoted" with its observed component values. No numeric threshold beyond these is added.

  Source: Prevatt and Roueche, 15 June 2020, Hurricane Michael Data Enhancement (Phase II), Report No. 06-20, University of Florida, for Florida DBPR, Project P0157245, Table 4 and Table 5, printed p. 15.
- −1 is excluded from wind \(Y\) and is not no-damage

Source for the component-field tables: Roueche, Nakayama, Kijewski-Correa, Prevatt, 2024, Natural Hazards Review 25(2):04023060, printed page 04023060-6, Tables 3 and 4. Which cells of WiSPD Table 3 are bold is unverified (footnote a depends on bold criteria). Do not cite those bold cells as if read.

### Surge

Surge 0–6 are damage states, not feet. Do not map feet, regional 8–12 ft inundation, or `first_floor_elevation_feet` to \(Y\). Surge 3 is not \(F\). 5 → major. 6 → destroyed. This package does not freeze a grade for any other surge code.

StEER Hurricane Michael EARR, printed p. 3: 8–12 ft is a regional hazard observation. Do not back-fit the 2023 Unified App Handbook onto the 2018 Michael field protocol.

### Palu

Palu is not one-to-one and is not frozen until Paulik et al. (2019) is in hand. DS1 → unknown or excluded from a strict crosswalk. DS2/DS3 only as exploratory sensitivity. The labels unknown, unclassified, ds1, and -1 are not four-class eligible. The Palu mapping remains unfrozen.

Paulik et al., 2019, Pure and Applied Geophysics 176:3305–3321, doi:10.1007/s00024-019-02254-9. Full text not obtained.

Secondary Palu only: Lahcene et al., 2021, NHESS 21:2315–2316, Table 1; Jalayer et al., 2023, NHESS 23:914, Table 2. Jalayer's phrase "complete structural collapse" is Jalayer's wording, not a verified Paulik definition.

### DINS sensitivity and FEMA

DINS Affected → no stays frozen (no rollback). It is flagged as conflicting with the xBD no-damage wording on shingle damage, quoted under Gupta Figure 3 below. Pre-registered sensitivity: Affected → minor and Affected → unknown. Those two sensitivities are not the frozen map. FEMA mapping stays frozen.

DINS methodological table: Camp Incident Damage Inspection Report, NIST-hosted, methodological table on Page 4 of 54. FEMA categories as used in the frozen sentence: 2016 Damage Assessment Category Guidelines, Florida Division of Emergency Management-hosted 2-page PDF, pp. 1–2. Official fema.gov Damage Assessment Operations Manual (2016-04-05) page numbers were not verified.

### Official xBD wording

Gupta et al., 2019, Creating xBD, CVPR Workshops, conference proceedings pp. 12–13, Figure 3. The body on p. 12 also calls it Table 3. Figure 3 is the Joint Damage Scale on a four-level scheme. The body states: "As such, this scale is not meant as an authoritative damage assessment rating, but it does provide the first attempt to create a unified assessment scale for building damage in satellite imagery across multiple disaster types, structure categories, and geographical location."

Figure 3 descriptions:

- 0 (no damage): Undisturbed. No sign of water, structural or shingle damage, or burn marks.
- 1 (minor damage): Building partially burnt, water surrounding structure, volcanic flow nearby, roof elements missing, or visible cracks.
- 2 (major damage): Partial wall or roof collapse, encroaching volcanic flow, or surrounded by water/mud.
- 3 (destroyed): Scorched, completely collapsed, partially/completely covered with water/mud, or otherwise no longer present.

Official xBD labels are never \(Y\). The no-damage line is the shingle-damage wording that conflicts with frozen DINS Affected → no. The conflict is flagged. The frozen map is not rolled back.

### Unverified, not cited as read

The following were not read for this revision. Do not cite them as if read.

- FAST Handbook v1.2 (2019)
- Vickery et al. (2006)
- Friedland 2009 dissertation
- Friedland and Levitan 2009 conference PDF page numbers
- official fema.gov Damage Assessment Operations Manual (2016-04-05) page numbers
- which cells of WiSPD Table 3 are bold (footnote a depends on bold criteria)
- a standalone official DINS "No Damage" definition

## Construct validity and pre-registered remedies

The weakest point is construct validity of \(Y\) and of \(U\), not the model architecture. xBD is a satellite proxy scale and "is not meant as an authoritative damage assessment rating" (Gupta et al., 2019, CVPR Workshops, conference proceedings pp. 12–13, Figure 3; the body on p. 12 also calls it Table 3).

Remedies, in order:

1. Indicator-blind grade adjudication.
2. Retain unknown.
3. Report rater agreement.
4. Code \(F\) from component fields, never from the ordinal grade.
5. Report P2–P5 under frozen and conservative mappings (P7).
6. Do not interpret a win as satellite unreliability if it appears only under a controversial one-to-one mapping.
7. Restrict main Claim 2 text to the overlapping support of \(Q\) and \(D\), with truncated IPW and Gamma bounds for any statement about \(D\).

P7 reports P2–P5 under the frozen mappings and under the conservative mappings. The frozen mappings are the Label-section DINS and FEMA sentences and the frozen StEER wind rules. The conservative mappings are the pre-registered sensitivities: DINS Affected → minor and Affected → unknown. Palu is not one-to-one and is not frozen. DS1 → unknown or excluded from a strict crosswalk. DS2/DS3 only as exploratory sensitivity. Do not interpret a win as satellite unreliability if it appears only under a controversial one-to-one mapping. Palu remains unfrozen.

## Change log

- 2026-11-17, protocol drafted from Algorithm Researcher's spec.
- 2026-11-17, froze N, grade mapping, facade damage, and footprint-matching thresholds before the first statistics run.
- 2026-11-17, froze insufficient-evidence rules and coverage classes; metrics stratified by disaster type only.
- 2026-11-17, restricted the scorable set to footprints with an independent reference and sufficient satellite evidence, so cloud or no-data cannot create free unreliable labels; froze the conflict-label head as the xBD baseline damage head (ViT-B, seed 1), trained without any test event, with out-of-fold predictions for events used to train the conflict signal; matching ambiguity is only two or more candidates with IoU ≥ 0.3, or a best/second-best score ratio < 1.2, and 30 m is the localization-error limit only.
- Checkpoint hash (sha256 of the frozen xBD baseline checkpoint file): TBD, to be filled before conflict labels are computed. After this revision the hash is recorded per fold head, for the head that never saw the event being labeled.
- 2026-11-17, pre-registered transfer claims, the satellite-only continued-pretraining control, and predictions P1–P6.
- 2026-11-17, froze the CAL FIRE DINS point-to-footprint matching rule, including the 10 m radius.
- Official CAL FIRE DINS release (POSTFIRE_MASTER_DATA_SHARE, https://gis.data.cnra.ca.gov/datasets/CALFIRE-Forestry::cal-fire-damage-inspection-dins-data), downloaded 2026-11-17 17:44 Asia/Shanghai (UTC+8); statewide CSV 132,522 rows, sha256 af01e965145420695504db3385782c6879dee00dc66586feec8a6bfc0938404f. Tubbs + Nuns 2017: 7,577 points (Tubbs 6,023, Nuns 1,554). Inside xBD santa-rosa-wildfire tile extents: 5,516 (all Tubbs); outside: 2,061 (all 1,554 Nuns and 507 Tubbs), reported as coverage, not matching drops. DINS grades of the 5,516 in-extent points: Destroyed 5,206, Affected 158, Minor 60, No Damage 67, Major 25. Santa Rosa is a pipeline check only; the protocol still requires at least 3 events, and Claim 2 cannot rest on this event alone.
- 2026-11-17, clarified DINS in-footprint precedence: a point inside exactly one footprint matches even if other footprints lie within 10 m. The 10 m rule applies only to points outside every footprint, and distance is the shortest point-to-polygon distance, not the distance to the centroid.
- 2026-11-17, added public DINOv3 ViT-L/16 SAT-493M weights (timm `vit_large_patch16_dinov3.sat493m`) as a strong satellite-only reference row, evaluated as frozen features with the same probe as the other rows. It is not the matched control for Claim 1, which remains continued pretraining on the same SAGIN satellite images with the same steps and augmentations and only the cross-view term removed. Weight file sha256: TBD, recorded when copied. The conflict-label head stays the frozen ViT-B seed 1 head; SAT-493M is only a compared method and is never used to define labels.
- 2026-11-17, factual correction of the frozen CAL FIRE DINS grade mapping, made against the official DINS grade definition, before any conflict labels are computed. It is not a post-hoc change after seeing conflict rates. CAL FIRE DINS Affected is 1–9% cosmetic or interior damage; the building remains habitable (roof-tile replacement, peeling paint, melted siding, broken windows, sinks, or accessory decks/carports). That does not meet this protocol's frozen facade-damage definition (structural wall damage or wall fire charring; broken windows alone do not count). Mapping Affected to minor-damage would invent unreliable labels. Santa Rosa matched pairs already show 127 of 146 Affected points labeled no-damage in official xBD, which is consistent with this definition, but the change is based on the source definition, not on that table. Source: NIST Camp Fire DINS standard write-up, https://www.nist.gov/system/files/documents/2020/11/16/2018%20Camp%20Incident%20DINS%20Final%20Report.pdf. Jlin approved this correction on 2026-11-17. Multi-DINS-point consistency on one footprint must use this mapping once it is merged (Affected and No Damage on the same footprint now agree). FEMA damage assessment records keep the previous mapping.
- 2026-11-17, Jlin approved three protocol packages. This entry writes them into this file only, before any \(U\) is computed. (1) Theory and evaluation corrections. Assumption A is conditioned on the full softmax \(\hat{p}\) and split into A-data and A-repr; A-repr does not imply beating entropy. Add a capacity-controlled baseline \(g_0(\hat{p})\) that sees only the full softmax (and any shared deployment covariates \(W\)). Split \(U\) into \(U_E\) and \(U_F\) and also report \(U_{F,\mathrm{only}}\). Claim 1 is better extraction of satellite-visible facade-damage correlates shared with the ground view, with \(I(\phi(X^{s}); F) \le I(X^{s}; F)\). If pretraining starts, add a shuffled-pair control. Claim 2 is restricted to the overlapping support of \(Q\) and \(D\). Covariate shift is allowed. Concept shift is not identifiable from unlabeled target images. Selection ignorability is not a default; it is replaced by a transport assumption plus truncated IPW and Gamma sensitivity bounds (e.g. 1, 2, 4). Compute AP / PR-AUC / precision@K / ECE on every scorable held-out footprint, never on the conflict subset alone. Compare methods only as within-event paired differences. Cancel precision@1% when \(N_e = 30\), or when fewer than one item is expected. P1–P6 are replaced by P1–P7 with one-sided 95% lower-bound falsifiers (spatial-block paired bootstrap). For event \(e\), \(\hat{Y}^{s}\) comes only from the ViT-B seed-1 fold head that never saw \(e\). "Frozen seed 1" means a frozen recipe, seed, and that fold's weight sha256, not one shared head for every event. (2) StEER / Palu / DINS sensitivity. StEER wind: 0 → no; 1 → minor (does not satisfy \(F\)); 2 → minor, never major; 3 → major only with recorded partial roof or wall structural failure, otherwise minor or unknown; 4 is not automatically destroyed; −1 is excluded from wind \(Y\) and is not no-damage. Code \(F\) from component fields, never from the grade. Surge 0–6 are damage states, not feet. Do not map feet, regional 8–12 ft inundation, or first_floor_elevation_feet to \(Y\). Surge 3 is not \(F\); 5 → major; 6 → destroyed. Palu is not one-to-one and is not frozen until Paulik et al. (2019) is in hand. DS1 → unknown or excluded from a strict crosswalk; DS2/DS3 only as exploratory sensitivity. DINS Affected → no stays frozen (no rollback), flagged as conflicting with xBD no-damage wording on shingle damage, with pre-registered sensitivity Affected → minor and Affected → unknown. FEMA mapping stays frozen. (3) Construct-validity remedies. The weakest point is construct validity of \(Y\) and of \(U\), not the model architecture. xBD is a satellite proxy scale and "is not meant as an authoritative damage assessment rating" (Gupta et al., 2019, CVPR Workshops pp. 12–13, Figure 3). Remedies, in order: indicator-blind grade adjudication; retain unknown; report rater agreement; code \(F\) from component fields; report P2–P5 under frozen and conservative mappings (P7); do not interpret a win as satellite unreliability if it appears only under a controversial one-to-one mapping; restrict main Claim 2 text to the overlapping support of \(Q\) and \(D\), with truncated IPW and Gamma bounds for any statement about \(D\). Sources: Gupta et al., 2019, Creating xBD, CVPR Workshops, conference proceedings pp. 12–13, Figure 3 (the body on p. 12 also calls it Table 3); Camp Incident Damage Inspection Report, NIST-hosted, methodological table on Page 4 of 54; Roueche, Nakayama, Kijewski-Correa, Prevatt, 2024, Natural Hazards Review 25(2):04023060, printed page 04023060-6, Tables 3 and 4; 2016 Damage Assessment Category Guidelines, Florida Division of Emergency Management-hosted 2-page PDF, pp. 1–2. Official fema.gov Damage Assessment Operations Manual (2016-04-05) page numbers were not verified. StEER Hurricane Michael EARR, printed p. 3 (8–12 ft is a regional hazard observation). Paulik et al., 2019, Pure and Applied Geophysics 176:3305–3321, doi:10.1007/s00024-019-02254-9, full text not obtained. Secondary Palu only: Lahcene et al., 2021, NHESS 21:2315–2316, Table 1; Jalayer et al., 2023, NHESS 23:914, Table 2. Jalayer's phrase "complete structural collapse" is Jalayer's wording, not a verified Paulik definition. Unverified, and not cited as read: FAST Handbook v1.2 (2019); Vickery et al. (2006); Friedland 2009 dissertation; Friedland and Levitan 2009 conference PDF page numbers; official fema.gov manual page numbers; which cells of WiSPD Table 3 are bold (footnote a depends on bold criteria); a standalone official DINS "No Damage" definition. Do not back-fit the 2023 Unified App Handbook onto the 2018 Michael field protocol. Do not compute \(U\) until this entry is merged and the relevant fold-head sha256 values are written. Palu remains unfrozen. Claim 1 and Claim 2 test structure, the type-conditional rule, and failure-mode checks from the earlier registration are retained.
- 2026-11-17, P4 changed from per-event tests to a pooled primary test (equal-weight mean of within-event paired AP differences, spatial-block bootstrap within events, one-sided 95% lower bound); single events report direction only. Reason: power simulation showed a single event with 30 conflicts has power 0.03-0.09 at a 0.10 AP gain. Changed before any U was computed. Fallback numbers, mappings, and matching rules are unchanged.
- 2026-11-17, froze StEER wind 4: default major; destroyed only with an explicit collapse or no-longer-present field, or living-area wall structure failure with no intact roof structure remaining. Table 4 grade-4 shading is not a promotion rule. Source: Prevatt and Roueche 2020, Report 06-20, Table 4, printed p. 15. Frozen before any U was computed. Other mappings unchanged.
- 2026-11-17, P4 pass criterion restricted to entropy and 1 - max p; the g0 comparison stays in P5 and is not bundled into P4. Added the conflict-count-weighted mean of within-event paired AP differences as a secondary report; the equal-weight mean remains primary. Changed before any U was computed. Mappings, fallback numbers, and matching rules are unchanged.
- 2026-11-17, recorded the leave-one-reference-event-out rule for the satellite damage head (ViT-B, seed 1). The main fold is `minus-hurricane-michael`. `minus-palu-tsunami` is the second enabled reference fold. `minus-santa-rosa` and `minus-socal` are pipeline checks, not Claim-2 evidence. Santa Rosa and Woolsey/socal are a wildfire pipeline check. Manifest and checkpoint sha256 values for these heads stay TBD in `docs/folds.md` until a head is trained. This entry does not change grade mapping, matching thresholds, the 10 m DINS rule, or other frozen protocol numbers.
- 2026-11-17, removed the torch release assertion. Jlin chose option (b): do not unify torch across hosts. Each train, score, and conflict-head evaluation records Python, hostname, torch, CUDA, GPU names, and the NVIDIA driver in `run_env.json`.
- 2026-11-17, a fold's training, scoring, and conflict-head evaluation must use the same host, torch, CUDA, driver, and GPU names. A mismatch is refused unless `--cross-env-check` writes a separate directory.
- 2026-11-17, cross-env agreement of \(\hat{Y}^{s}\) at or above 99% counts as no effect. Below 99%, Palu-fold results are annotated with the version difference. Results stay stratified by event.
- 2026-11-17, `minus-hurricane-michael-seed2` is a stability repeat (`role: seed-repeat`, `claim_evidence: false`, seed 2). It does not define \(U\). Default routing uses the seed-1 Michael head only.
- 2026-11-17, removed the unused `torch_version` field. Cross-env `no_effect` is true only when every event's agreement is at least 99%; the summary lists per-event agreement and the events that fail. A prediction row must have `image_id` and `building_uid`, and those keys must be unique.
- 2026-11-17, Jlin reconfirmed the official xBD DINOv2 ViT-B fold recipe that defines \(U\): `backbone.image_size` and `data.image_size` are 518 in `configs/xbd_dinov2_vitb.yaml`, which `python -m mvbridge train-xbd --fold` loads. Fold entries do not override image size. This is a recipe change that enters the definition of \(U\). It is not a backbone swap and not a loader fix. The backbone stays DINOv2 ViT-B/14. The weight loader does not interpolate positional embeddings and still rejects size mismatches. 224 remains a short-side probe only. Batch size 32, AMP `bfloat16`, and `freeze: true` are unchanged. Changed before any \(U\) was computed.
- 2026-11-17, Jlin confirmed the five score-contract rules as official protocol. Ties take the smallest index in the official class order and are marked `tied=True`, with absolute tolerance 1e-12. Entropy uses the natural logarithm (nats); ranking is invariant to log base. Historical-source eligibility requires that the event is not this fold's holdout, that \(S\) and \(Y\) are in the official four classes, that \(R = 1\) and \(V = 1\), that the match was accepted, and that \(S\) was produced by a head that held out that event. In-sample reports are recorded and flagged and do not authorize a historically trained \(Q\). \(Z\) stays an open pre-disaster covariate mapping; no field table is frozen. The labels unknown, unclassified, ds1, and -1 are not four-class eligible, and the Palu mapping remains unfrozen. The P4 main test still compares only entropy and \(1 - \max p\); \(g_0\) belongs only to P5. No previously frozen threshold, mapping, or training recipe is changed.
