# Evaluation protocol: predicting when the satellite damage judgment is unreliable

This is a pre-registered evaluation protocol. It is fixed before any statistics are seen. Thresholds, label rules, splits, baselines, metrics, and the fallback cutoff below are part of that registration. They are not to be chosen, tuned, or dropped after results are known.

## Task

For buildings (open-source footprints) without ground photos at inference time, predict from satellite imagery and the footprint alone whether the satellite damage judgment for that building is unreliable.

Supervision comes from buildings that have an independent, non-satellite damage reference (field survey or UAV). xBD/xView2 satellite labels must never be used to define a conflict. Cross-view continued pretraining is used only so that satellite features can hint that a facade may be damaged.

## Scorable test set

The scorable test set contains only footprints that have an independent non-satellite damage reference AND sufficient satellite evidence. Sufficient satellite evidence is the frozen rule: a post-event image intersects the footprint and clear, valid pixels cover at least 80% of the footprint area. That is the satellite+ground and satellite-only coverage classes. The reference, and any ground photo, is hidden at inference. The model sees only satellite imagery and the footprint.

Ground-only and insufficient-evidence footprints are reported in the inventory and descriptive statistics only. They are never scored. The label needs a satellite judgment; otherwise cloud or no-data would create free "unreliable" labels.

Photo-less buildings without an independent reference are deployment targets only. They are never used for evaluation.

## 1. Label

**Satellite judgment unreliable** means either of the following:

- the satellite-predicted damage grade differs from the independent reference by at least one grade, or
- the satellite predicts no-damage or minor-damage while the reference shows facade damage.

These thresholds are frozen here, before any statistics are computed.

The satellite-predicted damage grade used to compute a conflict comes from a frozen xBD baseline damage head (ViT-B, seed 1). That head's training never saw any test event. For events used to train the conflict or unreliability signal, use out-of-fold predictions. The checkpoint hash (sha256 of the checkpoint file) must be recorded in the change log before conflict labels are computed.

### Frozen definitions (to fill before first statistics run)

The definitions in this section must be filled in and committed before the first statistics run. Grade mapping, facade damage, the footprint-matching rule, and the insufficient-evidence rules below are frozen. Later changes must be logged in the change log with the date and the reason.

- Damage-grade scale mapping between reference sources and the xBD 4-class scale: field references from CAL FIRE DINS wildfire damage inspection records or FEMA damage assessment records map as: No Damage/Unaffected → xBD no-damage; Affected and Minor → minor-damage; Major → major-damage; Destroyed → destroyed. UAV references are labeled by humans to the xBD 4-class guide; no separate grade system.
- Exact definition of "facade damage" in the reference: structural wall damage visible in the ground image — through-cracks; partial or full wall missing/collapsed; wall fire charring; OR flood waterline reaching first-floor interior height. Broken windows alone or surrounding debris alone do NOT count.
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

## 2. Split

- **Primary:** event-level holdout.
- **Secondary:** spatial block holdout within an event.

Buildings used to train the conflict signal, and their neighboring spatial blocks, never enter the test set.

## 3. Baselines

Under the same hidden-photo setting (satellite imagery and footprint only; reference and any ground photo hidden), compare against:

- satellite damage head entropy
- satellite damage head max probability (`1 - max prob`)
- random selection

The method must beat these baselines under that setting.

## 4. Metrics

Report all of the following, stratified by disaster type only. Photos are hidden at inference. Coverage classes are reported as descriptive statistics, not as a metric stratification.

- PR-AUC
- precision@K for K = 1%, 5%, and 10%
- ECE (expected calibration error, the gap between confidence and actual accuracy)

## Fallback

N is frozen as follows: at least 300 conflict footprints with independent references in the test set, AND at least 3 events each with ≥30. If either fails, report descriptive results plus small-scale tests only; do not train a dedicated prediction head.

This cutoff was filled and committed before the first statistics run. Later changes must be logged in the change log with the date and the reason.

## Pre-registered claims and theoretical motivation

Training may use satellite, UAV, and ground views. Deployment and evaluation inference use only the post-event satellite image plus the footprint. The claim under test is cross-modal and cross-event transfer of a satellite residual that predicts an unreliable satellite judgment. Fusion accuracy is outside this protocol.

### Notation

\(X^{s}\) is the satellite image together with the footprint. \(X^{g}\) is the ground or UAV view. \(Y\) is the independent non-satellite damage grade. \(F\) is the frozen facade-damage indicator. The frozen satellite judgment is \(\hat{Y}^{s} = h_{\mathrm{frozen}}(X^{s})\), the xBD baseline damage head (ViT-B, seed 1) defined above. The conflict label is

\(U = \mathbf{1}\{\hat{Y}^{s} \neq Y\} \lor \mathbf{1}\{\hat{Y}^{s} \in \{\mathrm{no},\mathrm{minor}\} \land F = 1\}\).

The target is \(\eta(x^{s}) = P(U = 1 \mid X^{s} = x^{s})\). \(U\) is observable only on the scorable set \(Q\) (an independent reference and sufficient satellite evidence). The deployment set is \(D\): sufficient satellite evidence and no \(X^{g}\). \(Z^{s}_{\theta}\) is the satellite representation produced by encoder parameters \(\theta\). \(\theta_{0}\) is the off-the-shelf DINOv2 initialization. \(\theta_{1}\) is the same architecture after SAGIN multi-view continued pretraining. \(g(X^{s})\) is a predictor of \(U\) read from \(X^{s}\).

### Claim 1 (cross-modal transfer)

Under the hidden-photo protocol, a satellite encoder after SAGIN multi-view continued pretraining predicts unreliability better than the same architecture from the off-the-shelf DINOv2 initialization: \(I(Z^{s}_{\theta_{1}}; U) > I(Z^{s}_{\theta_{0}}; U)\). The gain must concentrate on the facade-conflict subset \(\{\hat{Y}^{s} \in \{\mathrm{no},\mathrm{minor}\}, F = 1\}\) and must beat the entropy and \(1 - \max p\) baselines. Any failure falsifies Claim 1.

Required control. A satellite-only continued-pretraining encoder uses exactly the same SAGIN satellite images, the same compute and steps, and the same augmentations, with only the cross-view term removed. Claim 1 is attributed to multi-view transfer only if the SAGIN multi-view encoder beats this control.

### Claim 2 (cross-event transfer)

\(g(X^{s})\) learned on some events with privileged references beats the uncertainty baselines on held-out events. The primary test is a within-disaster-type event holdout. The secondary, harder test is a cross-disaster-type holdout. If the event holdout fails and the within-event spatial-block holdout succeeds, Claim 2 is falsified and only within-event generalization is claimed.

### Identification

\(\eta\) is identifiable from the privileged samples if and only if, on the satellite-sufficient slice, \(U\) is independent of membership in \(Q\) given \(X^{s}\) (selection ignorability). This is Assumption C. It is recorded here and is not treated as holding by default.

### Information decomposition

Information about \(Y\) splits into a redundant part (roof collapse or a large burn visible in both views), a satellite-unique part, and a ground-unique part \(I(X^{g}; Y \mid X^{s})\). Facade conflicts arise mainly from the ground-unique part. They are predictable from \(X^{s}\) only if a satellite residual \(W(X^{s})\) exists with \(I(W; U) > 0\).

Assumption A, which is testable: \(I(X^{s}; U \mid \sigma(\hat{Y}^{s}, c(h_{\mathrm{frozen}}(X^{s})))) > 0\), where \(c\) is the entropy or the maximum probability of the frozen head, and \(\sigma\) is the sigma-algebra those statistics generate. If Assumption A fails, no satellite-only method should beat the uncertainty baselines on facade conflicts.

### Why Claim 1 may hold

Claim 1 is the inequality \(I(Z^{s}_{\theta_{1}}; U) > I(Z^{s}_{\theta_{0}}; U)\), and it is expected to be more pronounced on \(\{\hat{Y}^{s} \in \{\mathrm{no},\mathrm{minor}\}, F = 1\}\). The mechanism is that the multi-view objective forces the satellite encoder to retain factors that are legible in \(X^{g}\) and weak in \(X^{s}\).

Risk, Assumption D. A DINO-style identity alignment may erase damage state as a nuisance. Identity and geometry features therefore stay separate from a damage residual pathway, and the invariance term must not be too strong.

### Shift levels for Claim 2

Three shift levels are distinguished.

- Covariate shift: \(P_{e}(X^{s})\) changes across events while \(P(U \mid X^{s})\) stays invariant. Ranking transfers where the supports overlap.
- Prior shift: \(P_{e}(U)\) changes. ECE degrades. PR-AUC and precision@K may still hold.
- Concept shift: \(P_{e}(U \mid X^{s})\) changes, as when flood, fire, and earthquake physics differ.

Assumption B: \(P(U \mid X^{s}, \tau)\) is invariant within a disaster type \(\tau\), and only within that type.

### Failure modes

- Assumption A fails: facade-subset PR-AUC is near random, and there is no win over the uncertainty baselines.
- Assumption B fails: event holdout drops to the baselines while the spatial-block holdout stays good (an event shortcut).
- Assumption C fails: performance is good on \(Q\) and worse in calibration or ranking on low photo-density slices. \(P(X^{s}, \hat{Y}^{s} \mid Q)\) differs from the same joint on satellite-only buildings.
- Assumption D fails: cross-view distance drops while the facade probe worsens.

### Pre-registered predictions

These predictions are fixed before the first statistics run. They are not to be changed after results are seen.

- **P1.** Freeze the median entropy of non-conflict samples from the training events. Report the fraction of test conflicts whose entropy is below that median. If that fraction is not significantly above 50%, conflicts are mostly high-uncertainty cases and both claims have little room. P1 is the cheapest check and is run first.
- **P2.** The SAGIN gain comes mainly from the facade-conflict subset, not from roof-visible major-damage or destroyed disagreements.
- **P3.** Under hidden photos and event holdout, the method must beat both entropy and \(1 - \max p\). This is necessary for both claims.
- **P4.** Within-type event transfer is stronger than cross-type transfer. If only the former holds, the reported claim is type-conditional transfer.
- **P5.** Coverage bias is an identification threat, not a contribution. Compare \(P(X^{s}, \hat{Y}^{s}, \mathrm{grade})\) between buildings with a photo and buildings without one, and pre-register inverse-propensity or overlap-reweighting robustness checks.
- **P6.** If continued pretraining increases view invariance and lowers the facade probe, weaken the invariance term or add the damage residual head rather than increasing alignment.

## Change log

- 2026-11-17, protocol drafted from Algorithm Researcher's spec.
- 2026-11-17, froze N, grade mapping, facade damage, and footprint-matching thresholds before the first statistics run.
- 2026-11-17, froze insufficient-evidence rules and coverage classes; metrics stratified by disaster type only.
- 2026-11-17, restricted the scorable set to footprints with an independent reference and sufficient satellite evidence, so cloud or no-data cannot create free unreliable labels; froze the conflict-label head as the xBD baseline damage head (ViT-B, seed 1), trained without any test event, with out-of-fold predictions for events used to train the conflict signal; matching ambiguity is only two or more candidates with IoU ≥ 0.3, or a best/second-best score ratio < 1.2, and 30 m is the localization-error limit only.
- Checkpoint hash (sha256 of the frozen xBD baseline checkpoint file): TBD, to be filled before conflict labels are computed.
- 2026-11-17, pre-registered transfer claims, the satellite-only continued-pretraining control, and predictions P1–P6.
