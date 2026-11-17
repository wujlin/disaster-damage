# Evaluation protocol: predicting when the satellite damage judgment is unreliable

This is a pre-registered evaluation protocol. It is fixed before any statistics are seen. Thresholds, label rules, splits, baselines, metrics, and the fallback cutoff below are part of that registration. They are not to be chosen, tuned, or dropped after results are known.

## Task

For buildings (open-source footprints) without ground photos at inference time, predict from satellite imagery and the footprint alone whether the satellite damage judgment for that building is unreliable.

Supervision comes from buildings that have an independent, non-satellite damage reference (field survey or UAV). xBD/xView2 satellite labels must never be used to define a conflict. Cross-view continued pretraining is used only so that satellite features can hint that a facade may be damaged.

## Scorable test set

The scorable test set contains only footprints that have an independent non-satellite damage reference. The reference, and any ground photo, is hidden at inference. The model sees only satellite imagery and the footprint.

Photo-less buildings without an independent reference are deployment targets only. They are never used for evaluation.

## 1. Label

**Satellite judgment unreliable** means either of the following:

- the satellite-predicted damage grade differs from the independent reference by at least one grade, or
- the satellite predicts no-damage or minor-damage while the reference shows facade damage.

These thresholds are frozen here, before any statistics are computed.

### Frozen definitions (to fill before first statistics run)

The definitions in this section must be filled in and committed before the first statistics run. Grade mapping, facade damage, the footprint-matching rule, and the insufficient-evidence rules below are frozen. Later changes must be logged in the change log with the date and the reason.

- Damage-grade scale mapping between reference sources and the xBD 4-class scale: field references from CAL FIRE DINS wildfire damage inspection records or FEMA damage assessment records map as: No Damage/Unaffected → xBD no-damage; Affected and Minor → minor-damage; Major → major-damage; Destroyed → destroyed. UAV references are labeled by humans to the xBD 4-class guide; no separate grade system.
- Exact definition of "facade damage" in the reference: structural wall damage visible in the ground image — through-cracks; partial or full wall missing/collapsed; wall fire charring; OR flood waterline reaching first-floor interior height. Broken windows alone or surrounding debris alone do NOT count.
- Footprint-to-reference matching rule (IoU / distance threshold): open-source footprint matched to an xBD polygon only if IoU ≥ 0.5. If two or more candidates have IoU ≥ 0.3, treat as ambiguous and drop. A ground photo must attach to exactly one building with localization error ≤ 30 m; otherwise drop. Counts of dropped footprints go in the statistics table.
- Predefined insufficient-evidence rules. These rules assign coverage categories and must never use model uncertainty, entropy, or confidence.

  Satellite evidence is insufficient when no post-event image intersects the footprint, or clear-sky valid pixels on the footprint (not cloud, not cloud shadow, not no-data) are below 80%. Satellite evidence is sufficient only when both of the following hold: a post-event satellite image exists whose intersection with the footprint is nonempty, and clear, valid pixels cover at least 80% of the footprint area. Cloud masks use the imagery's own QA band first. If none exists, use a pre-trained cloud detector whose weights and decision threshold are frozen before any conflict labels are inspected. Do not retune the detector or the 80% cutoff on damage or conflict labels.

  Ground evidence is unavailable when no photo is uniquely attached to the building, localization error is greater than 30 m, the match is ambiguous, or the visual matcher fails the precision frozen on a calibration set (building-ID precision ≥ 0.90; default best/second-best score ratio ≥ 1.2). Ground evidence is usable only when all of the following hold: a ground or UAV photo is attached to exactly one building; localization error is ≤ 30 m; the match is not ambiguous (no second building within 30 m of the photo, and no second footprint candidate with IoU ≥ 0.3); and, if a visual matcher is used, its operating point is frozen on a held-out calibration set to building-ID precision ≥ 0.90. The score ratio of the best to the second-best candidate must be ≥ 1.2, unless the calibration set is large enough to freeze a different margin before test time. Do not retune on the test set.

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

## Change log

- 2026-11-17, protocol drafted from Algorithm Researcher's spec.
- 2026-11-17, froze N, grade mapping, facade damage, and footprint-matching thresholds before the first statistics run.
- 2026-11-17, froze insufficient-evidence rules and coverage classes; metrics stratified by disaster type only.
