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

The definitions in this section must be filled in and committed before the first statistics run. Grade mapping, facade damage, and the footprint-matching rule below are frozen. Later changes must be logged in the change log with the date and the reason.

- Damage-grade scale mapping between reference sources and the xBD 4-class scale: field references from CAL FIRE DINS wildfire damage inspection records or FEMA damage assessment records map as: No Damage/Unaffected → xBD no-damage; Affected and Minor → minor-damage; Major → major-damage; Destroyed → destroyed. UAV references are labeled by humans to the xBD 4-class guide; no separate grade system.
- Exact definition of "facade damage" in the reference: structural wall damage visible in the ground image — through-cracks; partial or full wall missing/collapsed; wall fire charring; OR flood waterline reaching first-floor interior height. Broken windows alone or surrounding debris alone do NOT count.
- Footprint-to-reference matching rule (IoU / distance threshold): open-source footprint matched to an xBD polygon only if IoU ≥ 0.5. If two or more candidates have IoU ≥ 0.3, treat as ambiguous and drop. A ground photo must attach to exactly one building with localization error ≤ 30 m; otherwise drop. Counts of dropped footprints go in the statistics table.
- Predefined "insufficient evidence" rules (no post-event image, cloud cover, low match confidence): TODO

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

Report all of the following, stratified by disaster type and also by evidence source:

- PR-AUC
- precision@K for K = 1%, 5%, and 10%
- ECE (expected calibration error, the gap between confidence and actual accuracy)

## Fallback

N is frozen as follows: at least 300 conflict footprints with independent references in the test set, AND at least 3 events each with ≥30. If either fails, report descriptive results plus small-scale tests only; do not train a dedicated prediction head.

This cutoff was filled and committed before the first statistics run. Later changes must be logged in the change log with the date and the reason.

## Change log

- 2026-11-17, protocol drafted from Algorithm Researcher's spec.
- 2026-11-17, froze N, grade mapping, facade damage, and footprint-matching thresholds before the first statistics run.
