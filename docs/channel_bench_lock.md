# Channel-bench analysis lock (pre-unblinding)

This document locks the analysis sheet, channel stress bench, and blinded
review sampler **before any Michael results are opened**. It is analysis
development only. No new image head is started here, and no formal fusion
experiment is run from this worktree.

Official scoring and retraining stay unchanged and out of this tree. Official
Michael exports are not inputs to this implementation. All executable checks
in this branch use synthetic fixtures only.

Locked choices below are the analysis contract for this design version
(`channel-bench-lock-v1`). Numeric widths that Jlin must still sign are
marked **UNLOCKED — pending Jlin**. Do not treat an example number as an
official freeze.

## Symbols

- `S`: frozen satellite report (same four-class support as official xBD grades).
- `Y`: aligned ground-survey grade.
- `Z`: conditions available before that building's survey result is inspected.
- Damage prior: `P(Y=y|Z=z)`.
- Observation table: `Q(s|y,z) = P(S=s|Y=y,Z=z)`, estimated by true `Y` rows
  from all eligible reports, then row-normalized over `S`.
- Split `Z = (C, T)`: imaging or visibility conditions `C`, and candidate
  traces `T`. Neighborhood traces must exclude the index building's own alarm.

Grade names follow the official score-contract order in
`mvbridge.score_contract.protocol.OFFICIAL_CLASS_ORDER`, which is the same as
`mvbridge.data.xbd_labels.DAMAGE_CLASSES`: `no-damage`, `minor-damage`,
`major-damage`, `destroyed`. The intact report class is `no-damage`. Later
joins to official inference tables use the score-contract alignment key
`(image_id, building_uid)` when those columns are present. This tree still
does not read official Michael dumps as inputs.

## Metric direction (locked)

| Quantity | Direction | Notes |
| --- | --- | --- |
| Grade-weighted leave-out log score of `Q` on true `S` | Higher is better | Nats per building, mean within each `Y`, then weighted across grades. |
| Channel gain `score(Q1)-score(Q0)` | Positive supports a better table | `Q0` uses `C` only; `Q1` uses `C+T`. Both may share one prior that already includes `C` and `T`. |
| Intact-report evidence log-ratio | Along the locked imprint gradient, the predicted channel direction is **down toward 0** | See formula below. A value near 0 means an intact report barely distinguishes undamaged buildings from the locked damaged mixture. A negative value is evidence in the opposite direction. |
| Budget hit rate at a fixed audit budget | Higher is better | Hits are eligible buildings whose frozen satellite grade differs from the aligned ground grade. |
| Paired block increment (95% interval) | A positive lower bound, with seed-direction agreement when two seeds are present, is the pass direction | Intervals that contain 0, or that are too wide to decide, are reported as such. Do not rewrite “interval too wide” as “no channel effect”. |

The intact-report evidence log-ratio, under a historically locked mixture
`m` over mild/moderate damage grades, is

```
log  Q(S=no-damage | Y=no-damage, z)
   - log  sum_y m(y) Q(S=no-damage | Y=y, z)
```

Multi-class fusion must consume the full likelihood row, not this scalar
alone. The scalar is a pre-specified readout only.

The table is estimated from **all eligible reports**, stratified by true `Y`.
It is then read out at the intact cell. Restricting the denominator to
buildings already reported intact is a protocol error.

## Primary contrasts (locked)

1. **Batch-1 decision table** (does visual mismatch add anything beyond
   geographic common sense?), on one eligible list, with the same buildings
   for both official seeds when two seeds exist:
   - Group-3 neighborhood traces only
   - Group-3 public-geo only
   - Group-3 neighborhood and public-geo combined
   - the same ranking after adding a frozen visual mismatch score
   - entropy and `1-max p` as industry-hesitation baselines
   Official protocol comparison of `U` against entropy and `1-max p` remains
   the protocol ruler and is not redefined here. This sheet asks whether the
   visual score has a positive paired increment over Group 3, especially on
   intact-report buildings with comparable imaging quality and exposure.
2. **Channel test** (Test 2): within each ground grade, compare reporting
   first; then ask whether the intact-report evidence ratio moves with traces
   after holding visibility comparable. Contrast `Q0(C)` vs `Q1(C,T)` with a
   shared prior that already uses `C` and `T`, so prior improvement is not
   credited to the channel.
3. Two synthetic identification families, which must be recovered before any
   real export is read:
   - **Prior-change only**: spatial damage priors change sharply, `Q` does
     not. The sheet must recover mismatch hotspots and must **not** claim
     channel information loss.
   - **Q-change**: the intact row of `Q` also changes along the imprint
     gradient. The sheet must recover a signed evidence-ratio change, and
     both undamaged-row and damaged-row intact-report rates must be reported.

## Negligible-gain band (width UNLOCKED — pending Jlin)

A practical negligible-gain band is required before any ceiling claim
(“for these observations, we have hit an observability upper bound”).
The plan’s example width is **±0.02 nats per building** on the
grade-balanced leave-out log score.

**This 0.02 figure is an example placeholder only.** It is wired as
`EXAMPLE_NEGLIGIBLE_GAIN_BAND_NATS = 0.02` with status
`unlocked_pending_jlin`. It is not an official freeze. Jlin must lock the
numeric width before unblinding. Code may use the placeholder inside
synthetic tests of the *mechanism*, and must refuse to emit an official
ceiling claim while the status remains unlocked.

## Feature splits (locked)

- `C` (imaging / visibility): `visibility_bin`, plus reserved optional
  columns `clear_pixel_fraction`, `gsd_m`, `off_nadir_deg` for later official
  exports. The channel readout is taken only inside a comparable-`C` slice.
- `T` (candidate traces, never including the index building’s own alarm):
  `sat_alarm_density_30m`, `sat_alarm_density_100_300m`,
  `dist_to_nearest_sat_collapsed`, `elevation`, `dist_to_water`.
- Imprint gradient used in Test 2: a scalar `imprint_score` computed from `T`
  alone, without test labels, then cut with **pre-fixed** edges
  `[0, 1/3, 2/3, 1]` (tertiles on a unit-scaled score). The exact production
  mapping from raw `T` columns onto `imprint_score` is **pending Jlin** and
  is not frozen here; fixtures generate `imprint_score` directly.
- Spatial blocks are the resampling unit. The two seeds, when present, share
  the same buildings and the same block-resampling draws for a paired
  interval.

## Review strata (locked)

The sampler locks a canonical score table (SHA-256 of a byte-stable payload)
**before** any image path is attached or viewed.

Locked strata:

- `high_score_error`: true mismatch (`S != Y`) in the high-score tail
  (proposed tail: top 20% of locked scores, pending Jlin confirmation).
- `missed_error`: true mismatch in the low-score region (proposed: bottom
  50% of locked scores, pending Jlin confirmation).
- `matched_condition_correct`: `S == Y`, matched to a sampled error on
  `(visibility_bin, imprint_bin)` when possible.

If a later audit causes a feature change, open a new design version and
mark Michael as a development set for that version.

## Other proposed knobs (not official freezes)

These are implemented so fixtures and tests are deterministic. They are
called out for Jlin rather than silently treated as registered official
numbers:

- Audit budget: 10% of the eligible list.
- Grade weights: equal on the four grades, renormalized over grades with
  evaluation support, with the dropped grades reported.
- Locked mild/moderate mixture `m`: `0.5` minor-damage + `0.5` major-damage
  (`destroyed` is not in this mixture).
- Dirichlet / Laplace smoothing `alpha = 0.5` when estimating `Q`.
- Block bootstrap: 1000 replicates, 95% percentile interval, paired on
  blocks, with a recorded RNG seed.
- Default review quotas: 8 per stratum, or all eligible if fewer, with
  shortfalls reported.
- Leave-out scheme for `Q` scores: leave-block-out.

## Identification fixtures

- If all class likelihoods for the observed report are equal across `Y`,
  the posterior equals the prior (“the prior is unchanged”).
- Prior-change family vs Q-change family as above.
- Analysis must report support: per contrast slice, counts of grades,
  reports, and independent survey blocks.

## What this tree will not do

- Start a new image head, or modify the official DINOv2 ViT-B recipe
  (`image_size` 518, no `pos_embed` interpolation, frozen backbone, damage
  head only, batch 32, bf16, `num_workers=16`).
- Run official scoring, official retraining, or a formal fusion experiment.
- Read official Michael score dumps, checkpoints, or records as inputs.
- Claim a frozen negligible-band width, or a cross-hazard transfer result.
