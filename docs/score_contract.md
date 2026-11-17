# Score contract

This document is the schema for `feat/score-contract`. It does not change the
official training recipe, does not start official Michael scoring, and does not
write official output directories or checkpoints.

The implementation lives in `src/mvbridge/score_contract/`. Tests use only
synthetic probabilities and synthetic (S, Y, Z) fixtures from
`mvbridge.score_contract.fixtures`.

## What this worktree provides

1. **Protocol reader.** `load_protocol_contract` reads `docs/eval_protocol.md`,
   checks that the frozen P4/P5 and baseline sentences are still present, and
   reads the official recipe file `configs/xbd_dinov2_vitb.yaml` without
   modifying it. Image size 518, frozen backbone, and batch size 32 are
   asserted. `authorized_artifact_inventory` reads the shipped fold list in
   `configs/folds/event_folds.yaml`. Manifest and checkpoint sha256 values are
   copied as written. Null or `TBD` stays unset. The reader never hashes a
   real checkpoint and never invents a digest.
2. **Score adapter.** `build_inference_table` consumes a four-class probability
   vector in official order `(no-damage, minor-damage, major-damage, destroyed)`
   and writes entropy (nats), `1 - max p`, the predicted class, and a tie flag.
   Evaluation labels are not read. `align_inference_and_labels` later joins on
   the unique key `(image_id, building_uid)`.
3. **Split inventory.** `inventory_historical_pairs_by_fold` lists historically
   eligible `(S, Y, Z)` pairs for each outer fold, records the reporting-model
   version, and records whether the historical report was out-of-sample. A fold
   with no eligible historical source event is a valid deliverable: its table
   role is `descriptive`, and `historically_trained_q_allowed` is false.

## Frozen comparisons

- The P4 main test compares only `entropy` and `one_minus_max_p`.
- `g0` belongs only to P5. Asking P4 to include `g0` is an error.
- This worktree does not implement or fit `g0`.

## Deterministic rules that need sign-off if the protocol should freeze them

- **Tie break.** When two or more classes share the maximum probability (within
  absolute tolerance `1e-12`), the adapter reports the tied class with the
  lowest official index, and sets `tied=True`. The protocol text does not yet
  name a tie rule.
- **Entropy units.** Entropy is reported in nats (natural log). Ranking is
  invariant to a change of base.
- **Historical source rule.** A pair is an eligible historical source for fold
  F only when its event is not held out by F, S and Y are on the official
  four-class scale, R=1, V=1, the footprint match is accepted, and S was
  produced by a head that held that event out. In-sample reports are
  inventoried and flagged, but they do not authorize a historically trained Q.
- **Z schema.** Z is a mapping of covariates available before the survey is
  inspected. The protocol does not freeze a field list, so the contract keeps Z
  as an open mapping.
- **Palu and other unmapped grades.** Tokens such as `unknown`, `unclassified`,
  `ds1`, and `-1` are not four-class eligible. Palu's mapping remains unfrozen
  in the protocol.

## Official recipe (unchanged)

DINOv2 ViT-B, leave-one-event folds, image size 518, no positional-embedding
interpolation, frozen backbone, damage head only, batch 32, bf16,
`num_workers=16` as stated in the training pin. This worktree does not edit
that recipe or the official training trees.
