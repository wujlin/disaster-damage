# Score contract

This document is the schema for `feat/score-contract`. It does not change the
official training recipe, does not start official Michael scoring, and does not
write official output directories or checkpoints.

The score-contract rules below are official protocol text in
`docs/eval_protocol.md`. This file follows that protocol. It does not add a
rule the protocol does not state.

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

These comparisons are stated in `docs/eval_protocol.md` (Baselines, P4, and
P5). This file does not choose a different comparison.

- The P4 main test compares only `entropy` and `one_minus_max_p`.
- `g0` belongs only to P5. Asking P4 to include `g0` is an error.
- This worktree does not implement or fit `g0`.

## Rules taken from the evaluation protocol

`docs/eval_protocol.md` is the official source of these rules.

- **Tie break.** Label section: when two or more classes share the maximum
  probability within absolute tolerance `1e-12`, choose the class with the
  smallest index in the official order (no-damage, minor-damage, major-damage,
  destroyed) and mark `tied=True`.
- **Entropy units.** Notation for \(H\): entropy uses the natural logarithm
  (nats). Ranking is invariant to log base.
- **Historical source rule.** Historical source eligibility: the event is not
  this fold's holdout; S and Y are in the official four classes; R=1; V=1; the
  match was accepted; and S was produced by a head that held out that event.
  In-sample reports are recorded and flagged, but they do not authorize a
  historically trained Q.
- **Z schema.** Notation for \(Z\): Z is currently an open pre-disaster
  covariate mapping. The protocol does not yet freeze a field table.
- **Four-class labels.** Four-class eligibility, and the Palu subsection: the
  labels unknown, unclassified, ds1, and -1 are not four-class eligible. The
  Palu mapping remains unfrozen.

## Official recipe (unchanged)

DINOv2 ViT-B, leave-one-event folds, image size 518, no positional-embedding
interpolation, frozen backbone, damage head only, batch 32, bf16,
`num_workers=16` as stated in the training pin. This worktree does not edit
that recipe or the official training trees.
