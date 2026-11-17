# Event folds for the satellite damage head

The satellite damage head is DINOv2 ViT-B/14, seed 1. A fold holds one reference event out of that head. Images are matched by the exact xBD `metadata.disaster` string, not by a fuzzy filename. The fold list is `configs/folds/event_folds.yaml`. Routing reads the list. Adding a fold does not require a change to the routing code.

For one fold head, every image of the held-out event is removed from official train, test, and hold. The count of that disaster name in each of those splits is asserted to be 0. Those images are a separate eval set (`fold-eval`). They are not train, val, or that head's in-fold test/hold. This repo calls the hold split `holdout`. Point `data.split_dirs.holdout` at `hold` when the release uses that directory name.

`minus-hurricane-michael` is the main seed-1 fold. It runs on WSB GPU 1. `minus-hurricane-michael-seed2` is a stability repeat of that hold-out at seed 2 and runs on WSB GPU 0. The seed is a config entry, not a CLI override. Its role is `seed-repeat` and `claim_evidence` is false, so it does not define \(U\). Checkpoints go to `/data/repo/outputs/minus-hurricane-michael-seed2`. Default routing sends a Michael building only to the seed-1 head. Train and score the seed-2 repeat by name with `--fold minus-hurricane-michael-seed2`. Heads are not averaged.

`minus-palu-tsunami` is the second reference fold and is enabled. `claim_evidence: true` on the Palu fold only means this head is usable for later scoring. Palu's mapping is not frozen and its four diagnostic gates have not passed, so Palu U must not be computed yet and Palu results are not Claim-2 evidence.

`minus-santa-rosa` (`santa-rosa-wildfire`) and `minus-socal` (`socal-fire`) are pipeline checks. Santa Rosa and Woolsey/socal are a wildfire pipeline check, not Claim-2 evidence. A head that has seen an event must not produce the satellite grade for buildings in that event.

No-fold training is the full official train split. Pass `--fold` only for a fold run.

Every fold uses the same training pin. The GPU name does not change it.

- batch size: 32
- AMP dtype: `bfloat16`
- torch is not pinned. WSA and WSB may use different builds. Each run writes `run_env.json` with Python, hostname, torch, CUDA (`torch.version.cuda`), GPU names, and the NVIDIA driver.

Host paths are per machine. `hold` is the on-disk hold directory on both.

| Host | GPU | Fold | Data | Weights | Outputs and checkpoints |
| --- | --- | --- | --- | --- | --- |
| WSB | RTX 4090 | `minus-hurricane-michael` (GPU 1, seed 1), `minus-hurricane-michael-seed2` (GPU 0, seed 2), pipeline checks | `/data/repo/datasets/xbd` | `/data/repo/weights/dinov2_vitb14/dinov2_vitb14_timm_lvd142m.pth` | `/data/repo/outputs/<fold>` |
| WSA | RTX 5880 | `minus-palu-tsunami` | `/mnt/data_hdd/datasets/xbd` | `/mnt/data_hdd/weights/dinov2_vitb14/dinov2_vitb14_timm_lvd142m.pth` | `/mnt/data_hdd/outputs/minus-palu-tsunami` |

The WSA root disk is full, so the Palu fold does not write data, weights, logs, or checkpoints outside `/mnt/data_hdd`. Both weight files are the same timm conversion and the same sha256.

Training, scoring, and conflict-head evaluation of one fold must see the same host, torch, CUDA, NVIDIA driver, and GPU names as that fold's `run_env.json`. A mismatch stops the score. `--cross-env-check` with a separate `--set output_dir=...` is only for measuring whether the difference changes \(\hat{Y}^{s}\). `cross-env-agree` compares the two `predictions_*.json` files. Agreement at or above 99% counts as no effect. Below 99%, Palu-fold results are annotated with the version difference. Cross-fold numbers are reported per event, not pooled.

The loaded file is the timm conversion `dinov2_vitb14_timm_lvd142m.pth`, 346,385,151 bytes, sha256 `6cce660bf1cdf01bc70ad96d53c3ad4d7d02fd4211a669b04abba9aaa848e66c`. It was converted from the Facebook file `dinov2_vitb14_pretrain.pth`, 346,378,731 bytes, sha256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`. Source: timm `timm/vit_base_patch14_dinov2.lvd142m`, Hugging Face revision `4685c99dabffe5affac90bd99dbffd25801ae58d`, upstream `https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth`. At run start the process hashes the file it loads and stops if the digest differs. `pretrained` is false when `weights_path` is set.

## Train and evaluate

From the repository root, with xBD mounted outside the work tree:

```bash
# WSB GPU 1, seed 1. Paths come from the fold file.
python -m mvbridge train-xbd --fold minus-hurricane-michael

# WSB GPU 0, seed 2. Same hold-out, separate output directory.
python -m mvbridge train-xbd --fold minus-hurricane-michael-seed2

# WSA (RTX 5880). Data, weights, and checkpoints stay under /mnt/data_hdd.
python -m mvbridge train-xbd --fold minus-palu-tsunami

python -m mvbridge eval-xbd \
  --fold minus-hurricane-michael \
  --split fold-eval \
  --checkpoint /data/repo/outputs/minus-hurricane-michael/checkpoint_last.pt

python -m mvbridge oof-route --disaster hurricane-michael
python -m mvbridge oof-route --disaster palu-tsunami
```

`oof-route` prints the one head that held that event out. It refuses when no such head exists, and it refuses to average when more than one head held the event out. `--claim-only` is the default, so pipeline-check folds are not used for conflict labels. `--no-claim-only` includes them.

At training start the run writes `output_dir/fold_<name>_manifest.json` and logs its sha256. The file lists, per official split, the image ids kept and the image ids held out, plus per-event counts. The sha256 is of the canonical JSON (sorted keys, compact separators, UTF-8) with the `sha256` field removed. The same root and fold write the same bytes.

`--split test` or `--split holdout` with `--fold` is the in-fold split: the held-out event is absent. Out-of-fold scores for that event use `--split fold-eval` and the matching checkpoint. The checkpoint's recorded fold must match `--fold`.

## Add a fold

Append an entry to the `folds:` list. Set `name`, `holdout_events` (exact `metadata.disaster` strings), `role`, `enabled`, `claim_evidence`, and `seed`. Do not add the name to the router. A third reference fold such as `minus-hurricane-florence` with `holdout_events: [hurricane-florence]`, `enabled: true`, and `claim_evidence: true` is then trainable and is the only claim head that may score Florence buildings.

## Hashes

Manifest sha256 is filled from the training log after a real xBD root is read. Checkpoint sha256 is `sha256sum` of that run's checkpoint file. Both stay TBD until then. Do not copy a hash from a fixture, a dry run, or another machine.

| Fold | Role | Held-out `metadata.disaster` | Seed | Manifest sha256 | Checkpoint sha256 |
| --- | --- | --- | --- | --- | --- |
| `minus-hurricane-michael` | main | `hurricane-michael` | 1 | TBD | TBD |
| `minus-hurricane-michael-seed2` | seed-repeat | `hurricane-michael` | 2 | TBD | TBD |
| `minus-palu-tsunami` | secondary | `palu-tsunami` | 1 | TBD | TBD |
| `minus-santa-rosa` | pipeline-check | `santa-rosa-wildfire` | 1 | TBD | TBD |
| `minus-socal` | pipeline-check | `socal-fire` | 1 | TBD | TBD |
