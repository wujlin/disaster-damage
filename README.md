# MVBridge

Week-1 scaffold for a method paper: adapt a DINO-style backbone on multi-view disaster imagery (SAGINDisaster / SAGINGeo), then transfer to building damage on the official xBD / xView2 splits.

Two pathways stay separate. The identity/geometry pathway is where a later multi-view invariance loss will attach. The damage-state pathway reads detached features and owns the damage head, so that invariance objective cannot wash out damage cues. Simulated UAV imagery is a viewpoint bridge. It is not damage evidence.

This repository does not download datasets or checkpoints, and it does not start training on import. The ViT-B entry point is a frozen DINOv2 probe with a small damage head. There is no ViT-L setup.

[Evaluation protocol](docs/eval_protocol.md). [Event folds](docs/folds.md).
## Layout

```
configs/overlap_audit.yaml      event buffer, date window, data roots
configs/event_catalog.yaml      date-specific event ids and known name collisions
configs/sagin_loader.yaml       place-grouped SAGIN loader
configs/xbd_dinov2_vitb.yaml    official-split ViT-B baseline
configs/folds/event_folds.yaml  leave-one-reference-event-out folds (seed 1)
src/mvbridge/                   package
scripts/                        thin wrappers around the same entry points
tests/                          fake metadata, fake xBD labels, pathway checks
```

Install and run the tests from the repository root:

```bash
pip install torch torchvision   # CPU or CUDA build for your machine
pip install -e ".[dev]"
pytest -q
```

`pytest` does not need SAGIN or xBD on disk. It builds a random-weight DINOv2 ViT-B once and checks that a damage loss does not update the identity pathway.

## Overlap rule

The audit is reproducible from `configs/overlap_audit.yaml` and `configs/event_catalog.yaml`. Every run prints the rule with the counts.

Event overlap requires both records to resolve to the same catalog id. Ids are date-specific. Midwest flooding 2015 (a SAGIN example) and the xBD Midwest floods of 3 Jan–31 May 2019 do not share an id. A bare shared word such as "flood" or "fire" is not a match. Names listed as ambiguous, and names that hit more than one id, are reported and are not exclusions. If a SAGIN row has a date and the catalog event has a range, the spans must meet after `date_window_days` (default 14). A date outside that window is a date disagreement, not an exclusion.

Spatial overlap is independent of the name. A SAGIN coordinate hits an xBD image when the haversine distance to the nearest building centroid is `<= buffer_m` (default 1000). The centroid is the mean of that building's `lng_lat` WKT vertices. Candidates are images whose centroid bounding box, expanded by the buffer, contains the point. The buffer is an axis-aligned pad in degrees, then a true haversine check. This uses annotated buildings, so a SAGIN point inside a tile but far from every labeled building is not a hit.

Default recommendation, before any continued pretraining: drop the SAGIN groups in `exclude_pretrain_sagin_groups.txt` and leave the official xBD test and holdout splits unchanged. Those groups are the ones that meet **test or holdout** geography or share an event id with those splits. Train-only geography is listed in `train_split_spatial_hits.txt` and is not removed by that default.

Alternative, if those places already entered pretraining: drop the image ids in `alternative_exclude_downstream_xbd_images.txt` from evaluation. That score is no longer the full official test split, and the paper should say so. Point the xBD config `data.exclude_ids_file` at that list. The loader will not invent a replacement split.

Reviewed name collisions (California 2025 vs Santa Rosa 2017 / Carr 2018 / Woolsey 2018, Illinois 2015 vs Joplin / Moore / Tuscaloosa, Louisiana 2016 vs Harvey and Matthew, plus Maui 2023, Nashville 2020, and Ian 2022) are printed on every report and are not exclusions. Unmapped disaster strings from your files are listed too, so a local filename slug can be added to the catalog.

### Source notes

Checked against the papers, not treated as a complete inventory of a local dump:

- SAGINGeo (Yin et al., IEEE JSTARS, [doi:10.1109/JSTARS.2026.3712316](https://doi.org/10.1109/jstars.2026.3712316)): 2080 groups, 118560 images, 2080 simulated UAV videos. Views are VGI, panoramic street view, UAV, and remote sensing. The abstract says the code and dataset would be released on acceptance.
- ProbGLC (Li et al., ISPRS 2026, [github.com/bobleegogogo/ProbGLC](https://github.com/bobleegogogo/ProbGLC)): distributes a multidisaster archive and describes SAGINDisaster as drawing on more than 50 NOAA events. The named examples are California fire (2025), Maui fire (2023), Hurricane Ian, Nashville tornadoes (2020), Illinois tornadoes (2015), Louisiana flooding (2016), and Midwest U.S. flooding (2015). The same section dates Ian's landfall to 28 September 2022 and, in one sentence, writes "IAN (2025)". This repo uses 2022. ProbGLC's loader pairs one RSI, SVI, UAV, and VGI path per row (`plonk/data/data.py`, `MultidisasterEarth20` / `30`). That row count and the 118560-image figure are not the same description; video frames may make up the difference. Week 1 does not decode video.
- xBD (Gupta et al., [arXiv:1911.09296](https://arxiv.org/abs/1911.09296), Table 1 and Table 2): 19 events, official train/test/holdout = 18336 / 1866 / 1866 images (632228 / 109724 / 108784 polygons). Damage classes are no-damage, minor-damage, major-damage, and destroyed. xView2 also shipped a separate tier3 archive and, during the challenge, a test set without labels. This code reads the directories you point at. It does not merge tier3, and it does not score unlabeled images.

## Expected data layout

Keep data outside the git work tree. Config roots are null until you set them.

Manifest CSV (one image per row; several rows share a `group_id`):

```text
group_id,event_name,event_date,lat,lon,view,path,is_simulated
maui-0001,Maui fire 2023,2023,20.88,-156.68,vgi,images/maui-0001/vgi.jpg,false
maui-0001,Maui fire 2023,2023,20.88,-156.68,svi,images/maui-0001/svi.jpg,false
maui-0001,Maui fire 2023,2023,20.88,-156.68,uav,images/maui-0001/uav.jpg,true
maui-0001,Maui fire 2023,2023,20.88,-156.68,rsi,images/maui-0001/rsi.jpg,false
```

`view` is `vgi`, `svi`, `uav`, or `rsi`. Paths are relative to `root` unless absolute. `is_simulated=true` marks a viewpoint bridge. For this dataset the UAV column defaults to simulated when the cell is empty (`uav_default_is_simulated: true`). Set `is_simulated` false only for a real UAV frame that is allowed as damage evidence.

ProbGLC / MultiDisaster root, as read by their `MultidisasterEarth` classes:

```text
MultiDisaster_20250320/
  RSI_SVI_UAV_VGI/
    80%train_RSI_SVI_UAV_VGI.csv
    20%test_RSI_SVI_UAV_VGI.csv
    70%train_RSI_SVI_UAV_VGI.csv
    30%test_RSI_SVI_UAV_VGI.csv
  Other/
    allData_20250320.xlsx          # newnewID, Latitude, Longitude
  <image paths stored in the CSVs, often with Windows separators>
```

Each CSV has no header and four columns: RSI, SVI, UAV, VGI. The place id is the VGI filename stem (`1636.jpg` -> `1636`), joined to `newnewID`. `which_tables: all` concatenates whichever of those CSVs exist and drops duplicate rows. Those 80/20 and 70/30 files are ProbGLC geolocalization splits. They are not an xBD split, and this loader does not use them as one. An `event_name` or `event_date` column in the workbook is picked up when present. The published workbook is not guaranteed to carry event names; the spatial audit still runs, and the report says how many groups had no event string.

xBD / xView2, official folders only:

```text
xBD/
  train/
    images/   hurricane-harvey_00000000_pre_disaster.png
              hurricane-harvey_00000000_post_disaster.png
    labels/   hurricane-harvey_00000000_pre_disaster.json
              hurricane-harvey_00000000_post_disaster.json
    targets/  raster masks, not read by this baseline
  test/
    images/
    labels/
  holdout/    # some releases use hold/; set split_dirs.holdout
    images/
    labels/
```

A directory named `val/` or any other custom split is ignored. Train, test, and holdout stay disjoint; an image id that appears in two of them is an error. If your holdout folder is `hold`, set `split_dirs.holdout` to `hold`. If `labels/` is missing, evaluation stops. The challenge test release sometimes shipped images without JSON, and this scaffold will not invent labels or a new split to replace them.

## Commands

Demo mode uses a tiny fixture under the output directory so you can see the report before the real roots are mounted. It does not download anything.

```bash
python -m mvbridge audit --demo
python -m mvbridge sagin-smoke --demo
python -m mvbridge train-xbd --dry-run
```

The same commands exist as `scripts/audit_overlap.py`, `scripts/sagin_smoke.py`, `scripts/train_xbd_baseline.py`, and `scripts/eval_xbd_baseline.py`.

Real overlap audit:

```bash
python -m mvbridge audit \
  --set sagin.format=probglc \
  --set sagin.probglc_root=/data/MultiDisaster_20250320 \
  --set xbd.root=/data/xBD \
  --set output_dir=/data/work/overlap_audit
```

Or, with a manifest:

```bash
python -m mvbridge audit \
  --set sagin.format=manifest \
  --set sagin.manifest_csv=/data/sagin/manifest.csv \
  --set sagin.root=/data/sagin \
  --set xbd.root=/data/xBD
```

The process writes `overlap_report.txt`, `overlap_report.json`, and the exclusion lists. Exit code 0 means the audit finished. Overlaps are reported in the files, not by a nonzero exit. Exit code 2 means a root or metadata file was missing; the message includes the path.

SAGIN loader smoke test on real metadata:

```bash
python -m mvbridge sagin-smoke \
  --config configs/sagin_loader.yaml \
  --set format=probglc \
  --set probglc_root=/data/MultiDisaster_20250320 \
  --set check_files=true
```

Each dataset index is one place. `__getitem__` returns one image per available view. `identity_views` are the DINO positives and may include a simulated UAV still. `damage_evidence_views` never includes a simulated or viewpoint-bridge asset. `missing_view_policy` is `allow_partial` (keep the place when at least `min_views` images exist), `skip_group` (drop a place that lacks any of the four views), or `error`. UAV videos are recorded on `bridge_videos` and are not decoded in this scaffold. To drop audit hits from a later pretraining run, set `exclude_ids_file` to `exclude_pretrain_sagin_groups.txt`.

xBD ViT-B baseline. This command trains. It reads only `split_dirs.train`. With `backbone.pretrained: true` and no `weights_path`, timm fetches DINOv2 ViT-B into the torch cache outside this repo.

```bash
python -m mvbridge train-xbd \
  --set data.root=/data/xBD \
  --set output_dir=/data/work/xbd_dinov2_vitb \
  --set batch_size=32 \
  --set backbone.image_size=224 \
  --set data.image_size=224
```

`--max-steps 20` stops after a few updates for a debugging pass. Logs go to stdout and `output_dir/run.log`: resolved config, device, class histogram, first image ids, first-batch shapes, per-step loss, and a checkpoint that stores the damage pathway and the head, not a second copy of the backbone. Identity parameters stay frozen (`identity_trainable=0`). The optimizer lists only the damage pathway and the damage head.

Leave-one-reference-event-out training passes `--fold`. The default command above does not, and still uses the full official train split. The main seed-1 fold holds out `hurricane-michael`. `minus-hurricane-michael-seed2` is that same hold-out at seed 2. `minus-palu-tsunami` is the second reference fold. `claim_evidence: true` on Palu only means the head can be used for later scoring. Palu U is not computed yet, and Palu results are not Claim-2 evidence. `minus-santa-rosa` and `minus-socal` are pipeline checks, not Claim-2 evidence. See [docs/folds.md](docs/folds.md).

```bash
python -m mvbridge train-xbd --fold minus-hurricane-michael
```

On WSB the Michael fold reads `/data/repo/datasets/xbd` and `/data/repo/weights/dinov2_vitb14/dinov2_vitb14_timm_lvd142m.pth`, and writes `/data/repo/outputs/minus-hurricane-michael`. On WSA the Palu fold uses `/mnt/data_hdd/datasets/xbd`, `/mnt/data_hdd/weights/dinov2_vitb14/dinov2_vitb14_timm_lvd142m.pth`, and `/mnt/data_hdd/outputs/minus-palu-tsunami`. Both folds pin batch size 32, AMP `bfloat16`, and torch `2.15.0`. The run logs the real torch, CUDA, and GPU name and stops if the torch release differs. It does not change the pin for the 4090 or the 5880. `python -m mvbridge oof-route --disaster hurricane-michael` prints the head that may score Michael buildings. It does not average heads.

Image size must be a multiple of 14. The default 224 is a short-side probe; 518 matches the DINOv2 training resolution and is a config change, not a different backbone. `vit_large_*` is rejected.

Evaluation without a fold accepts only `test` or `holdout`:

```bash
python -m mvbridge eval-xbd \
  --split test \
  --checkpoint /data/work/xbd_dinov2_vitb/checkpoint_last.pt \
  --set data.root=/data/xBD \
  --set output_dir=/data/work/xbd_dinov2_vitb
```

It writes `metrics_test.json` with accuracy, per-class precision/recall/F1, weighted F1, and the confusion matrix. Weighted F1 is the summary used in the xBD paper. The baseline itself trains with unweighted cross-entropy and logs the class histogram, because the label distribution is heavily no-damage.

With `--fold`, `--split fold-eval` scores only the images that fold removed from train, test, and hold. `--split test` or `--split holdout` with `--fold` is the in-fold split, and the held-out event is absent. The checkpoint's recorded fold must match `--fold`.

The supervised task is building-level classification. Each post-disaster polygon with subtype `no-damage`, `minor-damage`, `major-damage`, or `destroyed` becomes one example. `un-classified` polygons and non-building features are skipped. Crops are square boxes around the `xy` polygon, padded by `crop_padding`, taken from the pre and post images of that official pair. The damage pathway projects the two frozen ViT-B class tokens separately, forms `post_proj - pre_proj`, and the head reads that residual together with the post projection. Detached patch tokens are on `DamageState` so a later change-detection head can replace `DamageHead` without taking the identity parameters.

```text
encode_identity(image)              future multi-view loss, not used by the damage objective
damage_state(pre, post)             detached features, residual, patch tokens
DamageHead.forward(DamageState)     logits over the four damage classes
```

## Development

Keep `main` as the clean scaffold. Use one git worktree and one branch per experiment so parallel runs do not share a working tree:

```bash
git worktree add -b cursor/exp-dinov2-vitb-42b8 ../mvbridge-vitb main
cd ../mvbridge-vitb
```

Point every config at data and output directories outside the worktree. Do not commit raw imagery, full metadata dumps, or checkpoints. Merge code back when the experiment is ready to keep; leave `main` free of trial configs and large artifacts.
