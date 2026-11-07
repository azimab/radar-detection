# Status and results

As of 2026-11-02. How to reproduce everything here: [README.md](../README.md). Model cfg names below are archived in [configs_archive/](configs_archive/); the current recipe (v5) is `rear_radarpillar`.

## Summary

- The **best model so far is v4** (two Laguna Seca sessions for training, 4 stacked scans,
  Doppler-consistent augmentation). Its best centre-distance AP on Caltech val (`val_qa`) is
  **77.4–78.0** over three seeds.
- **Nothing has been scored on the held-out IU test split yet.**
- **v3 and v4 were trained on builds with a known label leak** in their 4-scan stacks (see
  [Known issues](#known-issues)). The fixed build (`data_nuscenes_2026_10_02`) exists and
  validates, and v5 (the v4 recipe on the 2026-11-02 rebuild of it) is training. Read the v3/v4 numbers as likely optimistic
  until it is.

## Model versions

| version | model cfg | dataset build | train | eval split | input |
|---|---|---|---|---|---|
| v1 | `rear_radarpillar_car` | `radar_kitti_rear`, an older KITTI-format build (retired) | — | — | 1 scan |
| v2 | `rear_radarpillar_car` | `data_nuscenes` (2026-09-21) | Laguna 07-26 | val | 1 scan |
| v3 | `rear_radarpillar_car_v3` | `data_nuscenes` | Laguna 07-26 | val | 4 scans + augmentation |
| v3 Laguna holdout | `rear_radarpillar_car_v3_lagunaholdout` | `data_nuscenes` | `laguna_train` | `laguna_holdout` | as v3 |
| v4 | `rear_radarpillar_car_v4` | `data_nuscenes_2026_09_28` | Laguna 07-25 + 07-26 | `val_qa` | as v3 |
| v5 | `rear_radarpillar_car_v5` | `data_nuscenes_2026_11_02` | Laguna 07-25 + 07-26 | `val_qa` | as v3 (training, 3 seeds) |

v3 and v4 share one model and schedule (35 epochs); only the data differs.

## Results

Metric: **AP_R40 with a 1 m centre-distance match** (`Car_center_AP_R40@1.0m`). The consumer is a
Kalman tracker that uses box centres. BEV IoU 0.5 AP is shown too; it is harsh on a 4.9 m box
whose error is mostly along-track. "Best" is the best single epoch; "smoothed" is the best
3-epoch mean, which is the fairer number because single-epoch val AP swings by ~4.5 AP.

| run | eval split | best epoch | centre AP @1 m | smoothed | BEV AP @IoU 0.5 |
|---|---|---:|---:|---:|---:|
| v2 | val | 27 | — | — | 35.1 |
| v2 | val_qa | 1 | 60.5 | 52.3 | 41.8 |
| v3 seed 1 | val | 18 | 58.6 | 55.3 | 42.9 |
| v3 seed 2 | val | 4 | 57.6 | 54.0 | 40.6 |
| v3 seed 3 | val | 15 | 53.8 | 51.5 | 40.1 |
| v3 seed 1 | val_qa | 23 | 69.6 | 66.4 | 50.3 |
| v3 seed 2 | val_qa | 4 | 70.4 | 67.1 | 49.1 |
| v3 seed 3 | val_qa | 15 | 67.2 | 63.9 | 50.5 |
| **v4 seed 1** | **val_qa** | 14 | **78.0** | 75.6 | 57.7 |
| **v4 seed 2** | **val_qa** | 12 | **77.4** | 72.9 | 53.3 |
| **v4 seed 3** | **val_qa** | 17 | **78.0** | 71.8 | 56.4 |
| v3 seed 1 | laguna_0725 | 10 | 92.7 | 91.5 | 85.7 |
| v3 seed 2 | laguna_0725 | 24 | 91.8 | 91.6 | 86.1 |
| v3 seed 3 | laguna_0725 | 18 | 91.7 | 91.5 | 86.2 |
| v3 Laguna holdout | laguna_holdout | 31 | 92.9 | 92.7 | 76.8 |

- **val** = Caltech at Putnam Park (22 scenes). **val_qa** = val without scene `rear-013`,
  whose labels sit ~1.3 m along-track off the car. **laguna_0725** = the 07-25 Laguna session,
  which v3 never trained on. **laguna_holdout** = 16 scenes of 07-26 held out of training.
- Same-track scores (~92) are far above cross-track scores (~70–78): **transfer to a new track
  is the open problem**, not fitting.
- v1 was scored with the older 3D-IoU metric on the retired KITTI-format build and isn't
  comparable. A train-split probe of v1 (scored on its own training frames, 2026-08-19)
  reached 100, which showed the data path and evaluator were sound at the time.
  `eval_trainsplit_probe.slurm` reruns that check.

Source summaries: `training/vendor/RadarPillar/output/rear_models/<cfg>/<run>/eval/eval_all_default/*/summary_*.json`
(not in git). Slide figures for Laguna 07-25: `notes/scripts/plot_laguna_0725.py`.

## Dataset

Current build: **`data_nuscenes_2026_11_02`**, validated with the official nuScenes devkit (passed, 0 errors).
It is the first build made from this repository, and it reproduces `data_nuscenes_2026_10_02` exactly:
the same splits, identical nuScenes tables (annotations, poses, sample data) and byte-identical point files.

| split | nuScenes version | run (opponent) | scenes | samples | with a car |
|---|---|---|---:|---:|---:|
| train | `v1.0-trainval` | Laguna Seca 07-26 (Purdue) + 07-25 (IU) | 136 | 13,869 | 11,096 |
| val | `v1.0-trainval` | Putnam Park 05-05 (Caltech) | 22 | 2,587 | 2,070 |
| test | `v1.0-test` | Putnam Park 04-15 (IU) | 43 | 456 | 365 |

Splits are run-blocked (each split is a different session) and 20% of each split is background.
Extra `splits.json` keys: `val_qa`, `laguna_train`, `laguna_holdout`, `laguna_0725`.

| build | train runs | sweeps | used by |
|---|---|---|---|
| `data_nuscenes` (2026-09-21) | Laguna 07-26 | selected scans only (leaky) | v2, v3 |
| `data_nuscenes_2026_09_28` | + Laguna 07-25 | selected scans only (leaky) | v4 |
| `data_nuscenes_2026_10_02` | + Laguna 07-25 | every scan ≤0.5 s before a keyframe | — |
| `data_nuscenes_2026_11_02` | + Laguna 07-25 | as 2026-10-02; first build from this repo | v5 |

### Format

```text
samples/RADAR_REAR/<run>__RADAR_REAR__<us>.pcd          nuScenes 18-field radar PCD, sensor frame
samples/RADAR_REAR_FRGEN21/<run>__RADAR_REAR__<us>.npy  (N, 11) float32 model input, ego frame:
                                                        x y z v_rel v_abs v_abs_x v_abs_y rcs snr confidence power
v1.0-trainval/*.json, v1.0-test/*.json                  nuScenes tables (test includes annotations)
splits.json            scene names per split
manifest.jsonl         every decoded scan and why it was kept or dropped
dataset_report.json    per-run geometry, counts, lateral-QA spans
validation_report.json validator output
viz/                   BEV images; viz/<run>/qa_dropped/ shows what the QA gate removed
```

- **Frames.** The ego frame is `rear_axle_middle`. `ego_pose` is the ego odometry at each
  scan's acquisition time; `calibrated_sensor` is the URDF's `radar_rear` mounting.
  Annotations are global, `size` in nuScenes' `[w, l, h]` order.
- **PCD.** Holds exactly nuScenes' 18 ARS408 fields, so `RadarPointCloud.from_file` loads it.
  FRGen21 has no ARS408 status fields; they hold "unknown / not filtered" (dyn_prop 4,
  ambig_state 3, invalid_state 0). `snr`, `confidence` and `power` exist only in the `.npy`,
  which is what training reads.
- **Not standard nuScenes.** There is no `LIDAR_TOP`, so point the devkit's detection eval at
  `RADAR_REAR`. `visibility_token` is empty and `num_lidar_pts` is 0. Non-key sweeps sit in
  `samples/`, not `sweeps/`.

### How labels are made

- A car is labelled when it is 2–100 m behind, within ±60°, and has ≥2 radar returns in its
  BEV box, at least one moving (|v_abs| ≥ 2 m/s). Boxes holding only stationary clutter are
  ignored. Scans with ego speed below 5 m/s are dropped.
- Box positions come from both cars' odometry (via the lab lidar pipeline's timeline code in
  `third_party/openpcdet_geometry/`), corrected by a measured per-run
  `box_center_forward_offset_m`. Sensor mounting comes from the URDF; the bags disagree about it.
- `v_abs` is ego-motion compensated with `/vehicle/uva_odometry`. FRGen21 Doppler folds every
  40 m/s; above ~20 m/s ego speed (Laguna only) folded stationary returns are unwrapped, and the
  validator fails the build if any remain.
- **Lateral QA gate.** Spans where the box sits ≥0.8 m sideways of the opponent's own moving
  returns (1 s rolling median, ≥1 s long) are dropped. Added for four Caltech episodes where
  the opponent's GPS was ~1 m off.
- Background scans need the whole opponent out of reach; cars just past 100 m are ignored.

### Label quality (validator, after the QA gate)

| run | returns vs box, across (median) | frames >1 m across | near-edge shortfall | URDF window |
|---|---:|---:|---:|---|
| laguna | −0.14 m | 1.7% | −0.04 m | inside |
| caltech | −0.10 m | 0.1% | −0.18 m | inside |
| iu | −0.17 m | 4.4% | −0.20 m | **outside** (warning) |

Lidar can't check these labels: the AV-24's lidars don't see behind the car.

## Known issues

1. **Sweep-stack leak in the builds v2–v4 used.** Those builds wrote only selected scans, so a
   keyframe's "previous scans" chain skipped to the previous *selected* scan. Positives come in
   dense runs and had 3 previous scans 98–99% of the time; sparse background had them only
   4–19% of the time. With 4-scan stacking (v3, v4) the stack depth gives the label away. The
   2026-10-02 build fixes this, and the validator now fails a build where the gap returns
   (`check_depth`, `max_full_stack_gap`). **v3/v4 need retraining on it.**
2. **IU (the test split) carries an unexplained bias**: a range-proportional lateral offset
   (~−1.3° of bearing), and a box offset 0.33 m outside the URDF window. A radar clock correction
   didn't survive a fine sweep. Test numbers will carry this.
3. **Caltech scene `rear-013`** has labels ~1.3 m along-track off for the whole approach. It is
   excluded at eval time (`val_qa`) rather than fixed. The along-track QA gate that would drop
   it is off because it would also drop 637 unreviewed Laguna training frames.
4. **One training track.** Every training scene is Laguna Seca; val and test are Putnam Park.

## Investigation notes

The reasoning behind each measured value (per-run clock sources and offsets, box-origin
re-expression on 2026-09-28, opponent clock offsets, Doppler unwrap) is in
[calibration_history.md](calibration_history.md); the values themselves are in `config/runs/`. Radar clock offsets can be re-scored with:

```bash
python3 build_nuscenes_dataset.py sweep-clock --run iu_2026_04_15_rear --sweep -0.1 0.3 0.01
```
