# Usage

How to build the dataset, train, evaluate and run the detector. Paths are relative to the repository root.

## 1. Set up (once)

```bash
git clone <this repo> radar_openpcdet && cd radar_openpcdet && mkdir -p logs training/logs
pip install -r requirements.txt
export AV24_DATA=<root of the recorded bags>          # used by config/runs/*.yaml
export LIDAR_OPENPCDET=<lidar OpenPCDet checkout>     # source of the shared odometry caches
python3 seed_odom_cache.py
```

No `pip` on the portal nodes? See [../vendor/README.md](../vendor/README.md).

To train, also set up RadarPillar from `training/`:

```bash
cd training
git clone https://github.com/fthbng77/RadarPillar.git vendor/RadarPillar && git -C vendor/RadarPillar checkout eb52eb0
git -C vendor/RadarPillar apply ../../radarpillar_patch/radarpillar_modifications.patch
cp radarpillar_patch/files/setup_iou3d_nms.py vendor/RadarPillar/
python3 install_openpcdet_extension.py
pip install -r requirements.txt
```

Rerun `install_openpcdet_extension.py` whenever you change `training/openpcdet_extension/`.

## 2. Add a recording

1. `cp config/runs/_template.yaml config/runs/<venue>_<yyyy>_<mm>_<dd>_rear.yaml` and fill it in
   (file name must equal `name:`).
2. Add the name to the end of `runs:` in `config/pipeline.yaml` and bump `dataset.output_root`.
3. Decode it (once per run): `sbatch build_nuscenes_dataset.slurm <run>`

## 3. Build and check

```bash
python3 build_nuscenes_dataset.py label --overwrite   # ~10 min, no bag reads
python3 validate_nuscenes_dataset.py                  # must end in PASSED
python3 visualize_nuscenes.py --per-run 6 --qa-dropped 4   # BEV images in <build>/viz/
```

## 4. Calibrate (if labels look off)

Edit `config/runs/<run>.yaml` under `# --- calibration`, then rerun step 3.

| symptom | change |
|---|---|
| near-edge gate fails / `shortfall_m` far from 0 | `box_center_forward_offset_m` ≈ old + `shortfall_m` |
| box lag scales with speed | `build_nuscenes_dataset.py sweep-clock --run <run> --sweep -0.2 0.2 0.01`, then `radar_time_offset_seconds` |
| stamps wildly off log time | `time_source: log` |
| constant metres-scale shift | `multitrack.opponents[].map_offset_xyz` |
| rules for every run (Doppler, min points, QA gates) | `config/pipeline.yaml` |

## 5. Train

From `training/` (args: `<run-tag> <epochs> <batch-size> [seed] [model-cfg] [dataset]`):

```bash
for seed in 1 2 3; do sbatch train_rear_radarpillar.slurm rear_radarpillar_v6 35 4 $seed; done
```

Use a new run tag per build. Defaults to the build named in `config/pipeline.yaml`.

## 6. Evaluate

From `training/` (args: `<run-tag-with-seed> <batch-size> [model-cfg] [split] [dataset]`):

```bash
sbatch eval_rear_radarpillar.slurm rear_radarpillar_v6_seed1 4
```

More tools: [../training/README.md](../training/README.md).

## 7. Run live

See [../live/README.md](../live/README.md).
