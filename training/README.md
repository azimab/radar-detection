# Training and evaluation

Setup and run commands are in [../docs/USAGE.md](../docs/USAGE.md) (sections 1, 5 and 6).
This file explains what is in this folder.

The model code is not in this repo. It is upstream RadarPillar plus our changes:

| piece | where |
|---|---|
| RadarPillar | https://github.com/fthbng77/RadarPillar, commit `eb52eb0` (2026-06-21), cloned to `vendor/RadarPillar` |
| our changes to it | `radarpillar_patch/radarpillar_modifications.patch` (15 files, `git apply`; applies cleanly to a fresh clone, checked 2026-11-02) |
| a new file it needs | `radarpillar_patch/files/setup_iou3d_nms.py` (builds only the IoU/NMS CUDA op) |
| radar dataset, augmentation, evaluator | `openpcdet_extension/radar/`, copied in by `install_openpcdet_extension.py` |
| dataset and model configs | `openpcdet_extension/cfgs/{datasets,models}/`, copied in by the same script |

`radarpillar_patch/deleted_files.txt` lists 19 upstream KITTI files that were deleted in the
checkout the results came from. The radar dataset doesn't need them and the patch doesn't
delete them.

## Configs

There is one current recipe:

| model cfg (`tools/cfgs/rear_models/`) | dataset cfg (`tools/cfgs/dataset_configs/`) | |
|---|---|---|
| `rear_radarpillar` | `rear_radar_dataset_base` | 4 stacked scans + Doppler-consistent augmentation; identical to v5 |

The dataset is **not** baked into the config. The training and eval scripts always pass
`--set DATA_CONFIG.DATA_PATH <build>`: by default the build `config/pipeline.yaml` names (for
training) or the build a run trained on (for eval, from its `dataset.txt`). A new build
therefore needs no new config file. Add a config only to change the model, schedule,
augmentation or splits. Put it in `cfgs/models/` or `cfgs/datasets/`, then rerun
`install_openpcdet_extension.py`.

## Scripts

| file | what |
|---|---|
| `train_rear_radarpillar.slurm` → `run_rear_radarpillar_native.sh` | resolves the build, checks its `validation_report.json` and the GPU, builds the CUDA op, trains; records the build in `<run>/dataset.txt` |
| `eval_rear_radarpillar.slurm` | scores every checkpoint, then runs `tools_eval/summarize_eval.py` |
| `tools_eval/summarize_eval.py` | best epoch, smoothed best, score-threshold sweep |
| `tools_eval/average_checkpoints.py` | averages the weights of the smoothed-best epoch window |
| `tools_eval/verify_rotate_iou.py` | checks the rotated-IoU fallback kernel; run by the eval script |

Single-epoch val AP swings by ~4.5 AP, so pick a checkpoint by the smoothed best
(`summarize_eval.py`), not the single best epoch.

To score a run on its own training frames (a sanity probe), pass `train` as the eval
script's `[split]`.

On the portal nodes (no pip), pure-Python training dependencies live in
`vendor/RadarPillar/.deps/native`; the Slurm scripts put it on `PYTHONPATH`.
