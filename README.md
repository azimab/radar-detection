# Radar Car Detection for Autonomous Racing

This project detects the car behind our AV-24 in the Indy Autonomous Challenge, using only
a rear 4D imaging radar. Wheel-to-wheel racing needs that position for defending, merging
back onto the racing line, and being overtaken safely.

![Live rear-radar detection at Laguna Seca](docs/demo.gif)

*Left: rear radar returns with the detector's box and score. Right: the rear camera, for
reference only (the model never sees it). Laguna Seca, a session the model was not trained
on, at 4× speed.*


The rear of the car is covered by a ZF FRGen21 radar, which gives x, y, z and Doppler
velocity at 16 Hz. A car 40 m back may return only a handful of points among far more
from walls and barriers, the elevation is noisy, and above about 20 m/s the velocity
reading folds over, so static clutter can look like a moving car. There is also no labelled
radar data for race cars, so every label had to be generated.

## How it works

**Step 1: Auto-labelling.** Both cars record their own GNSS/INS odometry.
[`build_nuscenes_dataset.py`](build_nuscenes_dataset.py) aligns the two cars' clocks and
map frames, places a box at the opponent's pose, corrects Doppler for ego motion, and
keeps a frame as a positive only when moving returns fall inside the box. Quality gates
drop misaligned labels, and the output is a nuScenes-format dataset that is checked
end to end with the official devkit.

**Step 2: Model.** [RadarPillar](https://github.com/fthbng77/RadarPillar), a PointPillars variant
for 4D radar, trained on four stacked scans. The opponent stays aligned across scans
while clutter smears out. Augmentations re-project the Doppler velocity so they stay
physically consistent.

**Step 3: Live node.** [`live/`](live) runs the trained model as a ROS 2 node on a laptop CPU in
40–55 ms per scan, inside the radar's 62 ms frame time. Its detections feed the car's
Kalman tracker.

## Results

On unseen data from the training track (Laguna Seca), the detector scores **0.92 F1 with
a 1 m centre match**. Training used 13.9k samples from multiple sessions at the track with different opponents.

## Repository layout

| path | what it is |
|---|---|
| `build_nuscenes_dataset.py` | decode bags, auto-label, write the nuScenes dataset |
| `validate_nuscenes_dataset.py` | end-to-end checks through the nuScenes devkit |
| `visualize_nuscenes.py` | bird's-eye-view images of labels for QA |
| `av24_geometry.py` | car and sensor geometry from the AV-24 URDF |
| `config/` | pipeline policy and per-session calibration |
| `training/` | RadarPillar patch, radar dataset, augmentation, evaluator, Slurm jobs |
| `live/` | real-time ROS 2 detector node and model export |
| `third_party/` | vendored car URDF and shared odometry/geometry code |

Build, training and deployment commands are in [docs/USAGE.md](docs/USAGE.md).

**Stack:** Python, PyTorch, OpenPCDet, ROS 2, nuScenes devkit, Slurm.
