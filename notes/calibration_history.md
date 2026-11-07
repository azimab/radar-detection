# Calibration and label-policy history

Every measured value in `config/` was chosen for a reason that was written next to it in the
original `rear_radar_config.yaml`. Those notes are kept here verbatim, grouped by where the
value lives now. Read the run's section before changing any value in `config/runs/<run>.yaml`.

## File header

```text
Rear-radar Car dataset, nuScenes v1.0 format (build_nuscenes_dataset.py).

Run geometry -- ego/opponent odometry, datum offsets, yaw corrections, opponent
clock offsets, degenerate-orientation repair -- is NOT declared here. It lives in
the shared pp_run_config_multitrack.yaml and reaches this pipeline through
build_pp_kitti.run_timelines, so the radar dataset and the lidar dataset cannot
disagree about where a car was. Each run below names the multitrack run it draws
from and adds only what is specific to the rear radar.
```

## `config/pipeline.yaml`: dataset

**`output_root`**

```text
2026-11-02: first build from this repository (vendored URDF), same policy as the
2026-10-02 build: every scan in the sweep window is a non-key sweep (see `sweeps`).
```

**`carry_splits_from`**

```text
Extra splits.json keys (val_qa, laguna_train, laguna_holdout, laguna_0725) are
copied from this file (the 2026-10-02 build's splits.json, kept in the repo so a
fresh clone can build). The build refuses if any named scene is missing or
starts at a different time, i.e. if the scene boundaries have moved.
```

**`scan_cache_root`**

```text
`decode` caches every scan here so label policy can change without re-reading bags.
```

**`scene_max_seconds`**

```text
nuScenes-style scenes: at most 20 s, and a gap longer than this between kept
samples starts a new scene (prev/next means "the neighbouring keyframe").
```

**`sweeps`**

```text
Every scan up to max_lag_s before a keyframe is written as a non-key sample_data of
it, as nuScenes does, so stacking previous scans gives every keyframe its true
history whether or not its neighbours were selected. Must be at least the
training cfgs' MAX_SWEEP_LAG_S (0.5).
```

**`check_depth`**

```text
Validator: keyframes with this many previous scans (NUM_SWEEPS - 1 in the v3/v4
cfgs) must be as common among background as among positives, within the gap.
```

**`sensor_extrinsic_tolerance_m`**

```text
The AV-24 rig, from the URDF the car itself publishes /tf_static from. Sensor
mounting is read from here, not transcribed and not taken from the bags: the three
bags were recorded five months apart and do not agree about the rear radar's
mounting pitch (see sensor_extrinsic_tolerance_m below).
How far a bag's tf_static may put the rear radar from where the URDF puts it before
the build refuses. This is a *translation* tolerance only. A rotation disagreement
is resolved in the URDF's favour and reported; a translation disagreement means the
bag is describing a different rig, which the URDF has no standing to overrule.
```

**`car_size_lwh`**

```text
Must match the anchor_sizes in the model yaml that pp_run_config_multitrack.yaml
points at; the builder asserts this rather than trusting it.
```

**`min_moving_points_in_box`**

```text
...of which at least this many must be moving (|v_abs| >= moving_speed_mps). Added
2026-09-21: without it, 78 train / 55 val / 72 of 437 IU test boxes were positives
only because stationary clutter fell inside a box whose car went undetected. Those
scans become ignored_no_moving_returns (neither positive nor background).
```

**`ego_odom_topic`**

```text
Ego velocity for Doppler compensation and the speed gate: the odometry's own
body-frame twist, not a derivative of its position track.
```

**`doppler_dealias`**

```text
FRGen21 Doppler folds with a 40 m/s period. Found 2026-09-21 on Laguna, the only run
above 20 m/s ego speed (max 25.2): with ego > 20 m/s, 15-35% of returns read
v_abs = -40.00 m/s (p5..p95 -40.49..-39.77, tail to -44 from twist interpolation) -- stationary clutter near boresight
folded from ~+22 to ~-18 m/s v_rel, 3.3% of all Laguna training points. Caltech and
IU (ego <= 15.4 m/s) have none. Measured period 40.0 (aliased cluster -40.00,
stationary cluster -0.1). Only returns that become stationary after one unwrap are
corrected; the validator fails the build if the -40 cluster reappears.
```

**`static_tolerance_mps`**

```text
folded tail reaches -44; the opponent reads ~ -25 at most
```

**`min_ego_speed_mps`**

```text
Scans below this ego speed are dropped (label_state ego_slow). Stopped, static
clutter has ~zero Doppler and the velocity features carry nothing.
```

**`point_cloud_range`**

```text
FRGen21 elevation is much noisier than range/azimuth. Preserve measured z as
a feature, but use one deliberately broad vertical pillar instead of deleting
otherwise valid BEV returns.
```

**`lateral_qa_gate`**

```text
Drop spans where the label box sits persistently sideways of the opponent's own
moving returns. Found 2026-09-21 on Caltech: four episodes (2.3-9.7 s) with a ~1 m
lateral offset while every other frame matched Laguna.
The radar clock (sweep optimum 0 to -0.02 s), our ego heading (static-wall
alignment within 0.2 deg) and the forward offset were ruled out, leaving the
opponent's /opponent/gps position. Single-frame misses are FRGen21 bearing noise
(Laguna: 2.6% of frames beyond 1 m, isolated), hence the rolling median and the
minimum span. Dropped scans leave the dataset entirely -- the opponent is in view,
so they are not background either -- and are listed in dataset_report.json.
First build: laguna 87 positives (4 spans), caltech 278 (4), iu 72 (2). Every span
sits at the box edge or beyond (+0.84..+1.06 m toward the ego, or Caltech's longest
-0.97 m on the far side the radar cannot see) at 5-25 deg aspect, where kept frames
sit at +0.2 m -- label error, not the visible near side of the car.
visualize_nuscenes.py --qa-dropped N draws them.
```

**`moving_speed_mps`**

```text
|v_abs| that separates the opponent from static clutter
```

**`search_along_m`**

```text
search window beyond the box, along its length
```

**`search_across_m`**

```text
...and across its width
```

**`window_s`**

```text
centred rolling-median window
```

**`along_qa_gate`**

```text
The along-track counterpart: drop spans where the opponent's moving returns stop
persistently short of (or run past) the box's near face. Found 2026-09-28 on Caltech
scene rear-013: -1.33 m for the whole 21 m -> 3 m approach, while other Caltech
scenes read -0.1..+0.3 and Laguna at 10-30 m +0.3. Not speed-dependent, so not a
clock error; it was every Caltech val box under 20 m. Measured only within
max_aspect_deg of tail/head-on, where the near face is the one in view.
```

**`enabled`**

```text
Off for the 2026-09-28 build: on the 2026-09-21 labels it would also drop 637
laguna_2026_07_26 train frames (18 spans) that have not been reviewed. Caltech's
scene 013 is excluded at eval time instead (splits.json val_qa).
```

**`search_across_m`**

```text
the near-face corridor, as the validator's near_edge uses
```

**`train_fraction`**

```text
Defaults for runs whose split policy is `trainval`. With three runs configured
the splits are run-blocked instead (see below), so these are unused today.
```

## `config/pipeline.yaml`: openpcdet

**`repo`**

```text
Moved 2026-09-29 to a vendored copy. Last build used 9e6e3fb; HEAD is
996c4ed (pp_kitti_common.py identical between the two).
vendored copy of upstream 996c4ed
```

**`local_cache_root`**

```text
run_timelines() caches decoded odometry under the run config's data_root, which
is a shared checkout. Point it at a private directory so a build can never write
there. Populate with seed_odom_cache.py.
```

**`expected_box_center_x_in_ram_m`**

```text
Where pp_kitti_common puts the opponent box centre, x in rear_axle_middle. Every
run's box_center_forward_offset_m below is measured on top of it, so the builder
refuses to run if the shared checkout moves it. Was CAR_L/2 = 2.438 until upstream
96b6d7a (2026-09-27) added OPP_REAR_OVERHANG = 1.00 m; the offsets were re-expressed
(+1.00 m each) the same day, which leaves every box exactly where it was.
```

**`extra_runs`**

```text
Runs the shared config lacks, in its format, appended to the private copy above.
```

## Runs (`config/runs/<run>.yaml`)

Split policy:

```text
Splits are run-blocked. Every split is a different session, so nothing in val or
test shares a recording, a track segment or a lap with anything in train. This
became possible only once Laguna was added; before that train and val were two
halves of one 500 s Caltech window separated by a 2 s guard.

  train  laguna   Laguna Seca, Purdue opponent, 2026-07-26
  val    caltech  Putnam Park, Caltech opponent, 2026-05-05
  test   iu       Putnam Park, IU opponent, 2026-04-15

The cost is that training sees one track. If cross-track transfer proves to be the
binding constraint, give laguna `split: trainval` (it has a natural 31% / 69%
opponent-ahead structure) and move caltech to train.
```

### laguna_2026_07_26_rear

**`location`**

```text
nuScenes log.location
```

**`box_center_forward_offset_m`**

```text
Measured by the validator's near_edge gate; see the note at the bottom.
-0.57 against the pre-2026-09-27 origin
```

**`opponent_time_offset_seconds`**

```text
see "opponent_time_offset_seconds" note at the bottom
```

**`opponent_pose_origin`**

```text
Purdue publishes base_link; treated as the opponent's centre of gravity, as
pp_kitti_common does. Used only by the validator's URDF window check.
```

**`time_source`**

```text
This bag is the recent one and its tf_static already matches the URDF exactly:
t=[-0.498, 0, 0.233], rpy=(0, -5, 180) deg. It is NOT the same rig description as
Caltech and IU, which is the whole point -- those two are missing the 5 deg pitch.
The FRGen21 header trails MCAP log time by a tight -78 ms here, which is an
ordinary acquisition-to-write latency, so the header is the better acquisition
stamp. Contrast Caltech below. Both recorders are GNSS-disciplined to 6-35 ms.
```

### laguna_2026_07_25_rear

**`location`**

```text
nuScenes log.location
```

**`multitrack_run`**

```text
Defined in extra_runs.yaml (the shared config lacks it). IU opponent, one day before
laguna_2026_07_26 on the same track; added 2026-09-28 as more Laguna training data.
```

**`box_center_forward_offset_m`**

```text
Fitted 2026-09-28 with the validator's near_edge definition: +0.465 m shortfall at
offset 0 over 3,949 tail/head-on frames, set to match laguna_2026_07_26's -0.033 m.
```

**`opponent_pose_origin`**

```text
IU's own /eskf/novatel_top/odometry (child_frame_id reads novatel_side; its yaw
tracks the direction of travel to +-1.5 deg, so the orientation is used as is).
Its top GNSS antenna (URDF top_antenna, x = +1.271 in rear_axle_middle). With the
fitted offset that puts the box centre at +1.888, inside the URDF window
[+0.738, +1.940]; read as the receiver midpoint (+1.636) it would sit 0.31 m outside.
```

**`time_source`**

```text
The FRGen21 header here is a time-since-boot clock (~1.785e12 ms behind log time),
as on IU, so log time is the only option. The next day's bag shows the header
trailing log by 78 ms. sweep-clock (2026-09-28, -0.20..+0.10 s, 4,307 positives):
spread score 0.809 m at 0 -> 0.799 at -0.06, and frames >1 m sideways 2.3% -> 1.8%,
both best at -0.06 -- they agree, as a real clock error requires (contrast IU), and
the value sits near the 78 ms acquisition-to-write latency seen the next day.
```

### caltech_2026_05_05_rear

**`location`**

```text
nuScenes log.location
```

**`box_center_forward_offset_m`**

```text
Measured by the validator's near_edge gate; see the note at the bottom.
-1.94 against the pre-2026-09-27 origin
```

**`opponent_pose_origin`**

```text
/opponent/gps is the opponent AV-24's Novatel antenna pair, not its CoG (URDF:
0.315 m ahead of it). Read that way, the -1.94 m offset puts the box centre at
+0.813 m, inside the URDF window -- the old 'outside the window' warning came from
assuming a CoG origin. Used only by the validator's URDF window check.
```

**`time_source`**

```text
Recorded 2026-05-05, before the rear radar's 5 deg nose-down pitch was measured,
so this tf_static publishes rpy=(0, 0, 180) -- pitch zero. Taken from the URDF.
This radar's header LEADS log time by +500 ms, which cannot be an acquisition
time -- the sensor clock is fast. /pcl_front, /lidar_points and
/vehicle/uva_odometry in the same bag all sit within 2 ms of their log times,
so the recorder is sound and log time is the only usable clock here.
```

### iu_2026_04_15_rear

**`location`**

```text
nuScenes log.location
```

**`box_center_forward_offset_m`**

```text
Measured by the validator's near_edge gate; see the note at the bottom.
-2.34 against the pre-2026-09-27 origin
```

**`opponent_pose_origin`**

```text
/opponent/gps: the opponent's Novatel antenna pair, as for Caltech.
```

**`time_source`**

```text
The IU rebuild relabeled the rear radar frame and kept an identity mounting
rotation, and like Caltech it predates the pitch measurement. Both the rear-facing
180 deg yaw and the 5 deg pitch now come from the URDF instead of a constant here.
This radar's header is a sensor time-since-boot clock, ~1.78e12 ms from the
epoch. Unusable; log time is the only option.
```

**`radar_time_offset_seconds`**

```text
Deliberately 0. A coarse sweep (2026-09-21) suggested the log clock runs ~120 ms
late here, but the fine sweep does not confirm it:
  sweep-clock --run iu_2026_04_15_rear --sweep -0.10 0.30 0.01
the spread score falls from 1.04 m (0 s) to 0.90 m (+0.13 s), yet frames more
than 1 m sideways of their box RISE from 16% to 29% over the same range and are
fewest at -0.02 s. A real clock error improves both, as it does on the Laguna
control (both optimal at 0..+0.02 s). With 457 usable positives this is noise,
not a measurement. IU's range-proportional lateral bias (~ -1.3 deg bearing,
median -0.32 m) remains unexplained -- not a clock error.
```

## Cross-run notes

These closed the original config file.

```text
--------------------------------------------------------------------------------
box_center_forward_offset_m

pp_kitti_common places the box centre CAR_L/2 = 2.438 m ahead of the opponent's
reported pose origin, which assumes that origin is the car's rear bumper. The AV-24
URDF says it is not. Two of its links bound the body: front_antenna sits on the nose
at x=+3.176, radar_rear sits in the rear bodywork at x=-0.498, both in
rear_axle_middle. A 4.876 m car containing both has its centre somewhere in

    [+0.738, +1.940] m ahead of rear_axle_middle

and 2.438 is outside it. That much is settled geometry, and av24_geometry.py computes
it from the URDF rather than from this comment; validate_nuscenes_dataset.py warns
for any run whose offset implies a centre outside the window.

What the URDF cannot settle is the rest. The offset also absorbs the opponent's pose
origin convention, and no opponent here reports in an AV-24 link:

  laguna   /odometry/global_filtered  child_frame_id=base_link   (Purdue's own stack)
  caltech  /opponent/gps              child_frame_id=opponent
  iu       /opponent/gps              child_frame_id=opponent

so there is nothing to look up. The values above therefore stay measured, as the
near_edge shortfall: the median leading edge of the returns in box coordinates minus
the +L/2 a correct box would put it at. Because they come from the gate itself, the
gate passing afterwards confirms the offset was applied, not that it is independently
right -- and the URDF window is the one independent check there is on them.

Read against that window, the implied centre depends on where each opponent reports
its pose from (``opponent_pose_origin`` per run; the validator's urdf_geometry_gate):

  laguna   -0.57, base_link as CoG        -> +1.868   inside
  caltech  -1.94, Novatel antenna midpoint -> +0.813   inside
  iu       -2.34, Novatel antenna midpoint -> +0.413   OUTSIDE, 0.33 m short

Caltech and IU report /opponent/gps: an antenna pair, not a centre of gravity. The
URDF puts the AV-24's Novatel midpoint 0.315 m ahead of the CoG, and read that way
Caltech's offset is ordinary body geometry (verified 2026-09-21; the earlier
"OUTSIDE" verdict assumed a CoG origin). Caltech's radar clock was checked by sweep
the same day (optimum 0 to -0.02 s), so no clock error hides in its offset either --
though with the opponent's speed nearly constant (12-14 m/s) an opponent-side clock
offset would be indistinguishable from a constant offset and equally harmless.

IU remains 0.33 m outside the window and carries a range-proportional lateral bias
(~ -1.3 deg of bearing). Both runs' opponent heading is derived from motion (the
orientation field is an identity quaternion on every sample), and a radar clock
correction for IU did not survive a fine sweep (see radar_time_offset_seconds). The
cause is not identified; IU is the test split, so it is left as measured and flagged.

RE-MEASURED 2026-09-21 on the nuScenes build (after the URDF extrinsic and the lateral
QA gate), the near_edge shortfalls are

  laguna   -0.043 m
  caltech  -0.202 m
  iu       -0.150 m

After Doppler dealiasing and the moving-return rule (same day): -0.038 / -0.178 / -0.200.
All pass the 1.0 m gate, so the values are left as fitted. Re-fitting Caltech to
re-zero the gate (-2.14) would move its implied centre to +0.61 m -- out of the URDF
window it now sits inside. Radar returns sit slightly inside a car's front face, so a
small negative shortfall is expected; re-fit only with a reason better than making the
gate read zero.
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
box origin re-expression (2026-09-28)

Everything above quotes offsets against the old origin (box centre at CAR_L/2 =
2.438 m ahead of the opponent's pose origin). Upstream 96b6d7a moved that origin to
1.438 m, so the offsets now read +1.00 m higher: laguna +0.43, caltech -0.94,
iu -1.34. The implied centres (+1.868 / +0.498 / +0.098 in rear_axle_middle) and the
URDF-window verdicts are unchanged -- this is a change of reference, not of label.
Verified by relabelling all three runs from the scan cache and comparing against the
2026-09-21 build.

--------------------------------------------------------------------------------
opponent_time_offset_seconds (per run; required)

The shared run config now gives every opponent a clock offset fitted to lidar
(upstream d644a69 / 21bf22e). For our three opponents it is -0.021 s, the median
fallback for opponents that review could not measure. It is not inherited; each run
states its own. 0.0 everywhere, because the radar measures it directly: an opponent
clock error dt puts the box dt*v along track, so the near-face shortfall would scale
with opponent speed. Regressing it on speed within scenes (scene-clustered SE, on the
2026-09-21 labels, i.e. at dt = 0):

  caltech  slope +0.000 s, 95% CI -0.014..+0.015   (excludes -0.021)
  laguna   slope -0.004 s, 95% CI -0.052..+0.044   (does not constrain it)
  iu       13.9-16.1 m/s only -- too narrow to fit; carries its own unexplained bias

Re-measure before changing any of these, and never to make a run agree with lidar.
--------------------------------------------------------------------------------
```
