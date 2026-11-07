"""Doppler-consistent training augmentation for the rear FRGen21 radar.

Generic OpenPCDet flips/rotations move ``x y`` and the boxes but not the velocity
features, so they produce scans whose Doppler contradicts their geometry. The transforms
here keep every feature consistent with the definitions in
``radar_openpcdet/build_nuscenes_dataset.py``:

* ``v_rel``  raw range-rate from the sensor.
* ``v_abs``  ``v_rel + los . v_radar`` -- ego-motion compensated radial speed, where
  ``v_radar`` is the radar's own velocity (ego twist at the mount).
* ``(v_abs_x, v_abs_y) = v_abs * los[:2]``. ``los`` is the *sensor-frame horizontal*
  unit line of sight, rotated into the ego frame by the full mount rotation. The mount
  is pitched 5 deg nose-down, so ``los[:2]`` is not exactly the ego-horizontal direction
  from the radar: on real scans it is up to 1.3 deg off it (p50 0.13 deg) and its norm
  is 0.996..0.999, not 1. :func:`builder_los_xy` reproduces the builder's ``los``
  exactly (max error 3e-6 m/s over 200 real scans).

Mount (``rear_axle_middle <- radar_rear``, from the AV-24 URDF via
``data_nuscenes/v1.0-trainval/calibrated_sensor.json`` and ``config/pipeline.yaml``):
t = [-0.498, 0.0, 0.233] m, rpy = (0, -5, 180) deg, R = Rz(yaw) Ry(pitch) Rx(roll).

Transforms (all numpy, all driven by one ``np.random.Generator``):

1. :func:`mirror_lr` -- reflection about the vertical plane ``y = mount_y`` (= 0 here).
   Exact: the reflection commutes with the mount rotation (yaw 180, pitch about the y
   axis), so ``los`` mirrors with the points. ``v_rel``/``v_abs`` are unchanged, which is
   consistent with the ego's own lateral velocity mirroring too.
2. :func:`rotate_about_radar` -- yaw rotation about the vertical axis through the radar.
   Ranges from the radar are preserved, ``v_abs`` (the target's radial speed in the
   world) is preserved, ``(v_abs_x, v_abs_y)`` is rotated and then corrected by the
   pitch term so it equals ``v_abs * builder_los_xy(new xyz)`` exactly (plain 2-D
   rotation of the vector is off by up to 0.085 m/s at 5 deg because of the pitch).
   The remaining inconsistency is in ``v_rel``: ``v_abs - v_rel = los . v_radar`` used
   the ego velocity projected on the *old* line of sight. Left alone (reproject_v_rel
   false), the scan is exactly that of an ego crabbing at a sideslip equal to the
   rotation angle, and ``v_rel`` is off from the non-slipping case by at most
   ``|v_radar| * 2 sin(a/2)`` = 25.2 m/s * 0.087 = 2.2 m/s at a = 5 deg (Laguna top
   speed); real ego slip in the data is only -2.0..+0.7 deg (p1..p99). With
   reproject_v_rel (default) ``v_radar`` is fitted per scan by least squares on
   ``v_abs - v_rel = los . v_radar`` (residual <= 2e-6 m/s on real scans) and
   ``v_rel`` is re-projected onto the new line of sight, which removes the
   inconsistency. If a ``time_lag`` column is present the fit is done per lag; a group
   whose fit is poor (< 3 points, degenerate geometry, or rms > fit_tolerance) keeps
   its ``v_rel`` (implied-slip case). Side effect of any rotation: the fixed +-60 deg
   coverage edge rotates by the same angle.
3. :func:`dropout_in_boxes` -- per box, with probability p, drop a random fraction of the
   points inside the box's BEV footprint, always keeping >= min_keep points of which
   >= min_moving_keep have |v_abs| >= moving_speed (the dataset's positive-label rule),
   so a dropped box is still a valid positive.
4. :func:`intensity_offset` -- one uniform dB offset per scan added to ``rcs`` and
   ``power``. Both are dB quantities on the same calibration: on all three runs
   ``power - rcs + 40 log10(range)`` has median 135.9 / 136.1 / 136.4 dB
   (laguna / caltech / iu), so a reflectivity change of o dB moves both by o dB.
   ``snr`` tracks ``power`` (snr ~ power - 40 dB) but is floored at the 12 dB detection
   threshold, so shifting it down would create detections the sensor cannot report;
   it is left alone unless shift_snr is set. ``confidence`` has no calibrated relation
   and is left alone. Note: of the ~18 dB static-clutter RCS gap between Laguna and
   Putnam, part is range mix (Laguna clutter median range 49 m vs 19 m, rcs rising with
   range); at matched range the gap is ~11-13 dB in rcs and ~0-6 dB in power.

Global scaling and translation are deliberately absent: they break range/Doppler
physics and the fixed sensor position.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np

# rear_axle_middle <- radar_rear, AV-24 URDF (data_nuscenes calibrated_sensor.json).
REAR_RADAR_MOUNT_XYZ = (-0.498, 0.0, 0.233)
REAR_RADAR_MOUNT_RPY_DEG = (0.0, -5.0, 180.0)

DEFAULT_CONFIG: dict[str, Any] = {
    "ENABLED": True,
    "MOUNT_XYZ": list(REAR_RADAR_MOUNT_XYZ),
    "MOUNT_RPY_DEG": list(REAR_RADAR_MOUNT_RPY_DEG),
    "MIRROR": {"PROB": 0.5},
    "ROTATION": {"PROB": 1.0, "MAX_DEG": 5.0, "REPROJECT_V_REL": True, "FIT_TOLERANCE_MPS": 0.1},
    "BOX_DROPOUT": {
        "PROB": 0.3,
        "FRACTION": [0.2, 0.8],
        "MIN_KEEP": 2,
        "MIN_MOVING_KEEP": 1,
        "MOVING_SPEED_MPS": 2.0,
    },
    "INTENSITY_OFFSET": {"PROB": 1.0, "MAX_DB": 10.0, "SHIFT_SNR": False},
}


# -- helpers ---------------------------------------------------------------------------


def _merge(defaults: Mapping[str, Any], cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(dict(defaults))
    for key, value in (cfg or {}).items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _cols(feature_names: Sequence[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(feature_names)}


def mount_matrix(xyz: Sequence[float] = REAR_RADAR_MOUNT_XYZ,
                 rpy_deg: Sequence[float] = REAR_RADAR_MOUNT_RPY_DEG) -> np.ndarray:
    """4x4 ego <- sensor transform, URDF convention R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = np.radians(np.asarray(rpy_deg, dtype=np.float64))
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    out = np.eye(4)
    out[:3, :3] = rz @ ry @ rx
    out[:3, 3] = xyz
    return out


def builder_los_xy(xyz: np.ndarray, mount: np.ndarray) -> np.ndarray:
    """Ego-frame xy of the builder's line of sight (sensor-horizontal unit vector, rotated).

    Mirrors build_nuscenes_dataset.py: points closer than 1 mm horizontally get 0.
    """
    rot, trans = mount[:3, :3], mount[:3, 3]
    xs = (np.asarray(xyz, dtype=np.float64) - trans) @ rot
    horizontal = np.linalg.norm(xs[:, :2], axis=1)
    los = np.zeros_like(xs)
    good = horizontal > 1e-3
    los[good, :2] = xs[good, :2] / horizontal[good, None]
    return (los @ rot.T)[:, :2]


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def points_in_bev_boxes(xy: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(N, M) bool: point n lies inside the BEV rectangle of box m (edges inclusive)."""
    if len(boxes) == 0:
        return np.zeros((len(xy), 0), dtype=bool)
    d = xy[:, None, :2].astype(np.float64) - boxes[None, :, :2].astype(np.float64)
    c, s = np.cos(boxes[:, 6]), np.sin(boxes[:, 6])
    along = d[..., 0] * c + d[..., 1] * s
    across = -d[..., 0] * s + d[..., 1] * c
    return (np.abs(along) <= boxes[:, 3] / 2) & (np.abs(across) <= boxes[:, 4] / 2)


# -- transforms ------------------------------------------------------------------------


def mirror_lr(points: np.ndarray, gt_boxes: np.ndarray, feature_names: Sequence[str],
              mirror_y: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Reflect about the vertical plane y = mirror_y (the radar's y). Self-inverse."""
    col = _cols(feature_names)
    points = points.copy()
    gt_boxes = gt_boxes.copy()
    points[:, col["y"]] = 2 * mirror_y - points[:, col["y"]]
    if "v_abs_y" in col:
        points[:, col["v_abs_y"]] = -points[:, col["v_abs_y"]]
    if len(gt_boxes):
        gt_boxes[:, 1] = 2 * mirror_y - gt_boxes[:, 1]
        gt_boxes[:, 6] = _wrap_angle(-gt_boxes[:, 6])
    return points, gt_boxes


def _fit_radar_velocity(los_xy: np.ndarray, d: np.ndarray, tol: float) -> np.ndarray | None:
    """Least-squares v_radar (xy) from d = v_abs - v_rel = los . v_radar, or None if unreliable."""
    if len(d) < 3:
        return None
    a = los_xy.astype(np.float64)
    if np.linalg.matrix_rank(a, tol=1e-3) < 2:
        return None
    v, *_ = np.linalg.lstsq(a, d.astype(np.float64), rcond=None)
    rms = float(np.sqrt(np.mean((a @ v - d) ** 2)))
    return v if rms <= tol else None


def rotate_about_radar(points: np.ndarray, gt_boxes: np.ndarray, feature_names: Sequence[str],
                       angle: float, mount: np.ndarray | None = None,
                       reproject_v_rel: bool = True, fit_tolerance: float = 0.1
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the scene by ``angle`` (rad, CCW) about the vertical axis through the radar.

    See the module docstring for the Doppler bookkeeping. ``rotate(+a)`` then
    ``rotate(-a)`` is the identity (to float precision).
    """
    mount = mount_matrix() if mount is None else mount
    col = _cols(feature_names)
    cx, cy = mount[0, 3], mount[1, 3]
    c, s = math.cos(angle), math.sin(angle)
    rot2 = np.array([[c, -s], [s, c]])

    src = points.astype(np.float64)
    out = points.copy()
    ix, iy = col["x"], col["y"]
    xy_old = src[:, [ix, iy]]
    xy_new = (xy_old - (cx, cy)) @ rot2.T + (cx, cy)
    out[:, ix], out[:, iy] = xy_new[:, 0], xy_new[:, 1]

    has_z = "z" in col
    xyz_old = np.column_stack([xy_old, src[:, col["z"]] if has_z else np.zeros(len(src))])
    xyz_new = np.column_stack([xy_new, xyz_old[:, 2]])
    los_old = builder_los_xy(xyz_old, mount)
    los_new = builder_los_xy(xyz_new, mount)

    if "v_abs_x" in col and "v_abs_y" in col:
        vxy = src[:, [col["v_abs_x"], col["v_abs_y"]]] @ rot2.T
        if "v_abs" in col:
            # Pitch correction: exact v_abs * los_new for single-scan points.
            vxy += src[:, [col["v_abs"]]] * (los_new - los_old @ rot2.T)
        out[:, col["v_abs_x"]], out[:, col["v_abs_y"]] = vxy[:, 0], vxy[:, 1]

    if reproject_v_rel and "v_rel" in col and "v_abs" in col and len(src):
        d = src[:, col["v_abs"]] - src[:, col["v_rel"]]
        groups = (np.unique(src[:, col["time_lag"]], return_inverse=True)[1]
                  if "time_lag" in col else np.zeros(len(src), dtype=np.int64))
        for g in np.unique(groups):
            m = groups == g
            v_radar = _fit_radar_velocity(los_old[m], d[m], fit_tolerance)
            if v_radar is not None:
                out[m, col["v_rel"]] = src[m, col["v_rel"]] - (los_new[m] - los_old[m]) @ v_radar

    gt_boxes = gt_boxes.copy()
    if len(gt_boxes):
        bxy = (gt_boxes[:, :2].astype(np.float64) - (cx, cy)) @ rot2.T + (cx, cy)
        gt_boxes[:, 0], gt_boxes[:, 1] = bxy[:, 0], bxy[:, 1]
        gt_boxes[:, 6] = _wrap_angle(gt_boxes[:, 6].astype(np.float64) + angle)
    return out, gt_boxes


def dropout_in_boxes(points: np.ndarray, gt_boxes: np.ndarray, feature_names: Sequence[str],
                     rng: np.random.Generator, prob: float = 0.3,
                     fraction: Sequence[float] = (0.2, 0.8), min_keep: int = 2,
                     min_moving_keep: int = 1, moving_speed: float = 2.0) -> np.ndarray:
    """Drop a random fraction of in-box points per box (with probability ``prob``).

    Each selected box keeps >= min_keep points, >= min_moving_keep of them moving. Boxes
    that already fail that rule are left untouched. Points protected for one box are
    never dropped on behalf of an overlapping box.
    """
    if len(gt_boxes) == 0 or len(points) == 0:
        return points
    col = _cols(feature_names)
    inside = points_in_bev_boxes(points[:, [col["x"], col["y"]]], gt_boxes)
    moving = np.abs(points[:, col["v_abs"]]) >= moving_speed
    protected = np.zeros(len(points), dtype=bool)
    plans = []
    for b in range(len(gt_boxes)):
        if rng.random() >= prob:
            continue
        idx = np.flatnonzero(inside[:, b])
        mov = idx[moving[idx]]
        if len(idx) <= min_keep or len(mov) < min_moving_keep:
            continue
        keep = list(rng.choice(mov, size=min_moving_keep, replace=False)) if min_moving_keep else []
        rest = np.setdiff1d(idx, keep)
        extra = max(0, min_keep - len(keep))
        if extra:
            keep += list(rng.choice(rest, size=extra, replace=False))
        protected[keep] = True
        plans.append((idx, float(rng.uniform(*fraction))))
    drop = np.zeros(len(points), dtype=bool)
    for idx, frac in plans:
        cand = idx[~protected[idx] & ~drop[idx]]
        n_drop = min(len(cand), int(round(frac * len(idx))))
        if n_drop:
            drop[rng.choice(cand, size=n_drop, replace=False)] = True
    return points[~drop]


def intensity_offset(points: np.ndarray, feature_names: Sequence[str], offset_db: float,
                     shift_snr: bool = False) -> np.ndarray:
    """Add one dB offset to rcs and power (and snr if asked). Other columns untouched."""
    col = _cols(feature_names)
    out = points.copy()
    names = ["rcs", "power"] + (["snr"] if shift_snr else [])
    for name in names:
        if name in col:
            out[:, col[name]] = out[:, col[name]] + offset_db
    return out


# -- entry point -----------------------------------------------------------------------


def augment_radar_sample(points: np.ndarray, gt_boxes: np.ndarray, feature_names: Sequence[str],
                         cfg: Mapping[str, Any] | None, rng: np.random.Generator
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Apply the enabled radar augmentations. Deterministic given ``rng``.

    points: (N, C) ego-frame features named by ``feature_names`` (unknown columns such
    as ``time_lag`` pass through). gt_boxes: (M, >=7) [x y z dx dy dz heading, ...];
    extra box columns pass through. ``cfg`` is merged over :data:`DEFAULT_CONFIG`
    (dict or EasyDict; ``None`` means all defaults). Returns new arrays, same dtypes.
    """
    cfg = _merge(DEFAULT_CONFIG, cfg)
    if not cfg["ENABLED"]:
        return points, gt_boxes
    dtype_p, dtype_b = points.dtype, gt_boxes.dtype
    mount = mount_matrix(cfg["MOUNT_XYZ"], cfg["MOUNT_RPY_DEG"])

    drop_cfg = cfg["BOX_DROPOUT"]
    if drop_cfg["PROB"] > 0:
        points = dropout_in_boxes(
            points, gt_boxes, feature_names, rng, prob=float(drop_cfg["PROB"]),
            fraction=tuple(drop_cfg["FRACTION"]), min_keep=int(drop_cfg["MIN_KEEP"]),
            min_moving_keep=int(drop_cfg["MIN_MOVING_KEEP"]),
            moving_speed=float(drop_cfg["MOVING_SPEED_MPS"]),
        )

    if rng.random() < float(cfg["MIRROR"]["PROB"]):
        points, gt_boxes = mirror_lr(points, gt_boxes, feature_names, mirror_y=float(mount[1, 3]))

    rot_cfg = cfg["ROTATION"]
    if rng.random() < float(rot_cfg["PROB"]):
        angle = math.radians(float(rng.uniform(-rot_cfg["MAX_DEG"], rot_cfg["MAX_DEG"])))
        points, gt_boxes = rotate_about_radar(
            points, gt_boxes, feature_names, angle, mount,
            reproject_v_rel=bool(rot_cfg["REPROJECT_V_REL"]),
            fit_tolerance=float(rot_cfg["FIT_TOLERANCE_MPS"]),
        )

    int_cfg = cfg["INTENSITY_OFFSET"]
    if rng.random() < float(int_cfg["PROB"]):
        offset = float(rng.uniform(-int_cfg["MAX_DB"], int_cfg["MAX_DB"]))
        points = intensity_offset(points, feature_names, offset, bool(int_cfg["SHIFT_SNR"]))

    return points.astype(dtype_p, copy=False), gt_boxes.astype(dtype_b, copy=False)
