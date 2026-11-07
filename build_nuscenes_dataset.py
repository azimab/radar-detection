#!/usr/bin/env python3
"""Build the rear-radar Car dataset in nuScenes v1.0 format from indexed ROS 2 MCAPs.

Two stages, so the expensive part runs once:

    decode   read each run's bags and cache every rear-radar scan, the ego twist and the
             bag's tf_static under ``cache/scans/<run>.npz``. Heavy I/O (Caltech's bag is
             213 GB); run it on a compute node -- see build_nuscenes_dataset.slurm.
    label    turn the cached scans into labelled nuScenes samples. Minutes, no bag I/O,
             so label policy (offsets, gates, selection) can be retuned freely.

    python3 radar_openpcdet/build_nuscenes_dataset.py decode --runs iu_2026_04_15_rear
    python3 radar_openpcdet/build_nuscenes_dataset.py label --overwrite
    python3 radar_openpcdet/build_nuscenes_dataset.py sweep-clock --run iu_2026_04_15_rear

Run geometry is not re-declared here. Ego and opponent timelines come from
``pp_run_config_multitrack.yaml`` through ``build_pp_kitti.run_timelines``, the same
function that assembles the lidar dataset, so datum offsets, ``ego_yaw_correction_deg``
and the degenerate-orientation repair cannot drift between the two datasets. Two things
are deliberately NOT inherited, because they are measured on the radar and must not move
when the shared checkout changes: each run's opponent clock
(``opponent_time_offset_seconds``, required per run) and, through the pinned
``openpcdet.expected_box_center_x_in_ram_m``, the box-origin convention that each run's
``box_center_forward_offset_m`` is measured against. The rear radar's mounting comes from the AV-24 URDF
(``av24_geometry``). This file owns only what is specific to the rear radar: which
topic and clock, coverage and selection, the label-QA gate, and the nuScenes layout.

Output (``dataset.output_root``)::

    samples/RADAR_REAR/<run>__RADAR_REAR__<us>.pcd          nuScenes 18-field radar PCD, sensor frame
    samples/RADAR_REAR_FRGEN21/<run>__RADAR_REAR__<us>.npy  (N, 11) float32 model input, ego frame
                             (keyframes and non-key sweeps alike; nuScenes would put the
                             sweeps under sweeps/, but one directory keeps the loader simple)
    v1.0-trainval/*.json     train + val scenes
    v1.0-test/*.json         test scenes (annotations included)
    maps/                    empty -- no map rasters exist for these tracks
    splits.json              scene names per split (plus keys carried from carry_splits_from)
    manifest.jsonl           every decoded scan and why it was kept or dropped; written
                             sweeps carry ``sweep_of_sample_token``
    dataset_report.json      per-run geometry, counts and QA
    feature_stats.json       train-split feature statistics

Frames and conventions
----------------------
nuScenes' ego frame is the rear-axle centre, which is ``rear_axle_middle``, the frame the
boxes and the Doppler compensation have always lived in.

* ``ego_pose`` is ``T_map_rear_axle_middle`` at the scan's (corrected) acquisition time.
* ``calibrated_sensor`` is the URDF's ``rear_axle_middle <- radar_rear``.
* Annotation ``translation``/``rotation`` are global. The rotation is ``q_ego * Rz(yaw)``
  so the devkit's global -> ego transform gives back the planar label yaw exactly.
* ``size`` is nuScenes' ``[width, length, height]``.
* The ``.pcd`` carries exactly nuScenes' 18 ARS408 fields, because the devkit reads
  them positionally. FRGen21 quantities are mapped only where the meaning matches:
  ``x y z`` (sensor frame), ``rcs``, ``vx vy`` = ``v_rel`` along the horizontal line of
  sight, ``vx_comp vy_comp`` = ``v_abs`` along it (rotated into the ego frame these are
  exactly the sidecar's ``v_abs_x v_abs_y``), ``id`` = point index. The ARS408 status
  fields have no FRGen21 source and hold the values that mean "unknown / not
  filtered": dyn_prop=4, ambig_state=3, invalid_state=0, is_quality_valid=1, the rest 0.
* The sidecar ``.npy`` is the model input: all 11 FRGen21 features in the ego frame,
  column order ``dataset.feature_names``. ``snr``, ``confidence`` and ``power`` exist
  only there.

Not nuScenes-standard: there is no ``LIDAR_TOP`` (devkit detection eval and lidar-keyed
converters must be pointed at ``RADAR_REAR``); ``visibility_token`` is empty because
nuScenes derives it from cameras; ``num_lidar_pts`` is 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml
from repo_config import DEFAULT_CONFIG, load_config
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import av24_geometry  # noqa: E402

POINT_FIELD_DTYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}
SOURCE_FIELDS = ["x", "y", "z", "velocity", "rcs", "confidence", "snr", "power"]
SCAN_CACHE_VERSION = 1

# The rear radar's PointCloud2 header clock differs per run and the difference is not
# cosmetic, so the choice is configured per run rather than assumed:
#
#   iu_2026_04_15   header is a sensor time-since-boot clock  -> unusable, use log
#   caltech         header leads log time by a tight +500 ms  -> impossible as an
#                   acquisition time, so the sensor clock is fast; use log
#   laguna          header trails log time by ~78 ms          -> an ordinary
#                   acquisition-to-write latency; the header is the better stamp
#
# Verified against /pcl_front, /lidar_points and /vehicle/uva_odometry in the same bags,
# whose headers sit within 2 ms of their log times in every run. A residual constant
# offset on top of the chosen clock is ``radar_time_offset_seconds`` (see sweep-clock).
TIME_SOURCES = ("log", "header")

CHANNEL = "RADAR_REAR"
FEATURE_DIR = "RADAR_REAR_FRGEN21"
CATEGORY = "vehicle.car"
SPLIT_VERSION = {"train": "v1.0-trainval", "val": "v1.0-trainval", "test": "v1.0-test"}
NUSCENES_TABLES = (
    "category", "attribute", "visibility", "instance", "sensor", "calibrated_sensor",
    "ego_pose", "log", "scene", "sample", "sample_data", "sample_annotation", "map",
)
# Same window the devkit's box_velocity uses to call a velocity estimate valid.
VELOCITY_MAX_DT = 1.5
MOVING_SPEED_MPS = 1.0

PCD_DTYPE = np.dtype(
    [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("dyn_prop", "i1"), ("id", "<i2"), ("rcs", "<f4"),
        ("vx", "<f4"), ("vy", "<f4"), ("vx_comp", "<f4"), ("vy_comp", "<f4"),
        ("is_quality_valid", "i1"), ("ambig_state", "i1"),
        ("x_rms", "i1"), ("y_rms", "i1"), ("invalid_state", "i1"), ("pdh0", "i1"),
        ("vx_rms", "i1"), ("vy_rms", "i1"),
    ]
)  # unaligned: 43 bytes a point, the layout nuScenes' own radar files use
PCD_TYPE = {"f": "F", "i": "I", "u": "U"}


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def strip_frame(frame: str) -> str:
    return str(frame).strip().lstrip("/")


def stamp_ns(msg: Any) -> int:
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def token(*parts: Any) -> str:
    """Deterministic 32-hex token, so a rebuild produces identical tables."""
    return hashlib.md5("/".join(map(str, parts)).encode()).hexdigest()


def quat_wxyz(rotation: np.ndarray) -> list[float]:
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    if w < 0:  # one hemisphere, for readable diffs; both signs are the same rotation
        x, y, z, w = -x, -y, -z, -w
    return [float(w), float(x), float(y), float(z)]


def fingerprint(path: Path) -> list[str]:
    stat = path.stat()
    return [str(path), str(stat.st_size), str(stat.st_mtime_ns)]


def transform_matrix(translation: Any, quaternion: Any) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_quat(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    ).as_matrix()
    out[:3, 3] = [translation.x, translation.y, translation.z]
    return out


def box_frame(xy: np.ndarray, center: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Along-box and across-box coordinates of BEV points."""
    delta = xy[:, :2].astype(np.float64) - center[:2]
    c, s = math.cos(-yaw), math.sin(-yaw)
    return c * delta[:, 0] - s * delta[:, 1], s * delta[:, 0] + c * delta[:, 1]


def git_head(repo: Path) -> str | None:
    """Commit the shared checkout is at, read from .git without invoking git."""
    git = repo / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        loose = git / ref
        if loose.exists():
            return loose.read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


def opponent_time_offset(run_entry: dict[str, Any]) -> float:
    """The run's opponent clock offset, which the rear-radar config must state.

    The shared run config also carries one per opponent, fitted to lidar. It is not used:
    our runs' values there are an unmeasured fallback, and Caltech's radar residuals
    exclude it (see the note on ``opponent_time_offset_seconds`` in the config).
    """
    if "opponent_time_offset_seconds" not in run_entry:
        raise KeyError(
            f"{run_entry['name']}: opponent_time_offset_seconds is required; the shared "
            "run config's value is deliberately not inherited"
        )
    return float(run_entry["opponent_time_offset_seconds"])


def count_points_in_bev_car_box(
    points: np.ndarray, center: np.ndarray, yaw: float, car_lwh: list[float]
) -> int:
    """Count radar returns in the oriented x/y footprint; ignore noisy elevation."""
    length, width, _ = map(float, car_lwh)
    along, across = box_frame(points, center, yaw)
    return int(((np.abs(along) <= length * 0.5) & (np.abs(across) <= width * 0.5)).sum())


# --------------------------------------------------------------------------------------
# bag reading
# --------------------------------------------------------------------------------------


def load_static_edges(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    with path.open("rb") as stream:
        messages = make_reader(
            stream, decoder_factories=[DecoderFactory()]
        ).iter_decoded_messages(topics=["/tf_static"])
        try:
            _, _, _, msg = next(messages)
        except StopIteration as exc:
            raise RuntimeError(f"no /tf_static in {path}") from exc
    edges: dict[str, tuple[str, np.ndarray]] = {}
    for item in msg.transforms:
        edges[strip_frame(item.child_frame_id)] = (
            strip_frame(item.header.frame_id),
            transform_matrix(item.transform.translation, item.transform.rotation),
        )
    return edges


def resolve_to_target(
    edges: dict[str, tuple[str, np.ndarray]], source: str, target: str
) -> np.ndarray:
    current = strip_frame(source)
    target = strip_frame(target)
    result = np.eye(4, dtype=np.float64)
    visited: set[str] = set()
    while current != target:
        if current in visited:
            raise RuntimeError(f"cycle while resolving {source} -> {target}")
        visited.add(current)
        if current not in edges:
            raise KeyError(f"no tf_static parent for {current!r} resolving {source} -> {target}")
        parent, t_parent_current = edges[current]
        result = t_parent_current @ result
        current = parent
    return result


def choose_sensor_transform(
    edges: dict[str, tuple[str, np.ndarray]], aliases: Iterable[str], target: str
) -> tuple[str, np.ndarray]:
    errors = []
    for alias in aliases:
        try:
            return strip_frame(alias), resolve_to_target(edges, alias, target)
        except (KeyError, RuntimeError) as exc:
            errors.append(str(exc))
    raise RuntimeError("could not resolve a rear radar frame: " + "; ".join(errors))


def reconcile_sensor_extrinsic(
    rig: av24_geometry.Av24Rig,
    t_ram_bag: np.ndarray,
    target_frame: str,
    tolerance_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconcile a bag's rear-radar extrinsic against the AV-24 URDF, URDF winning.

    The URDF wins on *rotation*, and the reason is specific rather than a preference for
    one file over another. The rear radar is bolted 5 degrees out of the horizontal
    plane -- the middle ``rpy`` term -- which ``radar_ego_velocity``'s ROLEQ calibrator
    measured from the Doppler fit itself (~4.8 deg) before it was written into the URDF.
    Laguna's 2026-07-26 tf_static carries it; the Caltech (2026-05-05) and IU
    (2026-04-15) bags predate the measurement and publish an identity pitch. So the same
    physical sensor is described two ways across the three splits -- train with the
    pitch, val and test without -- and left alone that is a calibration difference
    between the splits wearing the costume of a domain gap.

    It also subsumes the old per-run ``sensor_yaw_correction_deg``. IU's rebuild relabels
    the frame and drops the rear-facing 180 degree yaw; taking the whole rotation from
    the URDF restores it without a hand-entered constant.

    *Translation* is checked, never overridden. A bag that disagrees about where the
    sensor is bolted is describing a different rig, and quietly picking a winner is the
    failure this reconciliation exists to prevent, so it raises.
    """
    t_ram_urdf = rig.transform(target_frame, av24_geometry.RADAR_REAR)
    delta = t_ram_bag[:3, 3] - t_ram_urdf[:3, 3]
    if float(np.linalg.norm(delta)) > tolerance_m:
        raise ValueError(
            f"tf_static puts the rear radar at {np.round(t_ram_bag[:3, 3], 4).tolist()} "
            f"in {target_frame}, the URDF at {np.round(t_ram_urdf[:3, 3], 4).tolist()} "
            f"({np.linalg.norm(delta):.4f} m apart, tolerance {tolerance_m} m). These "
            f"describe different rigs; {rig.source} cannot arbitrate a mounting move."
        )
    correction = Rotation.from_matrix(t_ram_bag[:3, :3].T @ t_ram_urdf[:3, :3])
    info = {
        "sensor_extrinsic_source": str(rig.source),
        "sensor_translation_delta_m": np.round(delta, 6).tolist(),
        "sensor_rotation_correction_rpy_deg": np.round(
            np.degrees(correction.as_euler("xyz")), 4
        ).tolist(),
        "sensor_rotation_correction_deg": round(math.degrees(correction.magnitude()), 4),
    }
    return t_ram_urdf, info


def pointcloud_array(msg: Any, names: list[str]) -> np.ndarray:
    if msg.is_bigendian:
        raise RuntimeError("big-endian PointCloud2 is not supported")
    fields = {field.name: field for field in msg.fields}
    missing = [name for name in names if name not in fields]
    if missing:
        raise RuntimeError(f"PointCloud2 missing {missing}; fields={sorted(fields)}")
    dtype = np.dtype(
        {
            "names": names,
            "formats": [POINT_FIELD_DTYPES[int(fields[name].datatype)] for name in names],
            "offsets": [int(fields[name].offset) for name in names],
            "itemsize": int(msg.point_step),
        }
    )
    packed = np.frombuffer(msg.data, dtype=dtype, count=int(msg.width) * int(msg.height))
    return np.column_stack([packed[name] for name in names]).astype(np.float32, copy=False)


@dataclass
class EgoTwist:
    """Body-frame ego twist from nav_msgs/Odometry, keyed by header stamp (seconds)."""

    t: np.ndarray
    linear: np.ndarray
    angular: np.ndarray

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        interp = lambda values: np.array(
            [np.interp(t, self.t, values[:, axis]) for axis in range(3)], dtype=np.float64
        )
        return interp(self.linear), interp(self.angular)


def load_ego_twist(path: Path, topic: str, expected_child_frame: str) -> EgoTwist:
    # The twist is the odometry's own velocity estimate. Differentiating the position
    # track instead produced single-frame spikes of +7 m/s that corrupted v_abs.
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    t, linear, angular = [], [], []
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, _, _, msg in reader.iter_decoded_messages(topics=[topic]):
            child = strip_frame(msg.child_frame_id)
            if child != expected_child_frame:
                raise ValueError(
                    f"{path}:{topic} twist is in {child!r}, expected {expected_child_frame!r}"
                )
            lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
            t.append(stamp_ns(msg) * 1e-9)
            linear.append((lin.x, lin.y, lin.z))
            angular.append((ang.x, ang.y, ang.z))
    if not t:
        raise RuntimeError(f"{path}: no {topic} messages")
    order = np.argsort(t)
    return EgoTwist(
        np.asarray(t)[order], np.asarray(linear)[order], np.asarray(angular)[order]
    )


# --------------------------------------------------------------------------------------
# shared geometry (pp_kitti_common, via the lidar pipeline's run config)
# --------------------------------------------------------------------------------------


def local_run_config(run_config: Path, cache_root: Path, extra_runs: list[dict[str, Any]] | None = None) -> Path:
    """Copy of the shared run config whose data_root points at a private directory.

    ``run_timelines`` reads and writes an odometry cache under ``paths.data_root``.
    That directory belongs to a shared checkout, so redirect it rather than risk
    writing there. ``paths.repo_dir`` is left alone so ``model.cfg_file`` still
    resolves against the real checkout. ``extra_runs`` (entries in the same format, from
    the run files' ``multitrack:`` blocks) are appended; a name the shared config already
    defines is refused, never overridden.
    """
    raw = yaml.safe_load(run_config.read_text())
    raw["paths"]["data_root"] = str(cache_root)
    if extra_runs:
        shared = {run["name"] for run in raw["inputs"]["runs"]}
        clash = sorted(shared & {run["name"] for run in extra_runs})
        if clash:
            raise ValueError(f"config/runs: multitrack {clash} already defined in {run_config}; "
                             "drop the multitrack: block and use the shared definition")
        raw["inputs"]["runs"].extend(extra_runs)
    cache_root.mkdir(parents=True, exist_ok=True)
    local = cache_root / "pp_run_config.local.yaml"
    local.write_text(yaml.safe_dump(raw, sort_keys=False))
    return local


class Geometry:
    """pp_kitti_common, the parsed multitrack run config, and the AV-24 rig."""

    def __init__(self, config: dict[str, Any]):
        repo = Path(config["openpcdet"]["repo"])
        sys.path.insert(0, str(repo / "tools"))
        import build_pp_kitti  # type: ignore
        import pp_kitti_common  # type: ignore

        run_config = repo / config["openpcdet"]["run_config"]
        cache_root = config["openpcdet"].get("local_cache_root")
        extra_runs = config["openpcdet"].get("extra_runs")
        if extra_runs and not cache_root:
            raise ValueError("multitrack: blocks in config/runs need openpcdet.local_cache_root: "
                             "they go into the private copy")
        if cache_root:
            run_config = local_run_config(run_config, Path(cache_root), extra_runs)
        self.pp = pp_kitti_common
        self.pp_builder = build_pp_kitti
        self.run_config = pp_kitti_common.load_run_config(run_config)
        pp_kitti_common.configure_from_run_config(self.run_config)
        self.runs_by_name = {run.name: run for run in self.run_config.runs}

        dataset_cfg = config["dataset"]
        self.car_lwh = [float(v) for v in dataset_cfg["car_size_lwh"]]
        # One source of truth for car geometry: the model yaml the shared run config
        # points at already defines it, and the box centre computed by pp uses those
        # values. Two independent copies silently disagreeing is exactly how a z
        # offset gets baked into labels.
        expected = [float(self.pp.CAR_L), float(self.pp.CAR_W), float(self.pp.CAR_H)]
        if not np.allclose(self.car_lwh, expected, atol=1e-6):
            raise ValueError(
                f"car_size_lwh {self.car_lwh} disagrees with the model yaml {expected}; "
                "these must match or GT centre height and written extent diverge"
            )
        self.rig = av24_geometry.Av24Rig.load(dataset_cfg["av24_urdf"])
        # pp_kitti_common hardcodes its own copy of rear_axle_middle -> center_of_gravity
        # and builds every opponent box on it. Same quantity, two files: check it.
        av24_geometry.check_pp_kitti_agrees(self.rig, self.pp)
        if not np.allclose(self.pp.T_COG_RAM[:3, :3], np.eye(3)):
            raise ValueError("T_COG_RAM has a rotation; the body twist needs rotating too")
        # Every run's box_center_forward_offset_m is measured on top of the shared
        # checkout's box origin. If that origin moves, every label moves with it
        # (2026-09-27: OPP_REAR_OVERHANG moved it 1.0 m back), so pin it and fail loudly.
        expected = float(config["openpcdet"]["expected_box_center_x_in_ram_m"])
        actual = av24_geometry.pp_box_center_in_ram(self.pp)
        if abs(actual - expected) > 1e-6:
            raise ValueError(
                f"pp_kitti_common puts the opponent box centre at x={actual:.4f} m in "
                f"rear_axle_middle, but openpcdet.expected_box_center_x_in_ram_m is "
                f"{expected:.4f}. Every run's box_center_forward_offset_m is measured "
                f"against the latter: shift each by {expected - actual:+.4f} m to keep the "
                "labels where they are, then update the expected value."
            )
        self.upstream_commit = git_head(repo)

    def timelines(self, run_entry: dict[str, Any]) -> tuple[Any, Any, Any]:
        """Ego/opponent timelines with every configured correction already applied."""
        name = run_entry["multitrack_run"]
        if name not in self.runs_by_name:
            raise KeyError(f"{name!r} is not a run in the multitrack run config")
        run = self.runs_by_name[name]
        if len(run.opponents) != 1:
            raise ValueError(
                f"{name}: {len(run.opponents)} opponents configured; the rear-radar "
                "builder labels a single target and would silently drop the rest"
            )
        ego, opponents = self.pp_builder.run_timelines(self.run_config, run, dry_run=True)
        opponent_cfg, opponent = opponents[0]
        return ego, opponent, opponent_cfg


# --------------------------------------------------------------------------------------
# stage 1: decode
# --------------------------------------------------------------------------------------


def scan_cache_path(config: dict[str, Any], run_name: str) -> Path:
    return Path(config["dataset"]["scan_cache_root"]) / f"{run_name}.npz"


def decode_run(config: dict[str, Any], geometry: Geometry, run_entry: dict[str, Any]) -> None:
    """Cache every rear-radar scan in the run's ego/opponent overlap, untouched.

    Raw source fields in the bag's sensor frame, both clocks, the ego twist and the
    bag's rear-radar tf_static edge. Nothing is filtered or labelled here, so no label
    policy change ever needs the bags again.
    """
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    name = run_entry["name"]
    dataset_cfg = config["dataset"]
    out = scan_cache_path(config, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[decode] {name}", flush=True)

    ego, opponent, _ = geometry.timelines(run_entry)
    t0 = max(ego.t_start, opponent.t_start)
    t1 = min(ego.t_end, opponent.t_end)
    if t1 <= t0:
        raise RuntimeError(f"no ego/opponent overlap for {name}")

    edges = load_static_edges(Path(run_entry["tf_mcap"]))
    sensor_frame, t_ram_bag = choose_sensor_transform(
        edges, run_entry["sensor_frame_aliases"], run_entry["target_frame"]
    )
    twist = load_ego_twist(
        Path(run_entry["ego_odom_mcap"]),
        dataset_cfg["ego_odom_topic"],
        dataset_cfg["ego_odom_child_frame"],
    )
    print(f"  tf_static: {sensor_frame}  twist samples: {len(twist.t)}", flush=True)

    radar_path = Path(run_entry["radar_mcap"])
    header, log, counts, frames, chunks = [], [], [], [], []
    # The window is padded well beyond any plausible clock correction; labelling
    # re-checks every scan against the pose timelines anyway.
    pad = 3.0
    with radar_path.open("rb") as stream:
        iterator = make_reader(stream, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
            topics=[run_entry["radar_topic"]],
            start_time=int((t0 - pad) * 1e9),
            end_time=int((t1 + pad) * 1e9),
        )
        for _, _, record, msg in iterator:
            points = pointcloud_array(msg, SOURCE_FIELDS)
            header.append(stamp_ns(msg))
            log.append(int(record.log_time))
            counts.append(len(points))
            frames.append(strip_frame(msg.header.frame_id))
            chunks.append(points)
            if len(header) % 2000 == 0:
                print(f"  scans={len(header)}", flush=True)
    if not header:
        raise RuntimeError(f"{radar_path}: no {run_entry['radar_topic']} in the pose window")

    np.savez(
        out,
        cache_version=np.array([SCAN_CACHE_VERSION]),
        source_fingerprints=np.array(
            [fingerprint(radar_path), fingerprint(Path(run_entry["ego_odom_mcap"])),
             fingerprint(Path(run_entry["tf_mcap"]))],
            dtype=object,
        ),
        radar_topic=np.array(run_entry["radar_topic"]),
        source_fields=np.array(SOURCE_FIELDS),
        header_ns=np.asarray(header, dtype=np.int64),
        log_ns=np.asarray(log, dtype=np.int64),
        point_counts=np.asarray(counts, dtype=np.int64),
        points=np.concatenate(chunks).astype(np.float32),
        msg_frame_ids=np.asarray(frames),
        tf_sensor_frame=np.array(sensor_frame),
        tf_t_ram_sensor=t_ram_bag,
        twist_t=twist.t,
        twist_linear=twist.linear,
        twist_angular=twist.angular,
    )
    print(f"  wrote {out} ({len(header)} scans, {sum(counts)} points)", flush=True)


@dataclass
class ScanCache:
    header_ns: np.ndarray
    log_ns: np.ndarray
    offsets: np.ndarray
    points: np.ndarray
    msg_frame_ids: np.ndarray
    sensor_frame: str
    t_ram_bag: np.ndarray
    twist: EgoTwist

    def scan(self, index: int) -> np.ndarray:
        return self.points[self.offsets[index] : self.offsets[index + 1]]


def load_scan_cache(config: dict[str, Any], run_entry: dict[str, Any]) -> ScanCache:
    path = scan_cache_path(config, run_entry["name"])
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run the decode stage for {run_entry['name']}")
    with np.load(path, allow_pickle=True) as data:
        if int(data["cache_version"][0]) != SCAN_CACHE_VERSION:
            raise ValueError(f"{path}: cache version {int(data['cache_version'][0])}, expected {SCAN_CACHE_VERSION}")
        stored = [list(map(str, row)) for row in data["source_fingerprints"].tolist()]
        wanted = [fingerprint(Path(run_entry[key])) for key in ("radar_mcap", "ego_odom_mcap", "tf_mcap")]
        if stored != wanted:
            raise ValueError(f"{path} was decoded from different bags than the config names; re-run decode")
        if str(data["radar_topic"]) != run_entry["radar_topic"]:
            raise ValueError(f"{path}: cached topic {data['radar_topic']} != {run_entry['radar_topic']}")
        if list(map(str, data["source_fields"])) != SOURCE_FIELDS:
            raise ValueError(f"{path}: cached fields {list(data['source_fields'])} != {SOURCE_FIELDS}")
        counts = data["point_counts"]
        return ScanCache(
            header_ns=data["header_ns"],
            log_ns=data["log_ns"],
            offsets=np.concatenate([[0], np.cumsum(counts)]),
            points=data["points"],
            msg_frame_ids=data["msg_frame_ids"],
            sensor_frame=str(data["tf_sensor_frame"]),
            t_ram_bag=data["tf_t_ram_sensor"],
            twist=EgoTwist(data["twist_t"], data["twist_linear"], data["twist_angular"]),
        )


# --------------------------------------------------------------------------------------
# stage 2: label
# --------------------------------------------------------------------------------------


@dataclass
class Frame:
    run: str
    split: str
    stamp_ns: int  # corrected acquisition time
    features: np.ndarray  # (N, 11) ego frame
    center: np.ndarray | None
    yaw: float | None
    points_in_box: int
    meta: dict[str, Any]


@dataclass
class SweepSource:
    """Every scan of a run inside the ego window, for writing as non-key sweeps.

    The model stacks the scans before each keyframe, and a live system always has them,
    so a keyframe's history must not depend on whether its neighbours were selected.
    Features are computed on demand: only scans that end up as sweeps need them.
    """

    stamps_ns: np.ndarray  # corrected acquisition time, ascending
    meta: list[dict[str, Any]]  # the scan's manifest entry, aligned with stamps_ns
    features: Callable[[int], np.ndarray]  # position in stamps_ns -> (N, 11) ego frame


def fixed_coverage_mask(xy: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    radius = np.linalg.norm(xy, axis=1)
    angle = np.arctan2(xy[:, 1], -xy[:, 0])
    return (
        (xy[:, 0] <= 0.0)
        & (radius >= float(cfg["min_range_m"]))
        & (radius <= float(cfg["max_range_m"]))
        & (np.abs(angle) <= math.radians(float(cfg["half_angle_deg"])))
    )


def target_coverage(center: np.ndarray, cfg: dict[str, Any]) -> tuple[bool, float, float]:
    radius = float(np.linalg.norm(center[:2]))
    angle = float(math.atan2(center[1], -center[0]))
    covered = (
        center[0] <= 0.0
        and float(cfg["min_range_m"]) <= radius <= float(cfg["max_range_m"])
        and abs(angle) <= math.radians(float(cfg["half_angle_deg"]))
    )
    return bool(covered), radius, math.degrees(angle)


def dealias_static_doppler(
    v_rel: np.ndarray, v_abs: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unwrap returns whose Doppler folded by one ambiguity interval.

    FRGen21 folds radial velocity with a 40 m/s period (unambiguous about +-20 m/s; its
    own resolver extends that only sometimes). Above ~20 m/s ego speed, stationary
    returns near boresight fold from ~+22 to ~-18 m/s and so read v_abs = -period, as
    if closing at 40 m/s. A return is unwrapped only when adding one period makes it
    stationary to within the tolerance: nothing on a race track truly moves at
    -period, and the opponent (v_abs ~ -its own speed) is nowhere near it.
    Returns (v_rel, v_abs, unwrapped mask).
    """
    period = float(cfg["period_mps"])
    folded = np.abs(v_abs + period) < float(cfg["static_tolerance_mps"])
    return v_rel + period * folded, v_abs + period * folded, folded


def scan_features(
    raw: np.ndarray,
    t_ram_sensor: np.ndarray,
    sensor_in_cog: np.ndarray,
    v_cog: np.ndarray,
    w_cog: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, int]:
    """Model input for one scan: SOURCE_FIELDS rows (sensor frame, finite) -> feature_names.

    The one place raw FRGen21 returns become model features.
    ``v_cog``/``w_cog`` are the ego twist at the scan's time; ``cfg`` is config["dataset"].
    Returns (features float32, number of Doppler-unwrapped returns).
    """
    coverage_cfg = cfg["coverage"]
    roi = np.asarray(coverage_cfg["point_cloud_range"], np.float64)
    xyz_sensor = raw[:, :3].astype(np.float64)
    xyz_ram = xyz_sensor @ t_ram_sensor[:3, :3].T + t_ram_sensor[:3, 3]
    keep = fixed_coverage_mask(xyz_ram[:, :2], coverage_cfg)
    keep &= (
        (xyz_ram[:, 0] >= roi[0]) & (xyz_ram[:, 1] >= roi[1]) & (xyz_ram[:, 2] >= roi[2])
        & (xyz_ram[:, 0] < roi[3]) & (xyz_ram[:, 1] < roi[4]) & (xyz_ram[:, 2] < roi[5])
    )
    raw, xyz_sensor, xyz_ram = raw[keep], xyz_sensor[keep], xyz_ram[keep]

    # Velocity of the radar itself; the rotation from cog to ram is identity.
    ego_v_ram = v_cog + np.cross(w_cog, sensor_in_cog)
    los_sensor = np.zeros_like(xyz_sensor)
    horizontal = np.linalg.norm(xyz_sensor[:, :2], axis=1)
    good = horizontal > 1e-3
    los_sensor[good, 0] = xyz_sensor[good, 0] / horizontal[good]
    los_sensor[good, 1] = xyz_sensor[good, 1] / horizontal[good]
    los_ram = los_sensor @ t_ram_sensor[:3, :3].T
    v_rel = raw[:, 3].astype(np.float64)
    v_abs = v_rel + float(cfg["velocity_compensation_sign"]) * (los_ram @ ego_v_ram)
    folded = 0
    dealias_cfg = cfg["doppler_dealias"]
    if dealias_cfg.get("enabled", True):
        v_rel, v_abs, mask = dealias_static_doppler(v_rel, v_abs, dealias_cfg)
        folded = int(mask.sum())
    features = np.column_stack(
        [
            xyz_ram,
            v_rel,
            v_abs,
            v_abs[:, None] * los_ram[:, :2],
            raw[:, 4],  # rcs
            raw[:, 6],  # snr
            raw[:, 5],  # confidence
            raw[:, 7],  # power
        ]
    ).astype(np.float32)
    return features, folded


def split_for(run: dict[str, Any], t: float, t0: float, t1: float, cfg: dict[str, Any]) -> str:
    policy = run["split"]
    if policy in ("train", "val", "test"):
        return policy
    if policy != "trainval":
        raise ValueError(f"unsupported split policy {policy!r}")
    fraction = float(run.get("train_fraction", cfg["train_fraction"]))
    boundary = t0 + fraction * (t1 - t0)
    half_guard = float(cfg["guard_seconds"]) * 0.5
    if t < boundary - half_guard:
        return "train"
    if t >= boundary + half_guard:
        return "val"
    return "guard"


def stable_priority(meta: dict[str, Any]) -> str:
    return hashlib.sha256(f"{meta['run']}:{meta['radar_stamp_ns']}".encode()).hexdigest()


def opponent_lateral_residual(
    features: np.ndarray, center: np.ndarray, yaw: float, car_lwh: list[float], gate: dict[str, Any]
) -> float | None:
    """Median across-box offset of the opponent's moving returns, or None if too few.

    Static returns sit near 0 m/s after ego compensation, so |v_abs| separates the
    opponent from walls and grass without needing the box to already be right.
    """
    length, width, _ = car_lwh
    along, across = box_frame(features, center, yaw)
    moving = (
        (np.abs(features[:, 4]) >= float(gate["moving_speed_mps"]))
        & (np.abs(along) <= length * 0.5 + float(gate["search_along_m"]))
        & (np.abs(across) <= width * 0.5 + float(gate["search_across_m"]))
    )
    if int(moving.sum()) < int(gate["min_moving_returns"]):
        return None
    return float(np.median(across[moving]))


def opponent_near_face_residual(
    features: np.ndarray, center: np.ndarray, yaw: float, car_lwh: list[float], gate: dict[str, Any]
) -> float | None:
    """Near-face shortfall of the opponent's moving returns, or None if not measurable.

    The radar sees the end of the car that faces it, so the furthest-toward-the-sensor
    moving return should sit near that face, L/2 from the centre. Returns that stop
    short of it (negative) or run past it (positive) mean the box is displaced along
    its own length. Only measured near tail-on or head-on, where that face is the one
    in view; at a broadside aspect the returns come from the side.
    """
    length, width, _ = car_lwh
    toward = -(math.cos(yaw) * center[0] + math.sin(yaw) * center[1])  # sensor, in box coordinates
    across_sensor = math.sin(yaw) * center[0] - math.cos(yaw) * center[1]
    if math.degrees(math.atan2(abs(across_sensor), abs(toward))) > float(gate["max_aspect_deg"]):
        return None
    along, across = box_frame(features, center, yaw)
    moving = (
        (np.abs(features[:, 4]) >= float(gate["moving_speed_mps"]))
        & (np.abs(along) <= length * 0.5 + float(gate["search_along_m"]))
        & (np.abs(across) <= width * 0.5 + float(gate["search_across_m"]))
    )
    if int(moving.sum()) < int(gate["min_moving_returns"]):
        return None
    return float(np.max(math.copysign(1.0, toward) * along[moving])) - length * 0.5


def lateral_qa_spans(frames: list[Frame], gate: dict[str, Any]) -> list[tuple[float, float, float]]:
    """Time spans where the label box sits persistently sideways of the opponent's returns.

    Found on Caltech (2026-09-21): four episodes, 1.5-10 s each, where the box sat
    ~1.1 m to one side of the returns while every other frame matched Laguna's
    residuals. Radar clock, ego heading and the fitted forward offset were each ruled
    out, which leaves the opponent's own /opponent/gps position; a persistent offset
    over seconds is its signature. Single-frame outliers are ordinary FRGen21 bearing
    noise, so the test is a rolling median, and only a span that persists for
    ``min_span_s`` is dropped. Returns ``(start_s, end_s, median_offset_m)`` spans.
    """
    return qa_spans(frames, gate, "lateral_residual_m")


def along_qa_spans(frames: list[Frame], gate: dict[str, Any]) -> list[tuple[float, float, float]]:
    """Time spans where the label box sits persistently displaced along its length.

    The along-track counterpart of ``lateral_qa_spans``. Found on Caltech (2026-09-28):
    in scene rear-013 the opponent's returns stopped ~1.3 m short of the box's near face
    for its whole close approach (21 m to 3 m), while every other Caltech scene and
    Laguna at the same ranges sat within ~0.3 m of it. The offset did not scale with
    opponent speed, so it is not a clock error. Same rolling median and minimum span.
    """
    return qa_spans(frames, gate, "near_face_residual_m")


def qa_spans(frames: list[Frame], gate: dict[str, Any], key: str) -> list[tuple[float, float, float]]:
    """Spans where the rolling median of ``meta[key]`` stays beyond ``max_offset_m``."""
    positives = sorted((f for f in frames if f.center is not None), key=lambda f: f.stamp_ns)
    t = np.array([f.stamp_ns * 1e-9 for f in positives])
    residual = np.array([np.nan if f.meta.get(key) is None else f.meta[key] for f in positives])
    half = float(gate["window_s"]) * 0.5
    rolling = np.full(len(positives), np.nan)
    for i in range(len(positives)):
        lo, hi = np.searchsorted(t, t[i] - half), np.searchsorted(t, t[i] + half, side="right")
        window = residual[lo:hi]
        window = window[np.isfinite(window)]
        if len(window) >= int(gate["min_window_frames"]):
            rolling[i] = np.median(window)
    flagged = np.flatnonzero(np.abs(rolling) > float(gate["max_offset_m"]))
    # Consecutive flagged positives, split wherever the positives themselves have a gap.
    groups = np.split(flagged, np.flatnonzero((np.diff(flagged) > 1) | (np.diff(t[flagged]) > 1.0)) + 1)
    return [
        (float(t[g[0]]), float(t[g[-1]]), float(np.median(rolling[g])))
        for g in groups
        if len(g) and t[g[-1]] - t[g[0]] >= float(gate["min_span_s"])
    ]


class Labeller:
    def __init__(self, config: dict[str, Any], geometry: Geometry):
        self.config = config
        self.cfg = config["dataset"]
        self.geometry = geometry
        self.pp = geometry.pp
        self.car_lwh = geometry.car_lwh
        self.feature_names = list(self.cfg["feature_names"])
        self.box_center_bounds = av24_geometry.box_center_bounds_in_ram(geometry.rig, self.car_lwh[0])

    def run_frames(
        self, run_entry: dict[str, Any], *, radar_time_offset: float | None = None
    ) -> tuple[list[Frame], list[dict[str, Any]], dict[str, Any], Any, SweepSource]:
        """Every scan of one run, labelled. Returns (selectable frames, manifest, report, ego, sweeps)."""
        cfg, pp = self.cfg, self.pp
        name = run_entry["name"]
        cache = load_scan_cache(self.config, run_entry)
        ego, opponent, opponent_cfg = self.geometry.timelines(run_entry)
        opponent_dt = opponent_time_offset(run_entry)
        ground = pp.GroundSurface(ego)
        t0 = max(ego.t_start, opponent.t_start)
        t1 = min(ego.t_end, opponent.t_end)

        t_ram_sensor, extrinsic = reconcile_sensor_extrinsic(
            self.geometry.rig,
            cache.t_ram_bag,
            run_entry["target_frame"],
            float(cfg["sensor_extrinsic_tolerance_m"]),
        )
        # Lever arm from the centre of gravity (where the twist is reported) to the radar.
        sensor_in_cog = (pp.T_COG_RAM @ np.append(t_ram_sensor[:3, 3], 1.0))[:3]
        time_source = str(run_entry.get("time_source", "log"))
        if time_source not in TIME_SOURCES:
            raise ValueError(f"{name}: time_source must be one of {TIME_SOURCES}")
        if radar_time_offset is None:
            radar_time_offset = float(run_entry.get("radar_time_offset_seconds", 0.0))
        offset_ns = int(round(radar_time_offset * 1e9))
        box_offset = float(run_entry.get("box_center_forward_offset_m", 0.0))
        low, high = self.box_center_bounds
        implied_center = av24_geometry.pp_box_center_in_ram(pp) + box_offset
        coverage_cfg = cfg["coverage"]
        min_ego_speed = float(cfg["min_ego_speed_mps"])
        gate = cfg["lateral_qa_gate"]
        along_gate = cfg["along_qa_gate"]
        # A car whose centre is beyond max_range can still have its near end, and its
        # returns, inside the point mask and the model's range. Background needs the
        # whole car out of reach, not just its centre.
        far_negative_range = float(coverage_cfg["max_range_m"]) + float(self.car_lwh[0])
        v_abs_col = self.feature_names.index("v_abs")
        moving_speed = float(cfg["moving_speed_mps"])
        min_moving = int(cfg["min_moving_points_in_box"])

        def features_at(index: int, t: float) -> tuple[np.ndarray, int, int, float]:
            """(ego-frame features, folded returns unwrapped, raw points, ego speed)."""
            raw = cache.scan(index)
            finite = np.isfinite(raw).all(axis=1)
            v_cog, w_cog = cache.twist.at(t)
            features, folded = scan_features(raw[finite], t_ram_sensor, sensor_in_cog, v_cog, w_cog, cfg)
            if features.shape[1] != len(self.feature_names):
                raise AssertionError((features.shape, self.feature_names))
            return features, folded, int(len(finite)), float(np.hypot(v_cog[0], v_cog[1]))

        alignment = np.array(
            [(h if time_source == "header" else l) + offset_ns for h, l in zip(cache.header_ns, cache.log_ns)],
            dtype=np.int64,
        )
        counts: Counter[str] = Counter()
        frames: list[Frame] = []
        manifest: list[dict[str, Any]] = []
        for index in range(len(cache.header_ns)):
            header_ns, log_ns = int(cache.header_ns[index]), int(cache.log_ns[index])
            alignment_ns = int(alignment[index])
            t = alignment_ns * 1e-9
            counts["scans_seen"] += 1
            meta: dict[str, Any] = {
                "run": name,
                "radar_stamp_ns": header_ns,
                "log_time_ns": log_ns,
                "alignment_stamp_ns": alignment_ns,
                "alignment_clock": "mcap_log_time" if time_source == "log" else "header_stamp",
                "radar_time_offset_seconds": radar_time_offset,
                "sensor_frame": str(cache.msg_frame_ids[index]),
                "selected": False,
            }
            if not (t0 <= t <= t1):
                meta["label_state"] = "outside_pose_window"
                counts["outside_pose_window"] += 1
                manifest.append(meta)
                continue
            split = split_for(run_entry, t, t0, t1, cfg["split"])
            features, folded, raw_count, ego_speed = features_at(index, t)
            counts["invalid_points"] += int((~np.isfinite(cache.scan(index)).all(axis=1)).sum())
            counts["doppler_unwrapped_points"] += folded
            counts["doppler_points_checked"] += int(len(features))
            meta.update(
                {
                    "split": split,
                    "raw_point_count": raw_count,
                    "filtered_point_count": int(len(features)),
                    "ego_speed_mps": ego_speed,
                }
            )
            if ego_speed < min_ego_speed:
                meta["label_state"] = "ego_slow"
            elif split == "guard":
                meta["label_state"] = "split_guard"
            elif not len(features):
                meta["label_state"] = "empty_scan"
            if "label_state" in meta:
                counts[meta["label_state"]] += 1
                manifest.append(meta)
                continue

            box = pp.opponent_box_at(t, ego, opponent, opponent_dt, ground=ground)
            if box is None:
                meta["label_state"] = "missing_pose"
                counts["missing_pose"] += 1
                manifest.append(meta)
                continue
            center, yaw = box
            if box_offset:
                # Longitudinal correction along the target's own heading; measured per run.
                center = center.copy()
                center[0] += box_offset * math.cos(yaw)
                center[1] += box_offset * math.sin(yaw)
            covered, target_range, target_angle = target_coverage(center, coverage_cfg)
            points_in_box = count_points_in_bev_car_box(features, center, yaw, self.car_lwh)
            meta.update(
                {
                    "target_center_xyz": center.tolist(),
                    "target_yaw_rad": float(yaw),
                    "target_range_m": target_range,
                    "target_angle_from_rear_deg": target_angle,
                    "covered_by_rear": covered,
                    "points_in_box": points_in_box,
                }
            )
            # Returns that are the opponent rather than whatever clutter the box happens
            # to cover: it is moving, and at race speed behind us its Doppler is large.
            moving_in_box = count_points_in_bev_car_box(
                features[np.abs(features[:, v_abs_col]) >= moving_speed], center, yaw, self.car_lwh
            )
            meta["moving_points_in_box"] = moving_in_box
            frame = Frame(name, split, alignment_ns, features, None, None, points_in_box, meta)
            if (covered and points_in_box >= int(cfg["min_points_in_box"])
                    and moving_in_box < min_moving):
                # The box's returns are stationary clutter; the car itself was not
                # detected. Not a positive, and the car is in view, so not background.
                meta["label_state"] = "ignored_no_moving_returns"
            elif covered and points_in_box >= int(cfg["min_points_in_box"]):
                meta["label_state"] = "Car"
                meta["lateral_residual_m"] = opponent_lateral_residual(features, center, yaw, self.car_lwh, gate)
                meta["near_face_residual_m"] = opponent_near_face_residual(
                    features, center, yaw, self.car_lwh, along_gate
                )
                frame.center, frame.yaw = center, float(yaw)
                frames.append(frame)
            elif covered:
                meta["label_state"] = "ignored_one_point" if points_in_box == 1 else "ignored_zero_points"
            elif center[0] >= 0.0 and points_in_box == 0:
                meta["label_state"] = "front_negative_candidate"
                frames.append(frame)
            elif center[0] >= 0.0:
                # A nominally front-sector car with rear-radar returns is exactly the
                # contamination case the coverage audit is meant to expose. Keep it in
                # the manifest for inspection, never as automatic background.
                meta["label_state"] = "front_negative_contaminated"
            elif target_range > far_negative_range and points_in_box == 0:
                meta["label_state"] = "far_negative_candidate"
                frames.append(frame)
            elif target_range > float(coverage_cfg["max_range_m"]):
                # Beyond labelling range but still partly within reach: neither a
                # positive nor trustworthy background.
                meta["label_state"] = "far_boundary_ignored"
            else:
                meta["label_state"] = "outside_rear_fov"
            counts[meta["label_state"]] += 1
            manifest.append(meta)

        report = {
            "multitrack_run": run_entry["multitrack_run"],
            "time_overlap_seconds": t1 - t0,
            "sensor_frame": cache.sensor_frame,
            "sensor_translation_in_rear_axle_middle": t_ram_sensor[:3, 3].tolist(),
            "sensor_rpy_in_rear_axle_middle_deg": np.degrees(
                Rotation.from_matrix(t_ram_sensor[:3, :3]).as_euler("xyz")
            ).tolist(),
            **extrinsic,
            "t_ram_sensor": t_ram_sensor.tolist(),
            "time_source": time_source,
            "radar_time_offset_seconds": radar_time_offset,
            "box_center_forward_offset_m": box_offset,
            "box_center_implied_x_in_rear_axle_middle": implied_center,
            "box_center_urdf_window_m": [low, high],
            "min_ego_speed_mps": min_ego_speed,
            "opponent_time_offset_seconds": opponent_dt,
            "upstream_opponent_time_offset_seconds": opponent_cfg.time_offset_seconds,
            "upstream_box_center_x_in_ram_m": av24_geometry.pp_box_center_in_ram(pp),
            "upstream_commit": self.geometry.upstream_commit,
            "ego_yaw_correction_deg": float(
                self.geometry.runs_by_name[run_entry["multitrack_run"]].ego_yaw_correction_deg
            ),
            "opponent_z_offset": opponent_cfg.z_offset,
            "opponent_map_offset_xyz": list(opponent_cfg.map_offset_xyz),
            "counts": dict(counts),
        }
        # Sweeps need an ego pose and twist, not an opponent, so they may come from
        # anywhere in the ego window -- including before the opponent's recording starts.
        if len(manifest) != len(alignment):
            raise AssertionError("every scan must have exactly one manifest entry")
        in_ego = np.flatnonzero((alignment * 1e-9 >= ego.t_start) & (alignment * 1e-9 <= ego.t_end))
        in_ego = in_ego[np.argsort(alignment[in_ego], kind="stable")]
        sweeps = SweepSource(
            stamps_ns=alignment[in_ego],
            meta=[manifest[i] for i in in_ego],
            features=lambda k: features_at(int(in_ego[k]), alignment[in_ego[k]] * 1e-9)[0],
        )
        return frames, manifest, report, ego, sweeps

    def select(self, frames: list[Frame], report: dict[str, Any]) -> list[Frame]:
        """Apply the lateral and along-track QA gates, then draw background to the configured fraction."""
        # Both gates' spans are found on the same positives before either drops any, so
        # neither gate's result depends on the other having run first.
        gates = [
            ("lateral_qa", "qa_lateral_offset", self.cfg["lateral_qa_gate"], lateral_qa_spans),
            ("along_qa", "qa_along_offset", self.cfg["along_qa_gate"], along_qa_spans),
        ]
        found = [(name, state, gate, find(frames, gate) if gate.get("enabled", True) else [])
                 for name, state, gate, find in gates]
        for name, state, gate, spans in found:
            dropped = 0
            for frame in frames:
                if frame.center is None:
                    continue
                t = frame.stamp_ns * 1e-9
                if any(start <= t <= end for start, end, _ in spans):
                    # Out of the dataset entirely: the opponent is in view, so this scan is
                    # not background either.
                    frame.meta["label_state"] = state
                    frame.center = None
                    dropped += 1
            report[name] = {
                "gate": gate,
                "dropped_positives": dropped,
                "spans": [
                    {"start_s": s, "end_s": e, "duration_s": e - s, "median_offset_m": m} for s, e, m in spans
                ],
            }
            report["counts"][state] = dropped
        positives = [f for f in frames if f.meta["label_state"] == "Car"]
        report["counts"]["Car"] = len(positives)

        selected = list(positives)
        fraction = float(self.cfg["negative_fraction"])
        by_split: dict[str, list[Frame]] = defaultdict(list)
        for frame in frames:
            if frame.meta["label_state"].endswith("negative_candidate"):
                by_split[frame.split].append(frame)
        positives_by_split = Counter(f.split for f in positives)
        report["positives_by_split"] = dict(positives_by_split)
        report["negatives_by_split"] = {}
        for split, candidates in by_split.items():
            limit = (
                int(math.floor(positives_by_split[split] * fraction / (1.0 - fraction)))
                if fraction < 1.0 else len(candidates)
            )
            chosen = sorted(candidates, key=lambda f: stable_priority(f.meta))[:limit]
            chosen_ids = {id(f) for f in chosen}
            for frame in candidates:
                if id(frame) not in chosen_ids:
                    frame.meta["label_state"] += "_not_selected"
            selected.extend(chosen)
            report["negatives_by_split"][split] = len(chosen)
        for frame in selected:
            frame.meta["selected"] = True
        return sorted(selected, key=lambda f: f.stamp_ns)


# --------------------------------------------------------------------------------------
# nuScenes writer
# --------------------------------------------------------------------------------------


def write_radar_pcd(path: Path, points: np.ndarray) -> None:
    fields = PCD_DTYPE.names
    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS " + " ".join(fields),
            "SIZE " + " ".join(str(PCD_DTYPE[f].itemsize) for f in fields),
            "TYPE " + " ".join(PCD_TYPE[PCD_DTYPE[f].kind] for f in fields),
            "COUNT " + " ".join("1" for _ in fields),
            f"WIDTH {len(points)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(points)}",
            "DATA binary",
        ]
    )
    # RadarPointCloud.from_file asserts every field ends strictly before the end of the
    # buffer, so a file whose data stops exactly at its last byte fails to load. The
    # trailing newline is that one byte.
    path.write_bytes(header.encode() + b"\n" + points.tobytes() + b"\n")


def radar_pcd_points(features: np.ndarray, names: list[str], t_ram_sensor: np.ndarray) -> np.ndarray:
    col = {name: i for i, name in enumerate(names)}
    rotation, translation = t_ram_sensor[:3, :3], t_ram_sensor[:3, 3]
    xyz_sensor = (features[:, :3].astype(np.float64) - translation) @ rotation  # R^T (p - t)
    # Horizontal line of sight in the sensor frame, exactly as the features were built.
    los = np.zeros((len(xyz_sensor), 2), dtype=np.float64)
    horizontal = np.linalg.norm(xyz_sensor[:, :2], axis=1)
    good = horizontal > 1e-3
    los[good] = xyz_sensor[good, :2] / horizontal[good, None]

    out = np.zeros(len(features), dtype=PCD_DTYPE)
    out["x"], out["y"], out["z"] = xyz_sensor.T
    out["dyn_prop"] = 4  # unknown
    out["id"] = np.arange(len(features))
    out["rcs"] = features[:, col["rcs"]]
    v_rel = features[:, col["v_rel"]].astype(np.float64)
    v_abs = features[:, col["v_abs"]].astype(np.float64)
    out["vx"], out["vy"] = (v_rel[:, None] * los).T
    out["vx_comp"], out["vy_comp"] = (v_abs[:, None] * los).T
    out["is_quality_valid"] = 1
    out["ambig_state"] = 3  # unambiguous
    out["invalid_state"] = 0  # valid
    return out


def split_scenes(frames: list[Frame], max_seconds: float, max_gap_seconds: float) -> list[list[Frame]]:
    scenes: list[list[Frame]] = []
    for frame in frames:
        if (
            not scenes
            or (frame.stamp_ns - scenes[-1][-1].stamp_ns) * 1e-9 > max_gap_seconds
            or (frame.stamp_ns - scenes[-1][0].stamp_ns) * 1e-9 >= max_seconds
        ):
            scenes.append([])
        scenes[-1].append(frame)
    return scenes


def assign_sweeps(keys_ns: np.ndarray, scans_ns: np.ndarray, max_lag_s: float) -> np.ndarray:
    """For each scan, the index into ``keys_ns`` (ascending) of the keyframe it is a sweep
    of, or -1. That is the first keyframe after it, if no more than ``max_lag_s`` later --
    nuScenes' convention, which puts every scan within the lag before a keyframe on the
    keyframe's own ``prev`` chain. Keyframes themselves are not sweeps."""
    keys_ns, scans_ns = np.asarray(keys_ns, np.int64), np.asarray(scans_ns, np.int64)
    owner = np.searchsorted(keys_ns, scans_ns, side="left")
    if not len(keys_ns):
        return np.full(len(scans_ns), -1, dtype=np.int64)
    exists = owner < len(keys_ns)
    next_key = keys_ns[np.minimum(owner, len(keys_ns) - 1)]
    lag_ns = int(round(max_lag_s * 1e9))
    valid = exists & (next_key > scans_ns) & (next_key - scans_ns <= lag_ns)
    return np.where(valid, owner, -1)


def link(records: list[dict[str, Any]]) -> None:
    for i, record in enumerate(records):
        record["prev"] = records[i - 1]["token"] if i else ""
        record["next"] = records[i + 1]["token"] if i + 1 < len(records) else ""


def attribute_for(track: list[dict[str, Any]], i: int) -> str | None:
    """Moving/stopped from the annotation track, the way the devkit derives box velocity;
    None when no neighbour is close enough in time to say."""
    t = lambda k: track[k]["t"]
    before = i - 1 if i and t(i) - t(i - 1) <= VELOCITY_MAX_DT else i
    after = i + 1 if i + 1 < len(track) and t(i + 1) - t(i) <= VELOCITY_MAX_DT else i
    if before == after:
        return None
    step = np.subtract(track[after]["center_map"][:2], track[before]["center_map"][:2])
    speed = float(np.linalg.norm(step)) / (t(after) - t(before))
    return "vehicle.moving" if speed >= MOVING_SPEED_MPS else "vehicle.stopped"


class NuScenesWriter:
    def __init__(self, root: Path, config: dict[str, Any]):
        self.root = root
        self.cfg = config["dataset"]
        self.feature_names = list(self.cfg["feature_names"])
        self.tables: dict[str, dict[str, list[dict[str, Any]]]] = {
            version: defaultdict(list) for version in sorted(set(SPLIT_VERSION.values()))
        }
        self.splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        self.maps: dict[tuple[str, str], dict[str, Any]] = {}
        self.counts: Counter[str] = Counter()
        self.attributes: Counter[str] = Counter()
        (root / "samples" / CHANNEL).mkdir(parents=True)
        (root / "samples" / FEATURE_DIR).mkdir(parents=True)
        (root / "maps").mkdir()
        for t in self.tables.values():
            t["sensor"].append({"token": token("sensor", CHANNEL), "channel": CHANNEL, "modality": "radar"})
            t["category"].append(
                {"token": token("category", CATEGORY), "name": CATEGORY,
                 "description": "Opponent race car (the single rear-radar Car class)."}
            )
            for name, description in (
                ("vehicle.moving", "Vehicle is moving."),
                ("vehicle.stopped", "Vehicle, with a driver/rider in/on it, is currently stationary but has an intent to move."),
                ("vehicle.parked", "Vehicle is stationary (usually for longer duration) with no immediate intent to move."),
            ):
                t["attribute"].append({"token": token("attribute", name), "name": name, "description": description})
            for level, (vis, description) in enumerate(
                (("v0-40", "visibility of whole object is between 0 and 40%"),
                 ("v40-60", "visibility of whole object is between 40 and 60%"),
                 ("v60-80", "visibility of whole object is between 60 and 80%"),
                 ("v80-100", "visibility of whole object is between 80 and 100%")),
                start=1,
            ):
                t["visibility"].append({"token": str(level), "level": vis, "description": description})

    def add_run(self, run_entry: dict[str, Any], frames: list[Frame], ego: Any, t_cog_ram: np.ndarray,
                t_ram_sensor: np.ndarray, sweeps: SweepSource) -> None:
        run = run_entry["name"]
        splits = {f.split for f in frames}
        if len(splits) != 1:
            # trainval policy: each split gets its own log so scenes never straddle splits.
            # The split guard is longer than the sweep lag, so no scan is a sweep of both.
            for split in sorted(splits):
                self.add_run({**run_entry, "name": f"{run}_{split}"}, [f for f in frames if f.split == split],
                             ego, t_cog_ram, t_ram_sensor, sweeps)
            return
        split = splits.pop()
        version = SPLIT_VERSION[split]
        t = self.tables[version]
        location = run_entry["location"]

        log_token = token("log", run)
        t["log"].append(
            {
                "token": log_token,
                "logfile": run,
                "vehicle": "av24",
                "date_captured": datetime.fromtimestamp(frames[0].stamp_ns * 1e-9, tz=timezone.utc).strftime("%Y-%m-%d"),
                "location": location,
            }
        )
        self.maps.setdefault(
            (version, location),
            {"token": token("map", version, location), "log_tokens": [], "category": "semantic_prior", "filename": ""},
        )["log_tokens"].append(log_token)
        calibrated_token = token("calibrated_sensor", run, CHANNEL)
        t["calibrated_sensor"].append(
            {
                "token": calibrated_token,
                "sensor_token": token("sensor", CHANNEL),
                "translation": t_ram_sensor[:3, 3].tolist(),
                "rotation": quat_wxyz(t_ram_sensor[:3, :3]),
                "camera_intrinsic": [],
            }
        )

        def write_scan(stamp_ns: int, features: np.ndarray, sample_token: str, is_key_frame: bool) -> dict[str, Any]:
            """The scan's files, ego_pose and (unlinked) sample_data record."""
            timestamp = stamp_ns // 1000
            stem = f"{run}__{CHANNEL}__{timestamp}"
            pose_token = token("ego_pose", run, stamp_ns)
            t_map_ram = ego.pose_at(stamp_ns * 1e-9) @ t_cog_ram
            t["ego_pose"].append(
                {"token": pose_token, "timestamp": timestamp,
                 "rotation": quat_wxyz(t_map_ram[:3, :3]), "translation": t_map_ram[:3, 3].tolist()}
            )
            write_radar_pcd(self.root / "samples" / CHANNEL / f"{stem}.pcd",
                            radar_pcd_points(features, self.feature_names, t_ram_sensor))
            np.save(self.root / "samples" / FEATURE_DIR / f"{stem}.npy", features)
            return {
                "token": token("sample_data", run, stamp_ns, CHANNEL),
                "sample_token": sample_token,
                "ego_pose_token": pose_token,
                "calibrated_sensor_token": calibrated_token,
                "timestamp": timestamp,
                "fileformat": "pcd",
                "is_key_frame": is_key_frame,
                "height": 0,
                "width": 0,
                "filename": f"samples/{CHANNEL}/{stem}.pcd",
            }

        # Every scan within the stacking window before a keyframe is written as a sweep
        # of it, so background keyframes get the same history as positives (before
        # 2026-10-02 only selected scans were written, and the sweep chain skipped
        # straight to the previous *selected* scan: positives sit in dense runs and got
        # full stacks, sparse background mostly got none).
        keys_ns = np.array([f.stamp_ns for f in frames], dtype=np.int64)
        if np.any(np.diff(keys_ns) <= 0):
            raise AssertionError(f"{run}: keyframes must be in strictly increasing time order")
        owner = assign_sweeps(keys_ns, sweeps.stamps_ns, float(self.cfg["sweeps"]["max_lag_s"]))
        sweeps_of: dict[int, list[int]] = defaultdict(list)
        for k in np.flatnonzero(owner >= 0):
            sweeps_of[int(owner[k])].append(int(k))
        key_index = {id(f): i for i, f in enumerate(frames)}

        scenes = split_scenes(frames, float(self.cfg["scene_max_seconds"]), float(self.cfg["scene_max_gap_seconds"]))
        for scene_index, scene_frames in enumerate(scenes):
            scene_name = f"{run}-{scene_index:03d}"
            scene_token = token("scene", scene_name)
            samples, sample_datas, track = [], [], []
            for frame in scene_frames:
                sample_token = token("sample", run, frame.stamp_ns)
                timestamp = frame.stamp_ns // 1000
                for k in sweeps_of.get(key_index[id(frame)], []):
                    stamp_ns = int(sweeps.stamps_ns[k])
                    sample_datas.append(write_scan(stamp_ns, sweeps.features(k), sample_token, False))
                    sweeps.meta[k]["sweep_of_sample_token"] = sample_token
                    self.counts[f"{split}_sweeps"] += 1
                samples.append({"token": sample_token, "timestamp": timestamp, "scene_token": scene_token})
                sample_datas.append(write_scan(frame.stamp_ns, frame.features, sample_token, True))
                frame.meta["sample_token"] = sample_token
                t_map_ram = ego.pose_at(frame.stamp_ns * 1e-9) @ t_cog_ram

                if frame.center is not None:
                    length, width, height = map(float, self.cfg["car_size_lwh"])
                    center_map = t_map_ram[:3, :3] @ frame.center + t_map_ram[:3, 3]
                    rotation_map = t_map_ram[:3, :3] @ Rotation.from_euler("z", frame.yaw).as_matrix()
                    track.append(
                        {
                            "token": token("sample_annotation", run, frame.stamp_ns, 0),
                            "sample_token": sample_token,
                            "t": frame.stamp_ns * 1e-9,
                            "center_map": center_map,
                            "translation": center_map.tolist(),
                            "size": [width, length, height],  # nuScenes wlh
                            "rotation": quat_wxyz(rotation_map),
                            "num_radar_pts": int(frame.points_in_box),
                        }
                    )

            link(samples)
            link(sample_datas)
            t["sample"].extend(samples)
            t["sample_data"].extend(sample_datas)
            t["scene"].append(
                {
                    "token": scene_token,
                    "log_token": log_token,
                    "nbr_samples": len(samples),
                    "first_sample_token": samples[0]["token"],
                    "last_sample_token": samples[-1]["token"],
                    "name": scene_name,
                    "description": f"{split}; rear FRGen21 radar; {run}",
                }
            )
            self.splits[split].append(scene_name)
            self.counts[f"{split}_scenes"] += 1
            self.counts[f"{split}_samples"] += len(samples)
            if not track:
                continue
            # The run's single opponent is one physical car for the whole scene, so it is
            # one instance; samples where it is out of view simply have no annotation.
            instance_token = token("instance", scene_name, 0)
            link(track)
            for i, ann in enumerate(track):
                attribute = attribute_for(track, i)
                self.attributes[attribute or "none"] += 1
                t["sample_annotation"].append(
                    {
                        "token": ann["token"],
                        "sample_token": ann["sample_token"],
                        "instance_token": instance_token,
                        "visibility_token": "",
                        "attribute_tokens": [token("attribute", attribute)] if attribute else [],
                        "translation": ann["translation"],
                        "size": ann["size"],
                        "rotation": ann["rotation"],
                        "prev": ann["prev"],
                        "next": ann["next"],
                        "num_lidar_pts": 0,
                        "num_radar_pts": ann["num_radar_pts"],
                    }
                )
            t["instance"].append(
                {
                    "token": instance_token,
                    "category_token": token("category", CATEGORY),
                    "nbr_annotations": len(track),
                    "first_annotation_token": track[0]["token"],
                    "last_annotation_token": track[-1]["token"],
                }
            )
            self.counts[f"{split}_annotations"] += len(track)

    def finish(self) -> None:
        for (version, _), record in self.maps.items():
            self.tables[version]["map"].append(record)
        for version, t in self.tables.items():
            (self.root / version).mkdir()
            for table in NUSCENES_TABLES:
                (self.root / version / f"{table}.json").write_text(json.dumps(t[table], indent=0))
        (self.root / "splits.json").write_text(json.dumps(self.splits, indent=2) + "\n")


class FeatureStats:
    def __init__(self, names: list[str]):
        self.names = names
        self.count = 0
        self.sum = np.zeros(len(names), dtype=np.float64)
        self.sumsq = np.zeros(len(names), dtype=np.float64)
        self.min = np.full(len(names), np.inf)
        self.max = np.full(len(names), -np.inf)

    def update(self, points: np.ndarray) -> None:
        if not len(points):
            return
        x = np.asarray(points, dtype=np.float64)
        self.count += len(x)
        self.sum += x.sum(axis=0)
        self.sumsq += np.square(x).sum(axis=0)
        self.min = np.minimum(self.min, x.min(axis=0))
        self.max = np.maximum(self.max, x.max(axis=0))

    def as_dict(self) -> dict[str, Any]:
        if not self.count:
            return {"feature_names": self.names, "point_count": 0}
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - np.square(mean), 0.0)
        return {"feature_names": self.names, "point_count": self.count, "mean": mean.tolist(),
                "std": np.sqrt(var).tolist(), "min": self.min.tolist(), "max": self.max.tolist()}


def scene_starts(tables: Iterable[dict[str, list[dict[str, Any]]]]) -> dict[str, int]:
    """Scene name -> timestamp of its first sample."""
    starts = {}
    for t in tables:
        sample_time = {s["token"]: s["timestamp"] for s in t["sample"]}
        starts.update({scene["name"]: sample_time[scene["first_sample_token"]] for scene in t["scene"]})
    return starts


def carry_splits(writer: "NuScenesWriter", old_splits: Path) -> list[str]:
    """Copy an earlier build's extra split keys, refusing if any scene has moved.

    ``old_splits`` is either an earlier build's ``splits.json`` (scene start times are read
    from that build's tables) or a standalone file holding ``{"splits": ..., "scene_starts": ...}``
    such as ``splits/carried_splits.json``, which needs no earlier build on disk.
    """
    old = json.loads(old_splits.read_text())
    if "scene_starts" in old:
        old_starts = old["scene_starts"]
        old = old["splits"]
    else:
        old_root = old_splits.parent
        old_starts = scene_starts(
            {name: json.loads((old_root / version / f"{name}.json").read_text()) for name in ("scene", "sample")}
            for version in sorted(set(SPLIT_VERSION.values()))
        )
    new_starts = scene_starts(writer.tables.values())
    carried = []
    for key, scenes in old.items():
        if key in writer.splits:
            continue
        moved = [s for s in scenes if new_starts.get(s) != old_starts.get(s)]
        if moved:
            raise SystemExit(f"carry_splits_from: split {key!r} names {len(moved)} scenes that are missing or start "
                             f"at a different time in this build (e.g. {moved[0]}); regenerate that split")
        writer.splits[key] = list(scenes)
        carried.append(key)
    return carried


def label(config: dict[str, Any], runs: list[dict[str, Any]], overwrite: bool) -> None:
    root = Path(config["dataset"]["output_root"])
    if root.exists():
        if not overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    geometry = Geometry(config)
    labeller = Labeller(config, geometry)
    writer = NuScenesWriter(root, config)
    stats = FeatureStats(labeller.feature_names)
    report: dict[str, Any] = {"runs": {}}
    manifest: list[dict[str, Any]] = []
    for run_entry in runs:
        print(f"\n[label] {run_entry['name']}", flush=True)
        frames, run_manifest, run_report, ego, sweeps = labeller.run_frames(run_entry)
        selected = labeller.select(frames, run_report)
        writer.add_run(run_entry, selected, ego, np.asarray(geometry.pp.T_COG_RAM, np.float64),
                       np.asarray(run_report["t_ram_sensor"]), sweeps)
        for frame in selected:
            if frame.split == "train":
                stats.update(frame.features)
        report["runs"][run_entry["name"]] = run_report
        manifest.extend(run_manifest)
        qa, along = run_report["lateral_qa"], run_report["along_qa"]
        print(
            f"  positives={run_report['positives_by_split']} negatives={run_report['negatives_by_split']} "
            f"qa_dropped={qa['dropped_positives']} in {len(qa['spans'])} spans, "
            f"along_qa_dropped={along['dropped_positives']} in {len(along['spans'])} spans",
            flush=True,
        )
    carry_from = config["dataset"].get("carry_splits_from")
    if carry_from:
        print(f"\n[label] carried splits {carry_splits(writer, Path(carry_from))} from {carry_from}", flush=True)
    writer.finish()
    manifest.sort(key=lambda item: (item["run"], item["radar_stamp_ns"]))
    with (root / "manifest.jsonl").open("w") as stream:
        for item in manifest:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    (root / "feature_stats.json").write_text(json.dumps(stats.as_dict(), indent=2) + "\n")
    report["dataset"] = {
        "output_root": str(root),
        "format": "nuscenes-v1.0",
        "channel": CHANNEL,
        "feature_dir": FEATURE_DIR,
        "feature_names": labeller.feature_names,
        "feature_frame": "rear_axle_middle (ego)",
        "pcd_fields": list(PCD_DTYPE.names),
        "car_size_lwh": labeller.car_lwh,
        "min_points_in_box": int(config["dataset"]["min_points_in_box"]),
        "min_moving_points_in_box": int(config["dataset"]["min_moving_points_in_box"]),
        "moving_speed_mps": float(config["dataset"]["moving_speed_mps"]),
        "negative_fraction": float(config["dataset"]["negative_fraction"]),
        "sweep_max_lag_s": float(config["dataset"]["sweeps"]["max_lag_s"]),
        "versions": {v: sorted({s for s, vv in SPLIT_VERSION.items() if vv == v}) for v in writer.tables},
        "counts": dict(writer.counts),
        "attributes": dict(writer.attributes),
        "manifest_scans": len(manifest),
    }
    (root / "dataset_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n[done] {root}\n  {json.dumps(dict(writer.counts))}", flush=True)


# --------------------------------------------------------------------------------------
# diagnostic: radar clock sweep
# --------------------------------------------------------------------------------------


def sweep_clock(config: dict[str, Any], run_entry: dict[str, Any], lo: float, hi: float, step: float) -> None:
    """Score candidate ``radar_time_offset_seconds`` values for one run.

    A radar clock error moves the ego *and* opponent poses together, so it barely
    shifts the box along-track (the two cars travel together) but rotates it about
    the ego in corners: ``range * yaw_rate * dt`` sideways. The score is the spread of
    the opponent's moving returns about the box -- the median absolute deviation of
    their along- and across-box medians plus the across-box bias -- which is invariant
    to the constant ``box_center_forward_offset_m``. Positives are selected once, at
    the configured offset, so every candidate is scored on the same scans.
    """
    geometry = Geometry(config)
    labeller = Labeller(config, geometry)
    pp = geometry.pp
    frames, _, _, _, _ = labeller.run_frames(run_entry)
    frames = [f for f in frames if f.center is not None]
    gate = config["dataset"]["lateral_qa_gate"]
    ego, opponent, opponent_cfg = geometry.timelines(run_entry)
    configured = float(run_entry.get("radar_time_offset_seconds", 0.0))
    box_offset = float(run_entry.get("box_center_forward_offset_m", 0.0))
    center_in_cog = np.r_[pp.OPP_BOX_CENTER_IN_RAM - pp.COG_IN_RAM, 1.0]
    t_base = np.array([f.stamp_ns * 1e-9 for f in frames]) - configured
    returns = []
    for f in frames:
        along, across = box_frame(f.features, f.center, f.yaw)
        moving = (np.abs(f.features[:, 4]) >= float(gate["moving_speed_mps"])) & (np.hypot(along, across) < 8.0)
        returns.append(f.features[moving, :2].astype(np.float64) if moving.sum() >= 3 else None)
    keep = [i for i, r in enumerate(returns) if r is not None]
    yaw_of = lambda m: np.arctan2(m[..., 1, 0], m[..., 0, 0])
    mad = lambda v: float(np.median(np.abs(v - np.median(v))))
    print(f"\n[sweep-clock] {run_entry['name']}: {len(keep)} positives with >=3 moving returns, "
          f"configured offset {configured:+.3f} s")
    results = []
    for dt in np.round(np.arange(lo, hi + 1e-9, step), 4):
        t = t_base[keep] + dt
        t_map_ram = ego.pose_at(t) @ pp.T_COG_RAM
        t_opp = opponent.pose_at(t + opponent_time_offset(run_entry))
        center = np.einsum("nij,nj->ni", np.linalg.inv(t_map_ram), t_opp @ center_in_cog)[:, :2]
        yaw = yaw_of(t_opp[:, :3, :3]) - yaw_of(t_map_ram[:, :3, :3])
        center += box_offset * np.column_stack([np.cos(yaw), np.sin(yaw)])
        along_med, across_med = [], []
        for k, i in enumerate(keep):
            along, across = box_frame(returns[i], center[k], float(yaw[k]))
            along_med.append(np.median(along))
            across_med.append(np.median(across))
        across_med = np.asarray(across_med)
        score = mad(np.asarray(along_med)) + mad(across_med) + abs(float(np.median(across_med)))
        results.append((score, dt))
        print(f"  dt={dt:+.3f}s  score={score:.4f} m  |across|>1m {np.mean(np.abs(across_med) > 1):.1%}", flush=True)
    best = min(results)
    print(f"  best dt={best[1]:+.3f}s (score {best[0]:.4f} m); configured {configured:+.3f}s scores "
          f"{min(results, key=lambda r: abs(r[1] - configured))[0]:.4f} m")


# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["decode", "label", "all", "sweep-clock"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs", nargs="*", help="run names; defaults to every configured run")
    parser.add_argument("--run", help="sweep-clock: the run to sweep")
    parser.add_argument("--sweep", nargs=3, type=float, default=[-0.3, 0.3, 0.02], metavar=("LO", "HI", "STEP"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    runs = config["runs"]
    if args.runs:
        requested = set(args.runs)
        runs = [run for run in runs if run["name"] in requested]
        missing = requested - {run["name"] for run in runs}
        if missing:
            raise SystemExit(f"unknown runs: {sorted(missing)}")
    if args.stage == "sweep-clock":
        entry = next((r for r in config["runs"] if r["name"] == args.run), None)
        if entry is None:
            raise SystemExit(f"--run must name a configured run, got {args.run!r}")
        sweep_clock(config, entry, *args.sweep)
        return
    if args.stage in ("decode", "all"):
        geometry = Geometry(config)
        for run in runs:
            decode_run(config, geometry, run)
    if args.stage in ("label", "all"):
        if args.runs and set(args.runs) != {r["name"] for r in config["runs"]}:
            raise SystemExit("label writes the whole dataset; run it without --runs")
        label(config, runs, args.overwrite)


if __name__ == "__main__":
    main()
