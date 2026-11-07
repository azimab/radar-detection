#!/usr/bin/env python3
"""Shared Putnam Park -> KITTI helpers for PointPillars.

All geometry here uses the fused lidar frame as KITTI velodyne:
``rear_axle_middle``.  The odometry bags publish ``map -> center_of_gravity``.
"""
from __future__ import annotations

import argparse
import math
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parent

# Pipeline stages, in the order "all" runs them. "check" and "probe" are read-only data
# gates: check classifies each run's lidar rig against the URDF, probe measures GT boxes
# against real points. Both run before "data" under "all" so a bad source is caught before
# a multi-hour decode, not after.
# "infos" regenerates the info pickles and gt_database from whatever ImageSets currently
# says, without rebuilding the KITTI tree. It is deliberately absent from "all" -- the
# data stage already does it -- and exists because "data" honours dataset.clean, so
# refreshing a pickle through it costs a 5 h re-decode and destroys the labels first.
STAGES = ("all", "check", "probe", "data", "infos", "train", "eval", "visualize", "export")


def _read_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _base_dataset_cfg(model_cfg: dict, model_cfg_path: Path) -> dict:
    base = model_cfg.get("DATA_CONFIG", {}).get("_BASE_CONFIG_")
    if not base:
        return {}
    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = TOOLS_DIR / base_path
    return _read_yaml(base_path)


def _merged_data_config(model_cfg: dict, base_cfg: dict) -> dict:
    merged = dict(base_cfg)
    merged.update(model_cfg.get("DATA_CONFIG", {}))
    return merged


def _car_anchor(model_cfg: dict) -> dict:
    anchors = model_cfg["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"]
    return next((a for a in anchors if a.get("class_name") == "Car"), anchors[0])


def _min_points_for_class(data_cfg: dict, class_name: str = "Car") -> int:
    augmentors = data_cfg.get("DATA_AUGMENTOR", {}).get("AUG_CONFIG_LIST", [])
    for augmentor in augmentors:
        for item in augmentor.get("PREPARE", {}).get("filter_by_min_points", []):
            name, value = str(item).split(":", 1)
            if name == class_name:
                return int(value)
    return 1


def load_model_settings(cfg_file: Path) -> tuple[np.ndarray, tuple[float, float, float], float, int]:
    cfg_file = Path(cfg_file)
    model_cfg = _read_yaml(cfg_file)
    data_cfg = _merged_data_config(model_cfg, _base_dataset_cfg(model_cfg, cfg_file))
    anchor = _car_anchor(model_cfg)
    point_range = np.asarray(data_cfg["POINT_CLOUD_RANGE"], dtype=np.float64)
    car_lwh = tuple(float(x) for x in anchor["anchor_sizes"][0])
    anchor_bottom = float(anchor["anchor_bottom_heights"][0])
    min_points = _min_points_for_class(data_cfg)
    return point_range, car_lwh, anchor_bottom, min_points


COG_IN_RAM = np.array([1.3206, -0.030188, 0.23598], dtype=np.float64)
RAM_Z_ABOVE_GROUND = 0.198

# How far the opponent's bodywork extends BEHIND its own rear axle.
#
# The box centre is rear_axle + (CAR_L/2 - this). Until 2026-09-26 this term was absent,
# i.e. the rear overhang was implicitly 0 and the box's rear face was pinned exactly to
# the rear axle -- so every box in every run sat ~1 m too far forward. That is inherited
# from the original pipeline (5a3513f / 1bbfda5), so the deployed model carries it too.
#
# Measured 2026-09-26 off the built dataset, four independent ways that need no clock:
#   * the forward model reproduces written labels to 0.4 mm in x -> the offset really is
#     exactly CAR_L/2;
#   * av24.urdf puts radar_rear at x = -0.498 m, i.e. behind the axle, inside the rear
#     bodywork;
#   * on runs where the whole car is visible (Laguna cluster span 4.92 m against a true
#     4.876) points fall OUTSIDE the box's rear face while its front third is empty --
#     self-occlusion moves a centroid, it cannot do both at once;
#   * a cross-run fit that removes each run's speed-dependent clock term puts the centre
#     at +1.33 to +1.46 m from the axle, against +2.438 as configured.
#
# 1.00 m is the middle of that band (box centre +1.44 m, rear face -1.00, nose +3.88).
# Uncertainty is about +-0.2 m and comes from the spatial/clock degeneracy; the sign and
# the ~1 m magnitude are not in doubt. REPLACE THIS WITH THE CHASSIS DRAWING when someone
# has it -- this is a chassis constant and should not be fitted from racing data.
OPP_REAR_OVERHANG = 1.00
REFL_SCALE = 0.5
LIDAR_FRAME = "rear_axle_middle"

# GT boxes are always computed in LIDAR_FRAME (see opponent_box_from_poses), but fused
# clouds do not all arrive in that frame: on-car iris_fusion_node publishes
# rear_axle_middle, while the offline iris_fuse*.py scripts target
# rear_axle_middle_ground. tf_static (and av24.urdf) put rear_axle_middle_ground exactly
# RAM_Z_ABOVE_GROUND below rear_axle_middle, so a point expressed in RAMG needs -0.198 to
# become RAM. Previously this was ignored, silently putting every Caltech and IU GT box
# 0.198 m too high relative to its cloud. Values here are the z shift ADDED to incoming
# points to bring them into LIDAR_FRAME.
CLOUD_FRAME_Z_OFFSET = {
    "rear_axle_middle": 0.0,
    "rear_axle_middle_ground": -RAM_Z_ABOVE_GROUND,
}


def cloud_frame_z_offset(frame_id: str) -> float:
    """z shift bringing a cloud in ``frame_id`` into ``LIDAR_FRAME``.

    Raises on an unknown frame rather than assuming one: guessing the vertical datum is
    exactly the failure mode this guards against.
    """
    key = str(frame_id).strip().lstrip("/")
    if key not in CLOUD_FRAME_Z_OFFSET:
        raise ValueError(
            f"unrecognized lidar cloud frame_id {frame_id!r}; GT boxes are computed in "
            f"{LIDAR_FRAME!r} and the vertical datum will not be guessed. "
            f"known frames: {sorted(CLOUD_FRAME_Z_OFFSET)}. If {frame_id!r} is valid, add "
            "it to CLOUD_FRAME_Z_OFFSET with its measured offset."
        )
    return CLOUD_FRAME_Z_OFFSET[key]

POINT_CLOUD_RANGE = np.zeros(6, dtype=np.float64)
CAR_L = CAR_W = CAR_H = ANCHOR_BOTTOM_HEIGHT = 0.0
MIN_POINTS_IN_OPP = 0

CALIB_TEXT = (
    "P0: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "P1: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "P2: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "P3: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    "R0_rect: 1 0 0 0 1 0 0 0 1\n"
    "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n"
    "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0\n"
)

T_COG_RAM = np.eye(4, dtype=np.float64)
T_COG_RAM[:3, 3] = -COG_IN_RAM
# Placeholder only: CAR_L is 0 until configure_from_model_yaml runs, which overwrites
# this. Anything computing a box MUST call pp.configure_from_run_config() first.
OPP_BOX_CENTER_IN_RAM = np.array([0.0, 0.0, COG_IN_RAM[2]], dtype=np.float64)


@dataclass(frozen=True)
class Opponent:
    """One opponent car's ground-truth pose source within a run.

    A run has as many of these as there were other cars on track with published GPS.
    Each carries its own clock offset and vertical datum because each comes from a
    different team's localization stack; nothing may be shared between them by default.
    """

    name: str
    odom: list[Path]
    odom_topic: str
    time_offset_seconds: float
    # Constant vertical datum offset subtracted from this opponent's z, for opponents
    # logged in a different vertical reference than ours (e.g. Laguna Seca: our odometry
    # is in the exwayz local map frame, Purdue's /odometry/global_filtered is absolute
    # altitude, differing by ~201 m). 0.0 when both share a datum.
    z_offset: float = 0.0
    # Rigid correction ADDED to this opponent's position, bringing its localization into
    # our map frame. Unlike z_offset (a single subtracted datum constant) this is measured
    # rather than assumed: pp_align_datum.py fits each car's odometry against that car's
    # own GNSS, then composes the two, so neither fit depends on the other team's
    # conventions and there is no translation/clock degeneracy to resolve.
    map_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class DatasetRun:
    name: str
    ego_odom: list[Path]
    fused_lidar: list[Path]
    opponents: list[Opponent]
    # Take opponent box ground heights from the ego's own driven surface instead of the
    # opponents' reported z. See GroundSurface: the opponent's z is the least trustworthy
    # channel of its pose.
    ground_from_ego: bool = True
    # True when the run publishes only per-sensor /pcl_front|left|right and the fused
    # cloud must be built in memory at ingest (see iter_fused_raw_frames). All IMS/LVMS
    # and Laguna runs are like this.
    fuse_raw: bool = False
    # Bags to resolve /tf_static from when fusing. Defaults to ego_odom + fused_lidar,
    # which covers both layouts seen so far (tf_static in the lidar bag, as at IMS/LVMS,
    # or only in the no_lidar bag, as at Laguna Seca).
    tf_static: list[Path] = field(default_factory=list)
    # False keeps a run configured and probeable but out of the built dataset. Preferred
    # over deleting or commenting it out: a blocked run's paths, offsets and findings stay
    # in one place, and re-enabling it is a one-word change once the block clears.
    enabled: bool = True

    # Yaw correction applied to the EGO pose, degrees, positive = counter-clockwise.
    #
    # Mathematically identical to re-rotating `lidar_plate`: all three iris sensors hang
    # off that plate, so a plate yaw error and an ego yaw error produce the same thing --
    # opponent boxes rotated about the ego, i.e. a lateral error growing linearly with
    # range. One knob therefore covers both, and it is the only correction that can fix a
    # long-range lateral error, since a fixed lever arm cannot produce one.
    #
    # MEASURE IT, do not guess: fit lateral offset (box centre -> real point cluster,
    # resolved in the box frame) against range, over head-on/tail-on views at 30-80 m.
    # Slope constant across the run => extrinsic, this knob is the right fix. Slope
    # drifting => ego localization heading, and no static correction helps.
    # The known-good control pp_2026_04_28_run2 measures -0.04 deg, which is what
    # validates the method.
    ego_yaw_correction_deg: float = 0.0

    # --- single-opponent compatibility -----------------------------------------------
    # visualize.py renders one opponent at a time; these keep it, and any other
    # single-object consumer, working against the first configured opponent.
    @property
    def opp_odom(self) -> list[Path]:
        return self.opponents[0].odom

    @property
    def opp_odom_topic(self) -> str:
        return self.opponents[0].odom_topic

    @property
    def opponent_time_offset_seconds(self) -> float:
        return self.opponents[0].time_offset_seconds

    @property
    def tf_static_paths(self) -> list[Path]:
        return self.tf_static if self.tf_static else [*self.ego_odom, *self.fused_lidar]

    @property
    def opponent_source_paths(self) -> list[Path]:
        return [p for opp in self.opponents for p in opp.odom]


@dataclass(frozen=True)
class RunConfig:
    path: Path
    raw: dict[str, Any]
    run_name: str
    repo_dir: Path
    container_image: Path
    container_binds: list[Path]
    log_dir: Path
    data_root: Path
    ego_odom_topic: str
    fused_lidar_topic: str
    runs: list[DatasetRun]
    model_cfg: Path
    checkpoint: Path | None
    pretrained_model: Path | None
    clean_dataset: bool
    cache_enabled: bool
    cache_dir: Path
    build_max_frames: int
    build_stride: int
    min_points_in_opponent_box: Optional[int]
    opponent_range_gate_m: float
    far_sample_seconds: float
    far_sample_period_seconds: float
    empty_train_fraction: float
    empty_val_fraction: float
    chunk_seconds: float
    guard_seconds: float
    train_fraction: float
    test_fraction: float
    batch_size: int
    epochs: int
    workers: int
    eval_batch_size: int
    eval_score_thresh: float
    probe_max_frames: int
    probe_window_seconds: float
    probe_runs: list[str]
    visualize_start_seconds: Optional[float]
    visualize_duration_seconds: float
    visualize_stride: int
    visualize_run_index: int
    visualize_out: Path
    pred_topic: str
    onnx_out_dir: Path
    params_h: Path
    slurm_partition: str
    slurm_setup_gres: str
    slurm_pipeline_gres: str
    slurm_setup_cpus: int
    slurm_pipeline_cpus: int
    slurm_setup_mem: str
    slurm_pipeline_mem: str
    slurm_setup_time: str
    slurm_pipeline_time: str


def _require(section: dict[str, Any], key: str, dotted: str) -> Any:
    value = section.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required config value: {dotted}")
    return value


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required config section: {name}")
    return value


def _path(value: Any, base: Path | None = None) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return p if p.is_absolute() or base is None else base / p


def _path_list(value: Any, dotted: str) -> list[Path]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{dotted} must be a non-empty YAML list")
    return [_path(v) for v in value]


def _absolute_path(value: Any, dotted: str) -> Path:
    path = _path(value)
    if not path.is_absolute():
        raise ValueError(f"{dotted} must be an absolute path")
    return path


def _check_index(index: int, size: int, dotted: str) -> None:
    if index < 0 or index >= size:
        raise ValueError(f"{dotted}={index} is out of range for {size} configured run(s)")


def _xyz(value, dotted: str) -> tuple[float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{dotted} must be a list of exactly 3 numbers [dx, dy, dz]")
    return tuple(float(v) for v in value)


def _run_opponents(item: dict[str, Any], prefix: str) -> list[Opponent]:
    """Parse a run's opponents from either the list form or the single-opponent keys.

    The single-opponent keys (``opp_odom``/``opp_odom_topic``/...) are the original
    schema and stay valid for the many runs that really do have one opponent; the
    ``opponents:`` list is required for anything else (LVMS run3_fourcar has four).
    Mixing the two in one run is rejected rather than silently preferring one.
    """
    raw = item.get("opponents")
    legacy_keys = [k for k in ("opp_odom", "opp_odom_topic", "opponent_time_offset_seconds",
                               "opponent_z_offset") if k in item]
    if raw is not None and legacy_keys:
        raise ValueError(
            f"{prefix}: set either 'opponents' or the single-opponent keys "
            f"({', '.join(legacy_keys)}), not both"
        )

    if raw is None:
        return [
            Opponent(
                name=str(item.get("opp_name", "opponent")),
                odom=_path_list(_require(item, "opp_odom", f"{prefix}.opp_odom"), f"{prefix}.opp_odom"),
                odom_topic=str(_require(item, "opp_odom_topic", f"{prefix}.opp_odom_topic")),
                time_offset_seconds=float(
                    _require(item, "opponent_time_offset_seconds", f"{prefix}.opponent_time_offset_seconds")
                ),
                z_offset=float(item.get("opponent_z_offset", 0.0)),
                map_offset_xyz=_xyz(item.get("opponent_map_offset_xyz"), f"{prefix}.opponent_map_offset_xyz"),
            )
        ]

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{prefix}.opponents must be a non-empty YAML list")
    opponents = []
    for j, opp in enumerate(raw):
        if not isinstance(opp, dict):
            raise ValueError(f"{prefix}.opponents[{j}] must be a YAML mapping")
        opp_prefix = f"{prefix}.opponents[{j}]"
        opponents.append(
            Opponent(
                name=str(_require(opp, "name", f"{opp_prefix}.name")),
                odom=_path_list(_require(opp, "odom", f"{opp_prefix}.odom"), f"{opp_prefix}.odom"),
                odom_topic=str(_require(opp, "odom_topic", f"{opp_prefix}.odom_topic")),
                time_offset_seconds=float(
                    _require(opp, "time_offset_seconds", f"{opp_prefix}.time_offset_seconds")
                ),
                z_offset=float(opp.get("z_offset", 0.0)),
                map_offset_xyz=_xyz(opp.get("map_offset_xyz"), f"{opp_prefix}.map_offset_xyz"),
            )
        )
    names = [o.name for o in opponents]
    if len(set(names)) != len(names):
        raise ValueError(f"{prefix}.opponents names must be unique, got {names}")
    return opponents


def _dataset_runs(inputs: dict[str, Any]) -> list[DatasetRun]:
    raw_runs = _require(inputs, "runs", "inputs.runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("inputs.runs must be a non-empty YAML list")
    runs = []
    for i, item in enumerate(raw_runs):
        if not isinstance(item, dict):
            raise ValueError(f"inputs.runs[{i}] must be a YAML mapping")
        prefix = f"inputs.runs[{i}]"
        runs.append(
            DatasetRun(
                name=str(_require(item, "name", f"{prefix}.name")),
                ego_odom=_path_list(_require(item, "ego_odom", f"{prefix}.ego_odom"), f"{prefix}.ego_odom"),
                fused_lidar=_path_list(_require(item, "fused_lidar", f"{prefix}.fused_lidar"), f"{prefix}.fused_lidar"),
                opponents=_run_opponents(item, prefix),
                ground_from_ego=bool(item.get("ground_from_ego", True)),
                fuse_raw=bool(item.get("fuse_raw", False)),
                tf_static=[_path(v) for v in item.get("tf_static", [])],
                enabled=bool(item.get("enabled", True)),
                ego_yaw_correction_deg=float(item.get("ego_yaw_correction_deg", 0.0)),
            )
        )
    return runs


def load_run_config(path: Path | str) -> RunConfig:
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    data = _read_yaml(cfg_path)
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path} must contain a YAML mapping")

    paths = _section(data, "paths")
    container = _section(data, "container")
    inputs = _section(data, "inputs")
    model = _section(data, "model")
    dataset = _section(data, "dataset")
    cache_cfg = data.get("cache", {})
    if not isinstance(cache_cfg, dict):
        raise ValueError("cache must be a YAML mapping when provided")
    train = _section(data, "train")
    evaluate = _section(data, "evaluate")
    probe_cfg = data.get("probe", {})
    if not isinstance(probe_cfg, dict):
        raise ValueError("probe must be a YAML mapping when provided")
    visualize = _section(data, "visualize")
    export = _section(data, "export")
    slurm = _section(data, "slurm")

    repo_dir = _path(_require(paths, "repo_dir", "paths.repo_dir"))
    run_name = str(_require(data, "run_name", "run_name"))
    runs = _dataset_runs(inputs)

    visualize_run_index = int(_require(visualize, "run_index", "visualize.run_index"))
    _check_index(visualize_run_index, len(runs), "visualize.run_index")
    probe_runs = probe_cfg.get("runs", "all")
    if isinstance(probe_runs, str):
        probe_runs = [r.name for r in runs] if probe_runs == "all" else [probe_runs]
    if not isinstance(probe_runs, list):
        raise ValueError("probe.runs must be 'all', a run name, or a YAML list of run names")
    known = {r.name for r in runs}
    for name in probe_runs:
        if str(name) not in known:
            raise ValueError(f"probe.runs entry {name!r} is not a configured run: {sorted(known)}")

    stages = str(_require(_section(data, "pipeline"), "stages", "pipeline.stages"))
    if stages not in STAGES:
        raise ValueError(f"pipeline.stages must be one of: {', '.join(sorted(STAGES))}")

    return RunConfig(
        path=cfg_path,
        raw=data,
        run_name=run_name,
        repo_dir=repo_dir,
        container_image=_path(_require(container, "image", "container.image"), repo_dir),
        container_binds=_path_list(_require(container, "binds", "container.binds"), "container.binds"),
        log_dir=_path(_require(paths, "log_dir", "paths.log_dir")),
        data_root=_path(_require(paths, "data_root", "paths.data_root"), repo_dir),
        ego_odom_topic=str(_require(inputs, "ego_odom_topic", "inputs.ego_odom_topic")),
        fused_lidar_topic=str(_require(inputs, "fused_lidar_topic", "inputs.fused_lidar_topic")),
        runs=runs,
        model_cfg=_path(_require(model, "cfg_file", "model.cfg_file"), repo_dir),
        checkpoint=_path(model["checkpoint"]) if model.get("checkpoint") else None,
        pretrained_model=_path(model["pretrained_model"]) if model.get("pretrained_model") else None,
        clean_dataset=bool(_require(dataset, "clean", "dataset.clean")),
        cache_enabled=bool(cache_cfg.get("enabled", True)),
        cache_dir=_path(cache_cfg.get("dir", "cache"), _path(_require(paths, "data_root", "paths.data_root"), repo_dir)),
        build_max_frames=int(_require(dataset, "max_frames", "dataset.max_frames")),
        build_stride=int(_require(dataset, "stride", "dataset.stride")),
        min_points_in_opponent_box=(
            int(dataset["min_points_in_opponent_box"])
            if dataset.get("min_points_in_opponent_box") is not None
            else None
        ),
        # 0 disables gating and decodes every cloud, which is the old behaviour.
        opponent_range_gate_m=float(dataset.get("opponent_range_gate_m", 0.0)),
        far_sample_seconds=float(dataset.get("far_sample_seconds", 2.0)),
        far_sample_period_seconds=float(dataset.get("far_sample_period_seconds", 30.0)),
        empty_train_fraction=float(dataset.get("empty_train_fraction", 0.0)),
        empty_val_fraction=float(dataset.get("empty_val_fraction", 0.0)),
        chunk_seconds=float(_require(dataset, "chunk_seconds", "dataset.chunk_seconds")),
        guard_seconds=float(_require(dataset, "guard_seconds", "dataset.guard_seconds")),
        train_fraction=float(_require(dataset, "train_fraction", "dataset.train_fraction")),
        test_fraction=float(_require(dataset, "test_fraction", "dataset.test_fraction")),
        batch_size=int(_require(train, "batch_size", "train.batch_size")),
        epochs=int(_require(train, "epochs", "train.epochs")),
        workers=int(_require(train, "workers", "train.workers")),
        eval_batch_size=int(_require(evaluate, "batch_size", "evaluate.batch_size")),
        eval_score_thresh=float(_require(evaluate, "score_thresh", "evaluate.score_thresh")),
        probe_max_frames=int(probe_cfg.get("max_frames", 200)),
        probe_window_seconds=float(probe_cfg.get("window_seconds", 60.0)),
        probe_runs=[str(r) for r in probe_runs],
        visualize_start_seconds=(
            float(visualize["start_seconds"])
            if "start_seconds" in visualize and visualize["start_seconds"] is not None
            else None
        ),
        visualize_duration_seconds=float(_require(visualize, "duration_seconds", "visualize.duration_seconds")),
        visualize_stride=int(_require(visualize, "stride", "visualize.stride")),
        visualize_run_index=visualize_run_index,
        visualize_out=_path(_require(visualize, "out", "visualize.out")),
        pred_topic=str(_require(visualize, "pred_topic", "visualize.pred_topic")),
        onnx_out_dir=_absolute_path(_require(export, "onnx_out_dir", "export.onnx_out_dir"), "export.onnx_out_dir"),
        params_h=_absolute_path(_require(export, "params_h", "export.params_h"), "export.params_h"),
        slurm_partition=str(_require(slurm, "partition", "slurm.partition")),
        slurm_setup_gres=str(_require(slurm, "setup_gres", "slurm.setup_gres")),
        slurm_pipeline_gres=str(_require(slurm, "pipeline_gres", "slurm.pipeline_gres")),
        slurm_setup_cpus=int(_require(slurm, "setup_cpus", "slurm.setup_cpus")),
        slurm_pipeline_cpus=int(_require(slurm, "pipeline_cpus", "slurm.pipeline_cpus")),
        slurm_setup_mem=str(_require(slurm, "setup_mem", "slurm.setup_mem")),
        slurm_pipeline_mem=str(_require(slurm, "pipeline_mem", "slurm.pipeline_mem")),
        slurm_setup_time=str(_require(slurm, "setup_time", "slurm.setup_time")),
        slurm_pipeline_time=str(_require(slurm, "pipeline_time", "slurm.pipeline_time")),
    )


def add_run_config_arg(parser: Any) -> None:
    parser.add_argument("--run-config", type=Path, required=True, help="Required pipeline YAML config")


def log(message: str = "") -> None:
    print(message, flush=True)


def log_section(title: str) -> None:
    log(f"\n[{title}]")


def log_kv(key: str, value: Any, *, indent: int = 2) -> None:
    log(f"{' ' * indent}{key}: {value}")


def log_kvs(title: str, items: list[tuple[str, Any]]) -> None:
    log_section(title)
    for key, value in items:
        log_kv(key, value)


def format_cmd(cmd: list[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def pack_object_array(items: list[Any]) -> np.ndarray:
    out = np.empty(len(items), dtype=object)
    out[:] = items
    return out


def cache_path(cfg: RunConfig, name: str) -> Path:
    return cfg.cache_dir / f"{name}.npz"


def source_fingerprint(paths: list[Path]) -> np.ndarray:
    rows = []
    for path in paths:
        for mcap_path in mcap_paths([path], "cache source"):
            stat = mcap_path.stat()
            rows.append((str(mcap_path), str(stat.st_size), str(stat.st_mtime_ns)))
    return np.asarray(sorted(rows), dtype=object)


def _same_fingerprint(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and all(tuple(a) == tuple(b) for a, b in zip(left.tolist(), right.tolist()))


def load_npz_cache(path: Path, *, expected_fingerprint: np.ndarray, expected_version: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            version = int(np.asarray(data["cache_version"]).reshape(-1)[0])
            fingerprint = np.asarray(data["source_fingerprint"], dtype=object)
            if version != expected_version or not _same_fingerprint(fingerprint, expected_fingerprint):
                log_kvs("cache stale", [("path", path), ("version", version), ("expected_version", expected_version)])
                return None
            log_kvs("cache hit", [("path", path)])
            return {key: data[key] for key in data.files if key not in {"cache_version", "source_fingerprint"}}
    except Exception as exc:
        log_kvs("cache ignored", [("path", path), ("reason", exc)])
        return None


def write_npz_cache(path: Path, payload: dict[str, Any], *, source_fingerprint: np.ndarray, version: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            cache_version=np.asarray([version], dtype=np.int64),
            source_fingerprint=source_fingerprint,
            **payload,
        )
    tmp_path.replace(path)


def bash_quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def rel_to(path: Path, root: Path) -> str:
    path = path.resolve()
    root = root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def tools_cfg_arg(cfg: RunConfig) -> str:
    return rel_to(cfg.model_cfg, cfg.repo_dir / "tools")


def job_config_path(cfg: RunConfig) -> Path:
    try:
        return cfg.repo_dir / cfg.path.relative_to(REPO_DIR)
    except ValueError:
        return cfg.path


def check_container_requirements(cfg: RunConfig, *, require_image: bool, require_apptainer: bool = True) -> None:
    if require_apptainer and shutil.which("apptainer") is None:
        raise FileNotFoundError("apptainer is required but was not found on PATH")
    for bind in cfg.container_binds:
        if not bind.exists():
            raise FileNotFoundError(f"Configured Apptainer bind path does not exist: {bind}")
    if not cfg.repo_dir.is_dir():
        raise NotADirectoryError(cfg.repo_dir)
    if require_image and not cfg.container_image.is_file():
        raise FileNotFoundError(f"Configured Apptainer image does not exist: {cfg.container_image}")


def apptainer_exec_cmd(cfg: RunConfig, command: list[str], *, writable_tmpfs: bool = False) -> list[str]:
    # --containall is load-bearing, not tidiness. Without it apptainer hangs on this
    # cluster before reaching the container at all: `apptainer exec <sif> echo hi`
    # blocks past 120 s with zero CPU, image on NFS or on node-local disk alike, with
    # or without --nv. `apptainer --debug` shows it stalling in prepareAutofs() /
    # keepAutofsMount(), which opens a file descriptor on every autofs mount point it
    # finds -- /u, /p, /s, /sw, /bigtemp, /bigtemp1, /bigtemp2 -- to keep the
    # automounts alive. Touching those triggers an automount, and one of them is not
    # responding, so apptainer blocks there. --containall skips the default host binds
    # entirely, so the autofs walk never happens.
    #
    # This is a cluster fault, not a pipeline bug: the same container trained fine on
    # 2026-07-31 and started hanging by 2026-08-03. Worth a helpdesk ticket; until
    # then this is the workaround. It is also why eval jobs 6529312/6529325 wrote
    # zero-byte logs -- they never started.
    #
    # --containall costs the host's /tmp and $HOME (both become private tmpfs). Nothing
    # here needs either: everything read or written lives under the explicit binds
    # below. It does NOT cost /dev/shm, which apptainer keeps mounted, so PyTorch
    # dataloader workers still have shared memory to pass tensors through.
    cmd = ["apptainer", "exec", "--containall", "--nv", "--cleanenv"]
    # --containall drops the automatic cwd bind too, so a relative script path or
    # relative --cfg_file no longer resolves. Every caller runs from tools/.
    cmd.extend(["--pwd", str(cfg.repo_dir / "tools")])
    if writable_tmpfs:
        cmd.append("--writable-tmpfs")
    for bind in cfg.container_binds:
        cmd.extend(["--bind", f"{bind}:{bind}"])
    cmd.extend(
        [
            "--env",
            f"PYTHONPATH={cfg.repo_dir}",
            "--env",
            f"PIPELINE_APPTAINER_IMAGE={cfg.container_image.resolve()}",
            # numba's eager @cuda.jit compilation in kitti_object_eval_python/rotate_iou.py
            # segfaults acquiring a CUDA context in this container (numba/driver mismatch),
            # which silently reported 0.0000 official AP instead of failing. Force the
            # pure-Python CUDA simulator; only used at AP-scoring time, not train/inference.
            "--env",
            "NUMBA_ENABLE_CUDASIM=1",
            # Under sbatch every stream is a file, so CPython block-buffers stdout at
            # 8 KB and line-buffers stderr -- and tqdm's progress bars end in \r, which
            # never triggers a line flush. A job killed before it fills the buffer
            # therefore writes a zero-byte .out no matter how long it ran, which is
            # exactly what eval jobs 6529312 and 6529325 did: both were making progress
            # and both looked hung. Setting this here rather than adding -u per stage
            # covers the container re-exec in pp_pipeline.py and every child it spawns.
            "--env",
            "PYTHONUNBUFFERED=1",
            str(cfg.container_image),
            *command,
        ]
    )
    return cmd


def inside_pipeline_container(cfg: RunConfig) -> bool:
    expected = str(cfg.container_image.resolve())
    return any(
        os.environ.get(name) == expected
        for name in ("PIPELINE_APPTAINER_IMAGE", "APPTAINER_CONTAINER", "SINGULARITY_CONTAINER")
    )


class Runner:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def run(self, cmd: list[str], cwd: Path) -> None:
        log_kvs("command", [("cwd", cwd), ("cmd", format_cmd(cmd))])
        if self.dry_run:
            log_kv("dry_run", True)
            return
        subprocess.run(cmd, cwd=str(cwd), check=True)

    def capture_to_file(self, cmd: list[str], cwd: Path, out_file: Path) -> None:
        log_kvs("command", [("cwd", cwd), ("cmd", format_cmd(cmd)), ("stdout", out_file)])
        if self.dry_run:
            log_kv("dry_run", True)
            return
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            subprocess.run(cmd, cwd=str(cwd), check=True, stdout=f)


def submit(cfg: RunConfig, mode: str, dry_run: bool, stages: str | None = None,
           walltime_override: str | None = None,
           constraint: str | None = None,
           checkpoint: str | None = None,
           max_voxels_test: int | None = None) -> None:
    if not dry_run:
        check_container_requirements(cfg, require_image=True, require_apptainer=False)
    is_setup = mode == "setup"
    job = "pp_setup" if is_setup else f"pp_{cfg.run_name}"
    gres = cfg.slurm_setup_gres if is_setup else cfg.slurm_pipeline_gres
    cpus = cfg.slurm_setup_cpus if is_setup else cfg.slurm_pipeline_cpus
    mem = cfg.slurm_setup_mem if is_setup else cfg.slurm_pipeline_mem
    walltime = walltime_override or (
        cfg.slurm_setup_time if is_setup else cfg.slurm_pipeline_time
    )
    wrapped_mode = "setup" if is_setup else "run"
    cfg_for_job = job_config_path(cfg)
    if is_setup:
        inner = (
            "module purge && "
            "module load apptainer && "
            # module purge strips these; the setup job runs pp_pipeline.py on the HOST
            # (not in the container) and needs a python3 with numpy/pyyaml.
            "module load gcc/14.2.0 && "
            "module load python/3.12.3 && "
            "command -v apptainer >/dev/null && "
            f"cd {bash_quote(cfg.repo_dir / 'tools')} && "
            f"python3 -u pp_pipeline.py --run-config {bash_quote(cfg_for_job)} --mode {wrapped_mode}"
        )
    else:
        container_cmd = apptainer_exec_cmd(
            cfg,
            [
                "python3",
                "-u",
                str(cfg.repo_dir / "tools" / "pp_pipeline.py"),
                "--run-config",
                str(cfg_for_job),
                "--mode",
                wrapped_mode,
            ],
        )
        if stages:
            container_cmd.extend(["--stages", stages])
        if checkpoint:
            container_cmd.extend(["--checkpoint", checkpoint])
        if max_voxels_test:
            container_cmd.extend(["--max-voxels-test", str(max_voxels_test)])
        inner = (
            "module purge && "
            "module load apptainer && "
            f"cd {bash_quote(cfg.repo_dir / 'tools')} && "
            f"{shlex.join(container_cmd)}"
        )
    wrapped = f"bash -c {bash_quote(inner)}"
    cmd = [
        "sbatch",
        "-p",
        cfg.slurm_partition,
        "--gres",
        gres,
        "-c",
        str(cpus),
        "--mem",
        mem,
        "-t",
        walltime,
        "-J",
        job,
        "-o",
        str(cfg.log_dir / f"{job}-%A.out"),
        "-e",
        str(cfg.log_dir / f"{job}-%A.err"),
        "--export=NONE",
    ]
    # GPU model selection is done with --constraint on this cluster; --gres only
    # takes a count and a generic type name. Features are per node, e.g.
    # a100_80gb (cheetah04), a100_40gb (cheetah01), a40 (jaguar01/06).
    if constraint:
        cmd.extend(["--constraint", constraint])
    cmd.extend(["--wrap", wrapped])
    log_kvs(
        "submit",
        [
            ("job", job),
            ("partition", cfg.slurm_partition),
            ("gres", gres),
            ("cpus", cpus),
            ("mem", mem),
            ("time", walltime),
            ("stdout", cfg.log_dir / f"{job}-%A.out"),
            ("stderr", cfg.log_dir / f"{job}-%A.err"),
            ("cmd", format_cmd(cmd)),
        ],
    )
    if not dry_run:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)


def parse_pipeline_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_run_config_arg(parser)
    parser.add_argument("--mode", choices=("run", "setup", "submit-run", "submit-setup"), required=True)
    parser.add_argument("--stages", choices=STAGES)
    # The run config carries ONE walltime for every stage, defaulting to 4 days for
    # training. Short stages inherit it, and a job whose walltime overruns a cluster
    # reservation does not queue behind it -- it sits PENDING (ReqNodeNotAvail)
    # indefinitely. That has now blocked two eval submissions and one infos job ahead
    # of the 2026-08-05 outage. Pass a realistic walltime for short stages.
    parser.add_argument("--time", dest="walltime",
                        help="override slurm.pipeline_time for this submission, "
                             "e.g. 02:00:00")
    # The acceptance criterion compares against Jay's checkpoint_epoch_80.pth on OUR
    # val split, so two evals differing only in checkpoint have to be runnable
    # without cloning the whole run config for one field.
    parser.add_argument("--checkpoint",
                        help="override model.checkpoint for this run")
    parser.add_argument("--max-voxels-test", dest="max_voxels_test", type=int,
                        help="override MAX_NUMBER_OF_VOXELS[test] for the eval stage")
    parser.add_argument("--constraint",
                        help="slurm --constraint for GPU selection, e.g. a100_80gb")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_cfg = load_run_config(args.run_config)
    return args


def configure_from_run_config(cfg: RunConfig) -> None:
    """Load geometry from the model yaml, then apply the run config's overrides.

    Every tool that computes boxes should call this rather than
    ``configure_from_model_yaml``, so the builder and the diagnostics cannot disagree
    about what counts as a labelled box.

    ``min_points_in_opponent_box`` lives here rather than in the model yaml because the
    model yaml only exposes it through ``filter_by_min_points`` under the ``gt_sampling``
    augmentor -- and this dataset does not use gt_sampling, so setting it there would
    switch on GT-paste augmentation as a side effect. With no such entry the fallback is
    **1**, which silently made a box holding a single lidar point a positive label.
    """
    global MIN_POINTS_IN_OPP

    configure_from_model_yaml(cfg.model_cfg)
    if cfg.min_points_in_opponent_box is not None:
        MIN_POINTS_IN_OPP = int(cfg.min_points_in_opponent_box)


def configure_from_model_yaml(cfg_file: Path) -> None:
    """Load training-coupled geometry from the PointPillars model yaml."""
    global ANCHOR_BOTTOM_HEIGHT, CAR_L, CAR_W, CAR_H, MIN_POINTS_IN_OPP, POINT_CLOUD_RANGE, OPP_BOX_CENTER_IN_RAM

    POINT_CLOUD_RANGE, (CAR_L, CAR_W, CAR_H), ANCHOR_BOTTOM_HEIGHT, MIN_POINTS_IN_OPP = load_model_settings(cfg_file)
    # Box centre sits forward of the opponent's rear axle by half a car length MINUS the
    # rear overhang. Dropping the overhang term pins the box's rear face to the rear axle
    # and pushes the whole box ~1 m ahead of the car -- see OPP_REAR_OVERHANG.
    OPP_BOX_CENTER_IN_RAM = np.array(
        [CAR_L * 0.5 - OPP_REAR_OVERHANG, 0.0, COG_IN_RAM[2]],
        dtype=np.float64,
    )


@dataclass
class PoseTimeline:
    t: np.ndarray
    xyz: np.ndarray
    quat_xyzw: np.ndarray
    slerp: Slerp

    @classmethod
    def from_arrays(cls, t, xyz, quat_xyzw):
        from scipy.spatial.transform import Rotation, Slerp

        order = np.argsort(t)
        t, xyz, quat_xyzw = t[order], xyz[order], quat_xyzw[order]
        keep = np.concatenate(([True], np.diff(t) > 1e-9))
        t, xyz, quat_xyzw = t[keep], xyz[keep], quat_xyzw[keep]
        return cls(t=t, xyz=xyz, quat_xyzw=quat_xyzw, slerp=Slerp(t, Rotation.from_quat(quat_xyzw)))

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def pose_at(self, t_query: float | np.ndarray) -> np.ndarray:
        single = bool(np.isscalar(t_query))
        tq = np.atleast_1d(np.asarray(t_query, np.float64))
        tq = np.clip(tq, self.t[0], self.t[-1])
        T = np.zeros((len(tq), 4, 4), dtype=np.float64)
        T[:, :3, :3] = self.slerp(tq).as_matrix()
        T[:, 0, 3] = np.interp(tq, self.t, self.xyz[:, 0])
        T[:, 1, 3] = np.interp(tq, self.t, self.xyz[:, 1])
        T[:, 2, 3] = np.interp(tq, self.t, self.xyz[:, 2])
        T[:, 3, 3] = 1.0
        return T[0] if single else T


def stamp_seconds(ros_msg) -> float:
    return float(ros_msg.header.stamp.sec) + float(ros_msg.header.stamp.nanosec) * 1e-9


def time_to_stamp(t: float) -> tuple[int, int]:
    s = int(np.floor(t))
    ns = int(np.round((t - s) * 1.0e9))
    if ns < 0:
        s -= 1
        ns += 1_000_000_000
    if ns >= 1_000_000_000:
        s += 1
        ns -= 1_000_000_000
    return s, ns


def mcap_paths(paths: list[Path], label: str) -> list[Path]:
    out = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            out.append(path)
        elif path.is_dir():
            out.extend(sorted(path.glob("*.mcap")))
        else:
            raise FileNotFoundError(f"No {label} MCAP at {path}")
    if not out:
        raise FileNotFoundError(f"No {label} MCAP paths configured")
    return out


def fused_mcap_paths(paths: list[Path]) -> list[Path]:
    return mcap_paths(paths, "fused lidar")


def mcap_start_time(path: Path) -> float:
    """First message log time in seconds, from the MCAP summary (no full scan)."""
    from mcap.reader import make_reader

    with open(path, "rb") as f:
        summary = make_reader(f).get_summary()
    if summary is None or summary.statistics is None:
        raise ValueError(f"{path} has no MCAP summary; cannot order it chronologically")
    return float(summary.statistics.message_start_time) * 1e-9


def mcap_time_span(path: Path) -> tuple[float, float]:
    """(first, last) message log time in seconds, from the MCAP summary."""
    from mcap.reader import make_reader

    with open(path, "rb") as f:
        summary = make_reader(f).get_summary()
    if summary is None or summary.statistics is None:
        raise ValueError(f"{path} has no MCAP summary; cannot read its time span")
    st = summary.statistics
    return float(st.message_start_time) * 1e-9, float(st.message_end_time) * 1e-9


def sort_mcaps_chronologically(paths: list[Path]) -> list[Path]:
    """Order split bags by recording time rather than filename.

    rosbag2 splits are named ``..._0, _1, ... _10``, which sort lexically as
    ``_0, _1, _10, _2`` -- so a name sort interleaves the run's second minute with its
    eleventh. Fusion carries the latest flank cloud forward across files, so a wrong
    order silently pairs a front sweep with a flank sweep minutes away.
    """
    return [p for _, p in sorted((mcap_start_time(p), p) for p in paths)]


def merge_windows(windows: list[tuple[float, float]], merge_gap: float = 5.0) -> list[tuple[float, float]]:
    """Sort and coalesce overlapping or near-adjacent (lo, hi) intervals."""
    out: list[tuple[float, float]] = []
    for lo, hi in sorted(windows):
        if out and lo - out[-1][1] <= merge_gap:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def opponent_range_windows(
    ego,
    opponents,
    t_lo: float,
    t_hi: float,
    gate_m: float,
    far_sample_seconds: float,
    far_sample_period_seconds: float,
    pad_seconds: float = 2.0,
    grid_seconds: float = 0.5,
) -> tuple[list[tuple[float, float]], float]:
    """Time windows worth decoding, computed from odometry before any cloud is read.

    A frame whose nearest opponent is hundreds of metres away cannot produce a label:
    the box falls outside POINT_CLOUD_RANGE and OpenPCDet discards it. Decoding those
    clouds only to drop them is what made a build spend hours producing nothing --
    measured at Laguna/LVMS, only ~51% of run time has any opponent within 110 m, and
    on LVMS 2025_01_08 only 29%.

    Far stretches are still sampled sparsely rather than skipped outright, because they
    are the source of the empty frames that ``empty_train_fraction`` feeds on, and those
    negatives are the defence against false positives. Only a bounded number is needed
    (negatives are capped as a share of positives), so a short sample every so often is
    enough and costs a fraction of a full decode.

    Returns (windows, kept_fraction). An empty gate returns the whole span.
    """
    if gate_m <= 0.0:
        return [(t_lo, t_hi)], 1.0

    grid = np.arange(t_lo, t_hi, grid_seconds)
    if len(grid) == 0:
        return [(t_lo, t_hi)], 1.0
    nearest = np.full(len(grid), np.inf)
    ego_xyz = ego.pose_at(grid)[:, :3, 3]
    for opp, timeline in opponents:
        opp_t = grid + opp.time_offset_seconds
        inside = (opp_t >= timeline.t_start) & (opp_t <= timeline.t_end)
        if not inside.any():
            continue
        d = np.linalg.norm(ego_xyz[inside] - timeline.pose_at(opp_t[inside])[:, :3, 3], axis=1)
        nearest[inside] = np.minimum(nearest[inside], d)

    near = nearest <= gate_m
    windows: list[tuple[float, float]] = []
    if near.any():
        edges = np.flatnonzero(np.diff(near.astype(np.int8)))
        starts = ([0] if near[0] else []) + [i + 1 for i in edges if near[i + 1]]
        ends = [i for i in edges if near[i]] + ([len(near) - 1] if near[-1] else [])
        for s, e in zip(sorted(starts), sorted(ends)):
            windows.append((float(grid[s]) - pad_seconds, float(grid[e]) + pad_seconds))

    # Sparse negatives from the stretches no opponent was near.
    if far_sample_seconds > 0.0 and far_sample_period_seconds > far_sample_seconds:
        for start in np.arange(t_lo, t_hi, far_sample_period_seconds):
            mid = grid.searchsorted(start)
            if mid < len(near) and not near[mid]:
                windows.append((float(start), float(start) + far_sample_seconds))

    windows = merge_windows([(max(lo, t_lo), min(hi, t_hi)) for lo, hi in windows])
    total = t_hi - t_lo
    kept = sum(hi - lo for lo, hi in windows)
    return windows, (kept / total if total > 0 else 1.0)


def mcaps_covering(paths: list[Path], t_lo: float, t_hi: float) -> list[Path]:
    """Chronologically ordered bags whose recorded span intersects [t_lo, t_hi]."""
    out = []
    for path in sort_mcaps_chronologically(paths):
        start, end = mcap_time_span(path)
        if end >= t_lo and start <= t_hi:
            out.append(path)
    return out


def resolve_fused_lidar_mcap(paths: list[Path]) -> Path:
    return fused_mcap_paths(paths)[0]


def _read_uva_odometry(mcap_path: Path, topic: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    ts, xyz, quat = [], [], []
    with open(mcap_path, "rb") as f:
        for _, _, _, odom in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
            topics=[topic]
        ):
            t = stamp_seconds(odom)
            p, q = odom.pose.pose.position, odom.pose.pose.orientation
            ts.append(t)
            xyz.append([p.x, p.y, p.z])
            quat.append([q.x, q.y, q.z, q.w])
    if not ts:
        raise ValueError(f"No {topic} in {mcap_path}")
    return np.asarray(ts, np.float64), np.asarray(xyz, np.float64), np.asarray(quat, np.float64)


def load_odom_timeline(paths: list[Path], topic: str) -> PoseTimeline:
    if not paths:
        raise ValueError("Configured odometry paths must contain stamped odometry MCAP paths.")
    t_all, x_all, q_all = [], [], []
    for path in mcap_paths(paths, "odometry"):
        t, x, q = _read_uva_odometry(path, topic)
        t_all.append(t)
        x_all.append(x)
        q_all.append(q)
    return PoseTimeline.from_arrays(np.concatenate(t_all), np.vstack(x_all), np.vstack(q_all))


def load_opponent_timeline(paths: list[Path], topic: str) -> PoseTimeline:
    """Raw opponent timeline, exactly as recorded.

    Deliberately does NOT repair a degenerate orientation: this is what populates the
    odometry cache, and the cache must hold what the bag said.  The repair happens in
    ``build_pp_kitti.run_timelines``, the one place run geometry is assembled.
    """
    return load_odom_timeline(paths, topic)


# Minimum path length a timeline must cover before a constant heading is judged
# impossible rather than merely unlucky.  A car that genuinely never moved has a
# constant heading and nothing is wrong with it.
DEGENERATE_MIN_PATH_M = 50.0
# Below this, two quaternion samples count as the same orientation.  RTK heading noise is
# ~0.1 deg; 1e-3 rad (0.057 deg) is under that, so real data never trips it.
DEGENERATE_YAW_SPREAD_RAD = 1.0e-3
# Arclength window used to derive heading from position.  At 50 Hz and racing speed one
# sample step is ~0.8 m, so 2 m spans a few samples: long enough that RTK noise (~2 cm)
# contributes under a degree, short enough to track a corner.
MOTION_HEADING_WINDOW_M = 2.0


def path_length(xyz: np.ndarray) -> float:
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1).sum())


def apply_yaw_correction(quat_xyzw: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate each pose about z by ``degrees`` (positive = counter-clockwise).

    Used to correct a lidar_plate / ego heading error. Composed on the RIGHT
    (``q * dq``), i.e. in the ego's own body frame, because that is where a plate
    misalignment lives -- a left-multiply would rotate about the map's z instead and
    would not be equivalent to re-rotating the plate.
    """
    half = math.radians(degrees) * 0.5
    dq = np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float64)
    x1, y1, z1, w1 = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    x2, y2, z2, w2 = dq
    return np.column_stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def yaw_from_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """Z-axis yaw of each quaternion, without constructing rotation matrices."""
    x, y, z, w = quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2], quat_xyzw[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def orientation_is_degenerate(xyz: np.ndarray, quat_xyzw: np.ndarray) -> Optional[str]:
    """Reason the orientation channel is unusable, or None if it looks real.

    MEASURED FAILURE, not a hypothetical (2026-08-14).  Both runs sourced from
    ``/opponent/gps`` -- caltech_2026_05_05_long and iu_2026_04_15 -- publish the identity
    quaternion on *every* message (25,001 and 29,999 of them).  The field was never
    populated; it is a position-only fix.  Consumed as a heading it makes
    ``opponent_box_from_poses`` yield ``box_yaw = -ego_yaw``, so the box counter-rotates
    with our own car.  That produced 4,783 labels (9.1% of the dataset) whose heading is
    pure fiction, and it survived every audit: the yaw check was a PCA pooled across runs,
    and PCA is axis-only.

    The tell is that relative yaw should sit near zero on a racetrack.  Measured median
    |relative yaw| is 4.1-13.1 deg on the six ``uva_odometry`` / ``global_filtered`` runs
    and 93.5 / 103.7 deg on these two.
    """
    if len(quat_xyzw) < 2:
        return None
    if path_length(xyz) < DEGENERATE_MIN_PATH_M:
        return None  # a parked car really does hold one heading
    yaw = yaw_from_quat_xyzw(quat_xyzw)
    spread = float(np.abs((yaw - yaw[0] + np.pi) % (2 * np.pi) - np.pi).max())
    if spread < DEGENERATE_YAW_SPREAD_RAD:
        if np.allclose(quat_xyzw, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
            return "identity quaternion on every sample (orientation field never populated)"
        return f"constant orientation over {path_length(xyz):.0f} m of travel"
    return None


def heading_quat_from_motion(
    t: np.ndarray, xyz: np.ndarray, window_m: float = MOTION_HEADING_WINDOW_M
) -> tuple[np.ndarray, float]:
    """Quaternions whose yaw is course-over-ground, derived from the position track.

    Returns ``(quat_xyzw, valid_fraction)``.

    Heading at sample ``i`` is the bearing across an arclength window centred on ``i``,
    which both smooths RTK noise and survives the 50 Hz sample spacing.  Samples where the
    car covered less than ``window_m`` (stopped, or crawling) have no defined heading and
    hold the nearest valid one instead.

    Two honest limitations, because this is a recovery and not a GT source:

      * It is course-over-ground, not body yaw.  They differ by the slip angle, which on a
        racing car mid-corner is several degrees and not negligible.
      * Roll and pitch are dropped.  For the box that is harmless -- ``GroundSurface``
        already replaces the opponent's z -- but the returned pose is not a full attitude.
    """
    n = len(xyz)
    if n < 2:
        raise ValueError("need at least two samples to derive heading from motion")
    step = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(step)))
    half = window_m * 0.5
    lo = np.searchsorted(s, s - half, side="left").clip(0, n - 1)
    # -1 before the clip, not after: "last index within the window" is searchsorted(right)
    # minus one, and clipping first would cap it at n-2 and drop the final sample.
    hi = (np.searchsorted(s, s + half, side="right") - 1).clip(0, n - 1)
    hi = np.maximum(hi, lo)
    d = xyz[hi, :2] - xyz[lo, :2]
    valid = np.linalg.norm(d, axis=1) >= window_m * 0.5
    if not valid.any():
        raise ValueError(
            f"cannot derive heading from motion: no sample covers {window_m * 0.5:.1f} m "
            f"(total path {path_length(xyz):.1f} m)"
        )
    yaw = np.arctan2(d[:, 1], d[:, 0])
    # Hold the nearest valid heading across stopped stretches rather than emitting a
    # bearing computed from noise.
    idx = np.where(valid, np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    if idx[0] < 0:  # leading run of invalid samples: back-fill from the first valid one
        idx[idx < 0] = int(np.flatnonzero(valid)[0])
    yaw = yaw[idx]

    half_yaw = yaw * 0.5
    quat = np.zeros((n, 4), dtype=np.float64)
    quat[:, 2] = np.sin(half_yaw)
    quat[:, 3] = np.cos(half_yaw)
    return quat, float(valid.mean())


def opponent_timeline_repaired(
    t: np.ndarray,
    xyz: np.ndarray,
    quat_xyzw: np.ndarray,
    *,
    label: str,
) -> tuple[PoseTimeline, Optional[str]]:
    """Opponent timeline with a degenerate orientation channel replaced by motion heading.

    Returns the timeline and the repair note (None when the recorded orientation was used
    as-is).  Callers should log the note: a silently repaired run is exactly how the
    original bug went unnoticed for a full training cycle.
    """
    reason = orientation_is_degenerate(xyz, quat_xyzw)
    if reason is None:
        return PoseTimeline.from_arrays(t, xyz, quat_xyzw), None
    quat_fixed, valid_fraction = heading_quat_from_motion(t, xyz)
    note = (
        f"orientation unusable ({reason}); heading derived from motion, "
        f"{valid_fraction:.1%} of samples above the {MOTION_HEADING_WINDOW_M / 2:.1f} m "
        "window, rest hold nearest"
    )
    log_kvs(f"REPAIRED opponent orientation: {label}", [("reason", reason), ("action", note)])
    return PoseTimeline.from_arrays(t, xyz, quat_fixed), note


def build_point_dtype(ros_msg) -> np.dtype:
    wanted = {"x", "y", "z", "reflectance"}
    names, formats, offsets = [], [], []
    for field in ros_msg.fields:
        if field.name in wanted:
            if field.datatype != 7:
                raise RuntimeError(f"{field.name} must be float32, got datatype {field.datatype}")
            names.append(field.name)
            formats.append("<f4")
            offsets.append(int(field.offset))
    if set(names) != wanted:
        raise RuntimeError(f"need x,y,z,reflectance; got {names}")
    return np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": int(ros_msg.point_step)}
    )


def parse_fused_cloud(ros_msg, dtype: np.dtype) -> np.ndarray | None:
    n = int(ros_msg.width * ros_msg.height)
    if n == 0:
        return None
    arr = np.frombuffer(bytes(ros_msg.data), dtype, count=n)
    xyz = np.column_stack((arr["x"], arr["y"], arr["z"]))
    keep = (xyz[:, 0] != 0) | (xyz[:, 1] != 0) | (xyz[:, 2] != 0)
    xyz, refl = xyz[keep], arr["reflectance"][keep]
    intensity = np.clip(refl * REFL_SCALE, 0.0, 1.0).astype(np.float32)
    return np.column_stack((xyz.astype(np.float32), intensity))


# --- raw 3-lidar fusion -------------------------------------------------------------
# Most archived runs (all IMS/LVMS ones) publish only per-sensor /pcl_front|left|right
# and no fused /lidar_points, so they must be fused offline. This reproduces the on-car
# iris_fusion_node: transform each sensor cloud into a common frame and concatenate.
#
# Fusion targets LIDAR_FRAME directly and resolves the chain from the *session's own*
# tf_static, never a hardcoded constant. That is the Phase B lesson (see
# PIPELINE_PLAN.md 1.4): assuming a frame is how every Caltech/IU box ended up 0.198 m
# too high. Fusing straight into the label frame makes that class of bug impossible here.
RAW_LIDAR_TOPIC_FRAMES = {
    "/pcl_front": "iris_front",
    "/pcl_left": "iris_left",
    "/pcl_right": "iris_right",
}
FUSION_TRIGGER_TOPIC = "/pcl_front"


def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        raise ValueError("zero-norm quaternion in tf_static")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# The authoritative rig description. A bag's own /tf_static is whatever the car happened
# to be publishing that day and can be wrong; the URDF is the calibration of record, so
# fusion resolves sensor placement from it (user decision, 2026-07-30 -- this reverses the
# earlier §1.8 conclusion that each session's tf_static should win).
#
# av21 and av24 are different rigs, not different calibrations of one rig: av21 mounts
# three `luminar_*` sensors straight on rear_axle_middle, av24 mounts three `iris_*` on a
# rotated `lidar_plate`. Only av24 data is in scope, so the frame names in a bag's
# tf_static are what identifies the rig, and anything that is not av24 is excluded rather
# than reconciled.
URDF_DIR = TOOLS_DIR / "urdf"
AV24_URDF = URDF_DIR / "av24.urdf"
AV21_URDF = URDF_DIR / "av21.urdf"
AV21_LIDAR_FRAMES = ("luminar_front", "luminar_left", "luminar_right")


def _urdf_joint_graph(urdf_path: Path) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    """child_frame -> (parent_frame, R, t) for every fixed joint in a URDF."""
    import xml.etree.ElementTree as ET

    graph: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        xyz = np.zeros(3, dtype=np.float64)
        rpy = np.zeros(3, dtype=np.float64)
        if origin is not None:
            if origin.get("xyz"):
                xyz = np.asarray([float(v) for v in origin.get("xyz").split()], dtype=np.float64)
            if origin.get("rpy"):
                rpy = np.asarray([float(v) for v in origin.get("rpy").split()], dtype=np.float64)
        graph[child.get("link")] = (parent.get("link"), rpy_to_R(*rpy), xyz)
    return graph


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw (X then Y then Z) to a rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _resolve_chain(
    graph: dict[str, tuple[str, np.ndarray, np.ndarray]], frame: str, target: str, source: str
) -> tuple[np.ndarray, np.ndarray]:
    R, t, cur, hops = np.eye(3), np.zeros(3), frame, 0
    while cur != target and cur in graph:
        parent, Rp, tp = graph[cur]
        R, t = Rp @ R, Rp @ t + tp
        cur = parent
        hops += 1
        if hops > 32:
            raise ValueError(f"cycle resolving {frame} -> {target} in {source}")
    if cur != target:
        raise ValueError(
            f"{source}: chain for {frame!r} does not reach {target!r} (stopped at {cur!r}); "
            "refusing to guess sensor placement"
        )
    return R, t


def load_urdf_transforms(urdf_path: Path = AV24_URDF, target: str = LIDAR_FRAME) -> dict[str, tuple]:
    """Resolve each av24 lidar frame -> ``target`` from the URDF of record."""
    graph = _urdf_joint_graph(Path(urdf_path))
    return {
        frame: _resolve_chain(graph, frame, target, str(urdf_path))
        for frame in sorted(set(RAW_LIDAR_TOPIC_FRAMES.values()))
    }


def identify_rig(paths: list[Path]) -> tuple[str, set[str]]:
    """Classify a session's rig from the frames its /tf_static declares.

    Returns ("av24" | "av21" | "unknown", frames). Only av24 sessions are in scope.
    """
    frames = tf_static_frames(paths)
    if set(RAW_LIDAR_TOPIC_FRAMES.values()).issubset(frames):
        return "av24", frames
    if set(AV21_LIDAR_FRAMES) & frames:
        return "av21", frames
    return "unknown", frames


def tf_static_frames(paths: list[Path]) -> set[str]:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    frames: set[str] = set()
    for path in paths:
        with open(path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, _, _, msg in reader.iter_decoded_messages(topics=["/tf_static"]):
                for tr in msg.transforms:
                    frames.add(tr.child_frame_id)
                    frames.add(tr.header.frame_id)
    return frames


def compare_transforms(a: dict[str, tuple], b: dict[str, tuple]) -> dict[str, tuple[float, float]]:
    """Per-frame (translation metres, rotation degrees) difference between two rigs."""
    out = {}
    for frame in sorted(set(a) & set(b)):
        (Ra, ta), (Rb, tb) = a[frame], b[frame]
        cos = (np.trace(Ra.T @ Rb) - 1.0) * 0.5
        out[frame] = (
            float(np.linalg.norm(ta - tb)),
            float(math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))),
        )
    return out


def load_static_transforms(paths: list[Path], target: str = LIDAR_FRAME) -> dict[str, tuple]:
    """Resolve each raw lidar sensor frame -> ``target`` from the session's own tf_static.

    Raises if a sensor frame cannot reach ``target``: a missing chain means we do not
    know where the sensor is, and guessing would silently misplace every point.
    """
    from mcap.reader import make_reader  # local: keep module import-light
    from mcap_ros2.decoder import DecoderFactory

    graph: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for path in paths:
        with open(path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, _, _, msg in reader.iter_decoded_messages(topics=["/tf_static"]):
                for tr in msg.transforms:
                    t, q = tr.transform.translation, tr.transform.rotation
                    graph[tr.child_frame_id] = (
                        tr.header.frame_id,
                        quat_to_R(q.x, q.y, q.z, q.w),
                        np.array([t.x, t.y, t.z], dtype=np.float64),
                    )
    if not graph:
        raise ValueError(f"no /tf_static found in {[str(p) for p in paths]}")

    out: dict[str, tuple] = {}
    for frame in sorted(set(RAW_LIDAR_TOPIC_FRAMES.values())):
        R, t, cur, hops = np.eye(3), np.zeros(3), frame, []
        # tf convention: header.frame_id is the parent, so p_parent = R @ p_child + t.
        while cur != target and cur in graph:
            parent, Rp, tp = graph[cur]
            R, t = Rp @ R, Rp @ t + tp
            hops.append(cur)
            cur = parent
            if len(hops) > 32:
                raise ValueError(f"tf_static cycle resolving {frame} -> {target}")
        if cur != target:
            raise ValueError(
                f"tf_static chain for {frame!r} does not reach {target!r} "
                f"(stopped at {cur!r}); refusing to guess sensor placement"
            )
        out[frame] = (R, t)
    return out


# A flank sweep older than this relative to the front sweep is treated as missing rather
# than fused in. Sensors run at ~10 Hz and a sweep spans 63-66 ms (PIPELINE_PLAN.md 1.9),
# so anything past 200 ms means a dropout or a bag boundary, not normal staleness.
MAX_FLANK_AGE_SECONDS = 0.2


def mcap_window_args(t_lo: float | None, t_hi: float | None) -> dict:
    """start_time/end_time kwargs for iter_decoded_messages, in log-time nanoseconds.

    Lets the reader seek via the chunk index instead of scanning from byte zero -- the
    difference between seconds and half an hour on the 84 GB bags. Padded by 2 s because
    these are log times while everything else here uses header stamps.
    """
    args = {}
    if t_lo is not None:
        args["start_time"] = int((t_lo - 2.0) * 1e9)
    if t_hi is not None:
        args["end_time"] = int((t_hi + 2.0) * 1e9)
    return args


def iter_fused_raw_frames(
    paths: list[Path],
    transforms: dict[str, tuple],
    stats: dict | None = None,
    t_lo: float | None = None,
    t_hi: float | None = None,
):
    """Yield ``(timestamp_seconds, Nx4 points)`` fused into the transforms' target frame.

    Mirrors iris_fusion_node: hold the latest left/right cloud and emit one fused frame
    per front cloud (the front lidar sets the frame rate). Unlike the on-car node this
    refuses to emit a frame whose flank clouds are stale, and it requires ``paths`` in
    recording order (see sort_mcaps_chronologically).
    """
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    latest: dict[str, tuple[float, np.ndarray]] = {}
    dtypes: dict[str, np.dtype] = {}
    # Every sensor must have reported once before the first emit, otherwise the leading
    # frames are front-only partial clouds -- valid-looking but missing a third of the
    # scene, which would teach the model that the flanks are empty.
    need = {f for t, f in RAW_LIDAR_TOPIC_FRAMES.items() if t != FUSION_TRIGGER_TOPIC}
    counts = stats if stats is not None else {}
    for key in ("front_clouds", "fused", "skipped_no_flank", "skipped_stale_flank"):
        counts.setdefault(key, 0)
    previous_t = None
    for path in paths:
        with open(path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, channel, _, cloud in reader.iter_decoded_messages(
                topics=list(RAW_LIDAR_TOPIC_FRAMES), **mcap_window_args(t_lo, t_hi)
            ):
                frame = RAW_LIDAR_TOPIC_FRAMES[channel.topic]
                if frame not in dtypes:
                    dtypes[frame] = build_point_dtype(cloud)
                t = stamp_seconds(cloud)
                pts = parse_fused_cloud(cloud, dtypes[frame])
                if pts is not None and len(pts):
                    R, tr = transforms[frame]
                    pts = pts.copy()
                    pts[:, :3] = (pts[:, :3].astype(np.float64) @ R.T + tr).astype(np.float32)
                else:
                    pts = np.zeros((0, 4), dtype=np.float32)
                if channel.topic != FUSION_TRIGGER_TOPIC:
                    latest[frame] = (t, pts)
                    continue

                counts["front_clouds"] += 1
                if previous_t is not None and t < previous_t:
                    raise ValueError(
                        f"lidar timestamps went backwards ({previous_t:.3f} -> {t:.3f}) at "
                        f"{path}; the configured bags overlap or are out of recording order, "
                        "which would fuse each front sweep with an unrelated flank sweep"
                    )
                previous_t = t
                if not need.issubset(latest):
                    counts["skipped_no_flank"] += 1
                    continue  # still waiting on a flank sensor's first cloud
                if any(t - latest[f][0] > MAX_FLANK_AGE_SECONDS for f in need):
                    counts["skipped_stale_flank"] += 1
                    continue
                parts = [pts] + [v for _, (_, v) in sorted(latest.items()) if len(v)]
                counts["fused"] += 1
                yield t, (
                    np.concatenate(parts, axis=0) if parts else np.zeros((0, 4), np.float32)
                )


def tf_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def yaw_from_R(R: np.ndarray) -> float:
    return float(math.atan2(R[1, 0], R[0, 0]))


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


COG_ABOVE_GROUND = COG_IN_RAM[2] + RAM_Z_ABOVE_GROUND


class GroundSurface:
    """Track surface height sampled from the ego's own driven trajectory.

    The opponent's reported z is the least trustworthy channel of its pose: it comes from
    a different localization stack, sometimes a different vertical datum, and disagrees
    with ours by ~1.3 m (1 sigma) even where both cars occupied the same 25 cm of track
    (measured at Laguna Seca). Its x, y and yaw are far better. Since both cars drive the
    same surface, looking the ground height up from where *we* drove replaces the weakest
    input with the strongest one.

    Falls back to the opponent's own z when it strays further than ``max_dist`` from any
    point we drove, so an off-line excursion degrades gracefully instead of snapping to a
    distant, wrong height.

    **The surface is TILTED, not level.** Taking the nearest sample's height outright is only
    correct on flat ground. On a banked oval a lateral offset of a few metres is a real height
    difference, and the lookup happily returns a confidently wrong answer while still inside
    ``max_dist``. Measured at lvms_2025_01_06: the ego passed 7.5 m away, the box was placed
    1.0 m too high (z=+1.91 against a car whose returns spanned +0.45..+1.10), it therefore
    held ZERO points out of the 1372 inside its own footprint, and the opponent was dropped
    as "nothing observable" for ~645 frames.

    So the sample is extrapolated along the local surface instead. The ego sits ON the
    surface, so its body z-axis IS the surface normal, and the ground at a nearby query point
    is the plane through the sample with that normal. This needs no fitting, uses roll and
    pitch together, and collapses to the old behaviour on flat ground where n = (0, 0, 1).

    Validated out of sample against 419 hand-reviewed frames whose vertical offsets were
    dragged by eye, with no knowledge of this model: correlation +0.80, slope 0.84, residual
    0.17 m against a hand-measured error of 0.40 m. The slope below 1 means this OVERSHOOTS
    by about a fifth, most likely because the ego's reported roll includes suspension roll --
    the body leans outward relative to the road under cornering load, inflating the apparent
    bank. That residual is left uncorrected on purpose: a scale factor fitted to six
    opponents is the same mistake that once put a -0.120 s clock offset in the config, and
    0.17 m on a 1.55 m box still contains the car.
    """

    # A normal this far from vertical means the pose is bad, not the track banked; 35 deg is
    # steeper than any oval in this dataset (LVMS is 20).
    MIN_NORMAL_Z = 0.82
    # Cap on the extrapolated correction. Past this the local-plane model is being pushed
    # further than a track stays planar, so the magnitude is LIMITED rather than discarded:
    # reverting to the flat value would restore the very error this exists to remove, and a
    # partial correction in the right direction beats none. All corrections measured on this
    # dataset are <= 0.93 m, so this only bounds a tail.
    MAX_TILT_CORRECTION = 1.50

    def __init__(self, ego: "PoseTimeline", max_dist: float = 8.0):
        self.xy = ego.xyz[:, :2].copy()
        self.z = ego.xyz[:, 2] - COG_ABOVE_GROUND  # ego COG -> ground beneath it
        self.max_dist = float(max_dist)
        # Surface normal at each driven sample: the body z-axis, i.e. the third column of the
        # rotation matrix, written out rather than built via Rotation to keep this cheap.
        q = ego.quat_xyzw
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        self.normal = np.column_stack((
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ))
        self._tree = None
        try:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.xy)
        except Exception:  # pragma: no cover - scipy present in the container
            self._tree = None

    def ground_z(self, x: float, y: float) -> Optional[float]:
        """Ground height in map frame at (x, y), or None if we never drove near there."""
        q = np.array([x, y], dtype=np.float64)
        if self._tree is not None:
            dist, idx = self._tree.query(q, k=1)
        else:
            d = np.linalg.norm(self.xy - q, axis=1)
            idx = int(d.argmin())
            dist = float(d[idx])
        if dist > self.max_dist:
            return None
        idx = int(idx)
        z0 = float(self.z[idx])
        n = self.normal[idx]
        if not np.isfinite(n).all() or n[2] < self.MIN_NORMAL_Z:
            return z0                      # implausible attitude: stay flat rather than guess
        dz = -(n[0] * (x - self.xy[idx, 0]) + n[1] * (y - self.xy[idx, 1])) / n[2]
        if not np.isfinite(dz):
            return z0
        dz = float(np.clip(dz, -self.MAX_TILT_CORRECTION, self.MAX_TILT_CORRECTION))
        return z0 + dz


def opponent_box_from_poses(
    T_opp_cog: np.ndarray,
    T_ego_cog: np.ndarray,
    ground: Optional["GroundSurface"] = None,
) -> Optional[tuple[np.ndarray, float]]:
    """Return opponent box center/yaw in ego ``rear_axle_middle`` lidar coordinates.

    When ``ground`` is supplied the box bottom is taken from the ego's own driven surface
    at the opponent's planar position rather than from the opponent's reported z, which is
    the least reliable part of its pose (see GroundSurface).
    """
    T_map_ego_ram = T_ego_cog @ T_COG_RAM
    T_ego_ram_map = tf_inv(T_map_ego_ram)
    opp_center_in_cog = np.r_[OPP_BOX_CENTER_IN_RAM - COG_IN_RAM, 1.0]
    center_map = (T_opp_cog @ opp_center_in_cog)[:3]

    ground_z_map = None
    if ground is not None:
        ground_z_map = ground.ground_z(float(center_map[0]), float(center_map[1]))

    if ground_z_map is not None:
        # Trustworthy z: planar position from the opponent, height from our own surface.
        center_map = np.array(
            [center_map[0], center_map[1], ground_z_map + CAR_H * 0.5], dtype=np.float64
        )
        center = (T_ego_ram_map @ np.r_[center_map, 1.0])[:3]
    else:
        center = (T_ego_ram_map @ np.r_[center_map, 1.0])[:3]
        opp_ground_in_ram = np.array(
            [OPP_BOX_CENTER_IN_RAM[0], OPP_BOX_CENTER_IN_RAM[1], -RAM_Z_ABOVE_GROUND],
            dtype=np.float64,
        )
        bottom = (T_ego_ram_map @ (T_opp_cog @ np.r_[opp_ground_in_ram - COG_IN_RAM, 1.0]))[:3][2]
        center[2] = bottom + CAR_H * 0.5

    yaw = wrap_angle(yaw_from_R(T_opp_cog[:3, :3]) - yaw_from_R(T_map_ego_ram[:3, :3]))
    return center.astype(np.float64), yaw


def opponent_box_at(
    lidar_time: float,
    ego: PoseTimeline,
    opponent: PoseTimeline,
    opponent_time_offset: float,
    ground: Optional["GroundSurface"] = None,
) -> Optional[tuple[np.ndarray, float]]:
    opp_time = lidar_time + opponent_time_offset
    if lidar_time < ego.t_start or lidar_time > ego.t_end:
        return None
    if opp_time < opponent.t_start or opp_time > opponent.t_end:
        return None
    return opponent_box_from_poses(
        opponent.pose_at(opp_time), ego.pose_at(lidar_time), ground=ground
    )


def local_points(points: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    d = points[:, :3].astype(np.float64) - center
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    rot = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], np.float64)
    return d @ rot.T


def box_mask(points: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    local = local_points(points, center, yaw)
    hx, hy, hz = CAR_L * 0.5, CAR_W * 0.5, CAR_H * 0.5
    return (
        (np.abs(local[:, 0]) <= hx)
        & (np.abs(local[:, 1]) <= hy)
        & (np.abs(local[:, 2]) <= hz)
    )


def count_points_in_car_box(points: np.ndarray, center: np.ndarray, yaw: float) -> int:
    return int(box_mask(points, center, yaw).sum())


# Vertical slack, metres, for the footprint test below: how far the box height may be wrong
# before we stop believing a cluster belongs to the opponent. 2.0 m covers the worst measured
# GroundSurface error (~1.0 m at lvms_2025_01_06) with margin, without reaching an overpass.
FOOTPRINT_Z_SLACK = 2.0
# A return this far above the local ground is standing on it rather than part of it.
FOOTPRINT_CLEARANCE = 0.30
# Elevated footprint returns needed before we call the space occupied-but-unlabelable.
# Calibrated on both populations: 81 frames of genuine occlusion (pp/purdue, ims_2024_09_06,
# laguna, caltech) score exactly 0, while lvms_2025_01_06/unimore and
# lvms_2025_01_07/polimove_s2 -- where a car is demonstrably present -- have medians of 64
# and 69. Any threshold in the tens separates them; 20 keeps the false-positive rate at zero
# while still leaving genuinely sparse frames silent.
DONTCARE_FOOTPRINT_MIN = 20


def count_elevated_in_footprint(points: np.ndarray, center: np.ndarray, yaw: float) -> int:
    """Points standing above the ground inside the box's PLANAR outline, ignoring box height.

    Why this exists: a box whose z is wrong holds no points at all, and the builder then
    reads zero as "nothing is there" and writes no label -- teaching the model that a
    plainly visible car is background. That is how lvms_2025_01_06/unimore lost ~645 labels
    while 1372 points sat inside its footprint: GroundSurface sampled the ego's height from
    7.5 m away across banked track, putting the box 1.0 m too high, entirely above the car.

    This separates "the box is in the wrong place" from "there is nothing to label", so the
    former can become DontCare instead of silence. It deliberately does NOT try to fix the
    height -- it only refuses to claim the space is empty.

    The ground must be excluded or this counts tarmac: a bare planar footprint always holds
    road returns. So the local ground is taken as the 5th percentile inside the footprint and
    only returns at least FOOTPRINT_CLEARANCE above it are counted.
    """
    local = local_points(points, center, yaw)
    hx, hy = CAR_L * 0.5, CAR_W * 0.5
    foot = (
        (np.abs(local[:, 0]) <= hx)
        & (np.abs(local[:, 1]) <= hy)
        & (np.abs(local[:, 2]) <= CAR_H * 0.5 + FOOTPRINT_Z_SLACK)
    )
    if int(foot.sum()) < 10:
        return 0
    z = local[foot, 2]
    ground = float(np.percentile(z, 5))
    return int((z > ground + FOOTPRINT_CLEARANCE).sum())


def local_background_contrast(points: np.ndarray, center: np.ndarray, yaw: float) -> tuple[int, float]:
    local = local_points(points, center, yaw)
    hx, hy, hz = CAR_L * 0.5, CAR_W * 0.5, CAR_H * 0.5
    inside = (
        (np.abs(local[:, 0]) <= hx)
        & (np.abs(local[:, 1]) <= hy)
        & (np.abs(local[:, 2]) <= hz)
    )
    bg_counts = []
    for lateral_offset in (-3.0, 3.0):
        bg_counts.append(
            int(
                (
                    (np.abs(local[:, 0]) <= hx)
                    & (np.abs(local[:, 1] - lateral_offset) <= hy)
                    & (np.abs(local[:, 2]) <= hz)
                ).sum()
            )
        )
    return int(inside.sum()), float(np.mean(bg_counts))


def make_label_line(center: np.ndarray, yaw: float, cls: str = "Car") -> str:
    """One KITTI label line for a box, as ``cls``.

    ``cls="DontCare"`` marks a box we know holds a car but cannot label reliably, because
    too few lidar points landed in it. That is not the same as "no car here": KITTI
    training drops DontCare boxes from the targets, and KITTI eval excludes detections
    overlapping them from the false-positive count. Silently omitting such a box instead
    would both teach the model that an occupied place is empty and then penalise it for
    detecting the car anyway.
    """
    x_v, y_v, z_v = center
    z_bottom = z_v - CAR_H * 0.5
    x_c, y_c, z_c = -y_v, -z_bottom, x_v
    ry = wrap_angle(-yaw - math.pi * 0.5)
    alpha = wrap_angle(ry - math.atan2(x_c, z_c))
    return (
        f"{cls} 0.00 0 {alpha:.4f} 0.00 0.00 50.00 50.00 "
        f"{CAR_H:.4f} {CAR_W:.4f} {CAR_L:.4f} {x_c:.4f} {y_c:.4f} {z_c:.4f} {ry:.4f}"
    )
