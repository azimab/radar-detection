#!/usr/bin/env python3
"""Build a KITTI-style single-class PointPillars dataset from configured MCAP inputs."""
from __future__ import annotations

import argparse
import io
import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image

import pp_kitti_common as pp
import pp_run_manifest

_KITTI_DUMMY_BYTES: bytes | None = None

# Only odometry is cached, never points.
#
# The original cache stored every decoded cloud in one npz. That works for a single
# 34-minute run and does not survive this dataset: the seven configured runs hold 712 GB
# of lidar MCAP, which is roughly 150-240 GB as Nx4 float32, and the largest single run
# (LVMS 2025_01_06, 268 GB of MCAP) would need ~60-90 GB held in a Python list before
# np.savez could write it -- on a 64 GB job. Points are therefore streamed straight to
# KITTI .bin, which is the same data the cache would have held anyway.
#
# Odometry is small (a few million poses per run at 56 B each), reused by every opponent
# box in the run, and is what makes a rebuild cheap. That part of Jay's cache layer is
# kept, fingerprinting included.
ODOM_CACHE_VERSION = 1

COUNTER_KEYS = (
    "out", "label", "frames_with_label", "empty_low_point_train", "empty_low_point_val",
    "guard", "no_pose_time", "time", "empty", "low_points", "seen",
    # Boxes written as DontCare: a car is there and returned points, but too few to label.
    "dontcare_boxes", "dontcare_footprint",
    # Frames skipped entirely because every opponent in them was under-observed. They
    # cannot be labelled and must not be reused as negatives either.
    "ambiguous_frames_not_used_as_negative",
)


def _dummy_png() -> bytes:
    global _KITTI_DUMMY_BYTES
    if _KITTI_DUMMY_BYTES is None:
        buf = io.BytesIO()
        Image.new("L", (1242, 375), 0).save(buf, "PNG")
        _KITTI_DUMMY_BYTES = buf.getvalue()
    return _KITTI_DUMMY_BYTES


def split_assignment(t_rel: float, total: float, cfg: pp.RunConfig) -> str:
    if t_rel >= total * (1.0 - cfg.test_fraction):
        return "test"
    inside_chunk = t_rel - int(t_rel // cfg.chunk_seconds) * cfg.chunk_seconds
    train_end = cfg.train_fraction * cfg.chunk_seconds
    val_start = train_end + cfg.guard_seconds
    if inside_chunk < train_end - cfg.guard_seconds * 0.5:
        return "train"
    if inside_chunk >= val_start:
        return "val"
    return ""


def clean_output(root: Path) -> None:
    for split in ("training", "testing"):
        target = root / split
        if target.is_dir():
            shutil.rmtree(target)
    # Frame ids are reallocated on every build, so a surviving map would silently pair new
    # frames with old timestamps -- worse than having no map at all.
    (root / "frame_times.json").unlink(missing_ok=True)


def ensure_dirs(root: Path) -> None:
    for subdir in (
        "training/velodyne",
        "training/label_2",
        "training/calib",
        "training/image_2",
        "testing/velodyne",
        "testing/calib",
        "testing/image_2",
        "ImageSets",
    ):
        (root / subdir).mkdir(parents=True, exist_ok=True)


def order_label_lines(label_lines: list[str]) -> list[str]:
    """Real classes first, DontCare last -- required, not cosmetic.

    OpenPCDet's KittiDataset.get_infos takes ``location[:num_objects]`` where
    ``num_objects`` counts only non-DontCare entries, i.e. it assumes the real objects
    occupy the *first* rows. With a DontCare row first it builds gt_boxes_lidar from the
    DontCare's location and then IndexErrors in drop_info_with_name, because gt_boxes_lidar
    is shorter than the kept indices. Emitting boxes in opponent order produced 543 such
    frames and made the dataset unloadable.
    """
    return [l for l in label_lines if not l.startswith("DontCare")] + [
        l for l in label_lines if l.startswith("DontCare")
    ]


def write_frame(root: Path, frame_id: str, points: np.ndarray, label_lines: list[str]) -> None:
    label_lines = order_label_lines(label_lines)
    (root / "velodyne" / f"{frame_id}.bin").write_bytes(points.astype(np.float32).tobytes())
    (root / "calib" / f"{frame_id}.txt").write_text(pp.CALIB_TEXT)
    (root / "image_2" / f"{frame_id}.png").write_bytes(_dummy_png())
    if label_lines:
        (root / "label_2" / f"{frame_id}.txt").write_text("\n".join(label_lines) + "\n")
    elif (root / "label_2").is_dir():
        (root / "label_2" / f"{frame_id}.txt").write_text("")


def write_imageset(path: Path, frame_ids: list[str]) -> None:
    text = "\n".join(frame_ids)
    path.write_text(text + ("\n" if text else ""))


def keep_empty_low_point_frame(
    split: str,
    cfg: pp.RunConfig,
    positive_counts: dict[str, int],
    negative_counts: dict[str, int],
) -> bool:
    target_fraction = cfg.empty_train_fraction if split == "train" else cfg.empty_val_fraction if split == "val" else 1.0
    target_fraction = min(max(float(target_fraction), 0.0), 1.0)
    if split == "test":
        return True
    if target_fraction <= 0.0:
        return False
    if target_fraction >= 1.0:
        return True
    max_negatives = int(np.floor(positive_counts[split] * target_fraction / max(1e-6, 1.0 - target_fraction)))
    return negative_counts[split] < max_negatives


def _iter_clouds_in_windows(reader, topics: list[str], windows: list[tuple[float, float]]):
    """Decoded messages on ``topics`` inside each window, in order, no duplicates.

    Windows are merged and sorted, and mcap_window_args pads each end by 2 s while
    merge_windows only keeps gaps wider than that -- but the stamp guard makes a
    duplicated frame impossible regardless, which matters because a duplicate would
    become a second KITTI frame of the same instant.
    """
    previous = -np.inf
    for win_lo, win_hi in windows:
        for _, _, _, msg in reader.iter_decoded_messages(
            topics=topics, **pp.mcap_window_args(win_lo, win_hi)
        ):
            stamp = pp.stamp_seconds(msg)
            if stamp <= previous:
                continue
            previous = stamp
            yield msg


def iter_prefused_clouds(cfg: pp.RunConfig, run: pp.DatasetRun, windows: list[tuple[float, float]]):
    """Yield (t, Nx4 points) for a run that already publishes a fused cloud."""
    first_cloud_frame_id = ""
    point_dtype = None
    frame_z_offset = 0.0
    for fused in pp.sort_mcaps_chronologically(pp.fused_mcap_paths(run.fused_lidar)):
        pp.log_kv("fused_decode", fused)
        with open(fused, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            # One indexed read per window, so the reader seeks past everything between
            # them instead of decoding clouds that cannot produce a label.
            for cloud in _iter_clouds_in_windows(reader, [cfg.fused_lidar_topic], windows):
                if point_dtype is None:
                    point_dtype = pp.build_point_dtype(cloud)
                    first_cloud_frame_id = str(cloud.header.frame_id)
                    # Fails loudly on an unknown frame instead of assuming LIDAR_FRAME.
                    frame_z_offset = pp.cloud_frame_z_offset(first_cloud_frame_id)
                    pp.log_kvs(
                        "cloud frame",
                        [
                            ("frame_id", first_cloud_frame_id),
                            ("point_step", int(cloud.point_step)),
                            ("z_offset_to_" + pp.LIDAR_FRAME, f"{frame_z_offset:+.3f}"),
                        ],
                    )
                elif str(cloud.header.frame_id) != first_cloud_frame_id:
                    raise ValueError(
                        f"lidar frame_id changed mid-run: {first_cloud_frame_id!r} -> "
                        f"{str(cloud.header.frame_id)!r} in {fused}; refusing to mix "
                        "vertical datums within one run"
                    )
                points = pp.parse_fused_cloud(cloud, point_dtype)
                if points is None:
                    points = np.zeros((0, 4), dtype=np.float32)
                elif frame_z_offset:
                    points = points.copy()
                    points[:, 2] += frame_z_offset
                yield pp.stamp_seconds(cloud), points


def iter_raw_fused_clouds(run: pp.DatasetRun, fusion_stats: dict, windows: list[tuple[float, float]]):
    """Yield (t, Nx4 points) fused from a run's raw /pcl_front|left|right, in memory.

    Fusion resolves the session's own tf_static straight into the label frame, so no
    frame_id fixup applies afterwards -- that is the whole point (PIPELINE_PLAN.md 1.4).
    Fused clouds are never written to disk: at ~3.5 MB/frame the intermediates for these
    runs would be ~300 GB (see PIPELINE_PLAN.md 4).
    """
    # Sensor placement comes from the URDF of record, not from the bag: a bag's tf_static
    # is whatever the car was publishing that day and can be wrong. The bag is still read,
    # to confirm the session is av24 at all and to report any disagreement -- overriding
    # silently is what this pipeline keeps getting bitten by.
    tf_paths = pp.mcap_paths(run.tf_static_paths, "tf_static")
    rig, _ = pp.identify_rig(tf_paths)
    if rig != "av24":
        raise RuntimeError(
            f"{run.name}: lidar rig is {rig!r}, not 'av24'. Only av24 data is in scope; "
            "remove this run from the config."
        )
    transforms = pp.load_urdf_transforms(pp.AV24_URDF, target=pp.LIDAR_FRAME)
    diffs = pp.compare_transforms(transforms, pp.load_static_transforms(tf_paths, target=pp.LIDAR_FRAME))
    pp.log_kvs(
        "raw fusion (av24 URDF)",
        [("target_frame", pp.LIDAR_FRAME)]
        + [(f"{frame}_t", np.round(t, 4).tolist()) for frame, (_, t) in sorted(transforms.items())]
        + [
            (f"{frame}_bag_delta", f"{dt * 100:.2f} cm / {dr:.3f} deg")
            for frame, (dt, dr) in sorted(diffs.items())
        ],
    )
    lidar_paths = pp.sort_mcaps_chronologically(pp.fused_mcap_paths(run.fused_lidar))
    pp.log_kv("fused_decode", f"{len(lidar_paths)} bag(s) in recording order")
    # One pass per window. Each pass re-warms the flank sensors before it emits anything
    # (see iter_fused_raw_frames), which costs a frame or two at each window edge -- the
    # 2 s of padding around every window absorbs that.
    previous = -np.inf
    for win_lo, win_hi in windows:
        for t, points in pp.iter_fused_raw_frames(
            lidar_paths, transforms, stats=fusion_stats, t_lo=win_lo, t_hi=win_hi
        ):
            if t <= previous:
                continue
            previous = t
            yield t, points


def iter_run_clouds(cfg: pp.RunConfig, run: pp.DatasetRun, fusion_stats: dict, windows):
    if run.fuse_raw:
        return iter_raw_fused_clouds(run, fusion_stats, windows)
    return iter_prefused_clouds(cfg, run, windows)


def format_labels_per_frame(hist: Counter) -> str:
    """e.g. "1x:412 2x:83 3x:6" -- how often multiple opponents were visible at once."""
    if not hist:
        return "none"
    return " ".join(f"{n}x:{hist[n]}" for n in sorted(hist))


def load_or_build_odom_cache(cfg: pp.RunConfig, run: pp.DatasetRun, *, dry_run: bool = False) -> dict[str, Any]:
    path = pp.cache_path(cfg, f"{run.name}_odom")
    source_paths = [*run.ego_odom, *run.opponent_source_paths]
    fingerprint = pp.source_fingerprint(source_paths)
    if cfg.cache_enabled:
        cached = pp.load_npz_cache(path, expected_fingerprint=fingerprint, expected_version=ODOM_CACHE_VERSION)
        # The fingerprint only covers source files, so a config edit that changes which
        # opponents a run has -- reordering them, dropping one -- leaves the files
        # untouched and would otherwise reuse timelines belonging to different cars.
        if cached is not None:
            names = [str(n) for n in cached.get("opponent_names", [])]
            if names == [o.name for o in run.opponents]:
                return cached
            pp.log_kvs("cache stale", [("path", path), ("cached_opponents", names)])

    start = time.time()
    pp.log_kvs("cache miss", [("path", path), ("what", "ego + opponent odometry")])
    ego = pp.load_odom_timeline(run.ego_odom, cfg.ego_odom_topic)
    cache = {
        "ego_t": ego.t,
        "ego_xyz": ego.xyz,
        "ego_quat_xyzw": ego.quat_xyzw,
        "opponent_names": np.asarray([o.name for o in run.opponents], dtype=object),
    }
    for i, opp in enumerate(run.opponents):
        timeline = pp.load_opponent_timeline(opp.odom, opp.odom_topic)
        cache[f"opponent{i}_t"] = timeline.t
        cache[f"opponent{i}_xyz"] = timeline.xyz
        cache[f"opponent{i}_quat_xyzw"] = timeline.quat_xyzw
    if cfg.cache_enabled and not dry_run:
        pp.write_npz_cache(path, cache, source_fingerprint=fingerprint, version=ODOM_CACHE_VERSION)
        pp.log_kvs("cache wrote", [("path", path), ("elapsed_seconds", f"{time.time() - start:.1f}")])
    return cache


def run_timelines(
    cfg: pp.RunConfig, run: pp.DatasetRun, *, dry_run: bool = False
) -> tuple[pp.PoseTimeline, list[tuple[pp.Opponent, pp.PoseTimeline]]]:
    """Ego and per-opponent pose timelines, datum offsets already applied.

    The single place run geometry is assembled, so the diagnostics in pp_probe_boxes.py
    are guaranteed to be measuring the same boxes this file writes as labels.
    """
    raw = load_or_build_odom_cache(cfg, run, dry_run=dry_run)
    ego_quat = raw["ego_quat_xyzw"]
    if run.ego_yaw_correction_deg:
        # Rotating the ego pose is exactly equivalent to re-rotating `lidar_plate`, since
        # all three iris sensors hang off it -- and it is applied here, after the cache,
        # so retuning the angle never forces an odometry re-decode.
        ego_quat = pp.apply_yaw_correction(ego_quat, run.ego_yaw_correction_deg)
    ego = pp.PoseTimeline.from_arrays(raw["ego_t"], raw["ego_xyz"], ego_quat)
    pp.log_kv("ego_samples", f"{len(ego.t)} [{ego.t_start:.3f} .. {ego.t_end:.3f}]")
    pp.log_kv("ego_yaw_correction_deg", f"{run.ego_yaw_correction_deg:+.3f}")

    opponents: list[tuple[pp.Opponent, pp.PoseTimeline]] = []
    for i, opp in enumerate(run.opponents):
        xyz = np.asarray(raw[f"opponent{i}_xyz"], dtype=np.float64).copy()
        if opp.z_offset:
            # Applied after the cache, not before it, so retuning a datum offset does not
            # force a re-decode of the odometry bags.
            xyz[:, 2] -= opp.z_offset
        if any(opp.map_offset_xyz):
            xyz += np.asarray(opp.map_offset_xyz, dtype=np.float64)
        # Orientation guard runs here, after the datum offsets, for two reasons: the
        # offsets are pure translations so they cannot affect a heading either way, and
        # this is the one place run geometry is assembled -- so probe, build and visualize
        # all see the same repaired timeline, and none of them can drift from the labels.
        timeline, repair = pp.opponent_timeline_repaired(
            raw[f"opponent{i}_t"],
            xyz,
            raw[f"opponent{i}_quat_xyzw"],
            label=f"{run.name} / {opp.name} ({opp.odom_topic})",
        )
        opponents.append((opp, timeline))
        pp.log_kvs(
            f"opponent[{i}] {opp.name}",
            [
                ("topic", opp.odom_topic),
                ("samples", f"{len(timeline.t)} [{timeline.t_start:.3f} .. {timeline.t_end:.3f}]"),
                ("time_offset_seconds", f"{opp.time_offset_seconds:+.3f}"),
                ("z_offset", f"{opp.z_offset:+.3f}"),
                ("map_offset_xyz", f"{opp.map_offset_xyz}"),
                ("orientation", repair or "as recorded"),
            ],
        )
    return ego, opponents


def build_one_run(
    cfg: pp.RunConfig,
    run: pp.DatasetRun,
    frame_start: int,
    dry_run: bool,
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    pp.log_section(f"run: {run.name}")
    ego, opponents = run_timelines(cfg, run, dry_run=dry_run)

    # Frames need the ego pose plus at least *one* opponent, so the usable window is the
    # ego's range intersected with the union of the opponents' ranges. Opponents that
    # were on track for only part of the run (the three separate POLIMOVE sessions in
    # LVMS run3_fourcar) simply return no box outside their own span.
    opp_lo = min(t.t_start - o.time_offset_seconds for o, t in opponents)
    opp_hi = max(t.t_end - o.time_offset_seconds for o, t in opponents)
    t_lo = max(ego.t_start, opp_lo)
    t_hi = min(ego.t_end, opp_hi)
    duration = t_hi - t_lo
    if duration <= 0:
        raise RuntimeError(f"{run.name}: no overlapping ego/opponent time range after time offset")

    for opp, timeline in opponents:
        sample_lo = max(t_lo, timeline.t_start - opp.time_offset_seconds) + 1.0
        sample_hi = min(t_hi, timeline.t_end - opp.time_offset_seconds) - 1.0
        if sample_hi <= sample_lo:
            pp.log_kv(f"{opp.name}_ego_opp_map_distance_m", "no overlap with ego")
            continue
        sample_t = np.linspace(sample_lo, sample_hi, 50)
        dist = np.linalg.norm(
            ego.pose_at(sample_t)[:, :3, 3]
            - timeline.pose_at(sample_t + opp.time_offset_seconds)[:, :3, 3],
            axis=1,
        )
        pp.log_kv(
            f"{opp.name}_ego_opp_map_distance_m_min_med_max",
            f"{dist.min():.1f}/{float(np.median(dist)):.1f}/{dist.max():.1f}",
        )

    ground = pp.GroundSurface(ego) if run.ground_from_ego else None
    pp.log_kv("box_ground_height_from", "ego driven surface" if ground is not None else "opponent reported z")

    train_ids, val_ids, test_ids = [], [], []
    frame_times: dict[str, float] = {}
    stats: dict[str, Any] = {k: 0 for k in COUNTER_KEYS}
    stats["labels_per_frame"] = Counter()
    positive_counts = {"train": 0, "val": 0}
    negative_counts = {"train": 0, "val": 0}
    fusion_stats: dict[str, int] = {}

    # Decide what is worth reading *before* opening a single cloud. Odometry already tells
    # us when an opponent was close enough for a box to survive POINT_CLOUD_RANGE; the
    # rest of the run can only ever yield negatives, of which a bounded number is needed.
    windows, kept_fraction = pp.opponent_range_windows(
        ego, opponents, t_lo, t_hi,
        gate_m=cfg.opponent_range_gate_m,
        far_sample_seconds=cfg.far_sample_seconds,
        far_sample_period_seconds=cfg.far_sample_period_seconds,
    )
    pp.log_kvs(
        "decode gating",
        [
            ("opponent_range_gate_m", cfg.opponent_range_gate_m or "disabled"),
            ("windows", len(windows)),
            ("run_time_decoded", f"{100.0 * kept_fraction:.1f}%"),
        ],
    )

    # Streamed, never accumulated: see the ODOM_CACHE_VERSION note. Only one cloud is
    # resident at a time, so peak memory is independent of run length.
    for lidar_index, (lidar_time, points) in enumerate(iter_run_clouds(cfg, run, fusion_stats, windows)):
        stats["seen"] += 1
        if lidar_index % max(1, cfg.build_stride) != 0:
            continue
        if cfg.build_max_frames > 0 and stats["out"] >= cfg.build_max_frames:
            break

        lidar_time = float(lidar_time)
        if lidar_time < t_lo or lidar_time > t_hi:
            stats["time"] += 1
            continue

        split = split_assignment(lidar_time - t_lo, duration, cfg)
        if not split:
            stats["guard"] += 1
            continue

        if len(points) < 100:
            stats["empty"] += 1
            continue

        label_lines: list[str] = []
        if split != "test":
            # Each opponent is scored on its own point count, three ways:
            #
            #   >= MIN_POINTS_IN_OPP  -> a Car label
            #   1 .. MIN-1            -> DontCare: a car is demonstrably there (points
            #                            landed in the box) but too sparsely sampled to
            #                            label. Dropping it would teach the model that an
            #                            occupied place is empty, and then penalise it at
            #                            eval for detecting the car anyway.
            #   0                     -> nothing observable; safe to treat as background.
            n_boxes = 0
            ambiguous = False
            for opp, timeline in opponents:
                box = pp.opponent_box_at(lidar_time, ego, timeline, opp.time_offset_seconds, ground=ground)
                if box is None:
                    continue
                n_boxes += 1
                center, yaw = box
                n_points = pp.count_points_in_car_box(points, center, yaw)
                if n_points >= pp.MIN_POINTS_IN_OPP:
                    label_lines.append(pp.make_label_line(center, yaw))
                elif n_points > 0:
                    label_lines.append(pp.make_label_line(center, yaw, cls="DontCare"))
                    ambiguous = True
                elif pp.count_elevated_in_footprint(points, center, yaw) >= pp.DONTCARE_FOOTPRINT_MIN:
                    # Zero points in the box, yet a cluster stands inside its planar outline:
                    # the box is in the wrong PLACE, not looking at empty track. Treating that
                    # as background is what cost lvms_2025_01_06/unimore ~645 labels while
                    # 1372 points sat in its footprint -- GroundSurface sampled the ego's
                    # height 7.5 m away across banked track and put the box 1.0 m too high.
                    #
                    # DontCare is the honest answer: we cannot place the box, but we must not
                    # claim the space is empty and then penalise the model at eval for seeing
                    # the car that is plainly there. It costs a training target and protects
                    # against a false negative, which is the right way round -- a box moved
                    # onto the road would teach the model that tarmac is a car.
                    label_lines.append(pp.make_label_line(center, yaw, cls="DontCare"))
                    ambiguous = True
                    stats["dontcare_footprint"] += 1
            if n_boxes == 0:
                # No opponent had a pose at this lidar time at all.
                stats["no_pose_time"] += 1
                continue

            n_car = sum(1 for line in label_lines if line.startswith("Car "))
            if n_car:
                stats["label"] += n_car
                stats["frames_with_label"] += 1
                stats["labels_per_frame"][n_car] += 1
                positive_counts[split] += 1
                # Counted here, not where the box is built: a DontCare box only reaches
                # disk if its frame also holds a Car label, since an all-ambiguous frame
                # is dropped below. Counting earlier reported DontCare labels that were
                # then discarded with their frame.
                stats["dontcare_boxes"] += len(label_lines) - n_car
            else:
                stats["low_points"] += 1
                # A frame holding an under-observed car must not be reused as a negative:
                # its "emptiness" is a labelling limitation, not an empty scene. Only
                # frames where every opponent box came back with zero points qualify.
                if ambiguous:
                    stats["ambiguous_frames_not_used_as_negative"] += 1
                    continue
                if not keep_empty_low_point_frame(split, cfg, positive_counts, negative_counts):
                    continue
                negative_counts[split] += 1
                stats[f"empty_low_point_{split}"] += 1

        frame_id = f"{frame_start + stats['out']:06d}"
        split_root = cfg.data_root / ("testing" if split == "test" else "training")
        if not dry_run:
            write_frame(split_root, frame_id, points, label_lines)
        (train_ids if split == "train" else val_ids if split == "val" else test_ids).append(frame_id)
        # frame_id -> the lidar timestamp its box was sampled at. Costs 30 bytes a frame
        # and is the difference between auditing a label and re-decoding to find it: the
        # identity-quaternion bug could only be *measured* offline because the odom caches
        # survived, and could only be *repaired* by a full re-decode because this mapping
        # did not exist. Any future geometry fix -- a clock offset, a datum -- can now be
        # recomputed against the existing labels without touching a cloud.
        frame_times[frame_id] = lidar_time
        stats["out"] += 1

    if fusion_stats:
        pp.log_kvs("raw fusion yield", sorted(fusion_stats.items()))

    # Per-run yield, so one bad session is visible instead of averaged into the totals.
    pp.log_kvs(
        f"{run.name} yield",
        [
            ("clouds_seen", stats["seen"]),
            ("frames_written", f"{stats['out']} ({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)"),
            ("labels", f"{stats['label']} across {stats['frames_with_label']} frames"),
            ("labels_per_frame", format_labels_per_frame(stats["labels_per_frame"])),
            ("kept_negative_frames", stats["empty_low_point_train"] + stats["empty_low_point_val"]),
            ("skipped_split_guard_gap", stats["guard"]),
            ("skipped_outside_pose_time", stats["time"] + stats["no_pose_time"]),
            ("skipped_sparse_cloud", stats["empty"]),
            ("skipped_low_points_in_box", f"{stats['low_points']} (<{pp.MIN_POINTS_IN_OPP} pts)"),
            ("dontcare_boxes", f"{stats['dontcare_boxes']} (1..{pp.MIN_POINTS_IN_OPP - 1} pts)"),
            ("dontcare_footprint", f"{stats['dontcare_footprint']} (box empty but "
                                   f">={pp.DONTCARE_FOOTPRINT_MIN} elevated pts in footprint)"),
            ("ambiguous_frames_not_used_as_negative", stats["ambiguous_frames_not_used_as_negative"]),
        ],
    )
    if not dry_run:
        times_path = cfg.data_root / "frame_times.json"
        existing = json.loads(times_path.read_text()) if times_path.exists() else {}
        existing.update(frame_times)
        times_path.write_text(json.dumps(existing, indent=1, sort_keys=True))
        pp.log_kv("frame_times", f"{len(frame_times)} written to {times_path}")

    return train_ids, val_ids, test_ids, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    pp.add_run_config_arg(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only-runs",
        default="",
        help="comma-separated run names to rebuild IN PLACE, leaving every other run's "
        "frames untouched. Their old frames are deleted and new ones appended past the "
        "current highest frame id; ImageSets and run_manifest.json are merged rather "
        "than rewritten. Ignores dataset.clean. Use when a fix touches only some runs -- "
        "a full rebuild re-decodes ~116 GB to change 19% of it.",
    )
    args = parser.parse_args()
    args.run_cfg = pp.load_run_config(args.run_config)
    args.only_runs = [r.strip() for r in args.only_runs.split(",") if r.strip()]
    if args.only_runs:
        known = {r.name for r in args.run_cfg.runs}
        unknown = set(args.only_runs) - known
        if unknown:
            raise SystemExit(f"--only-runs: unknown run(s) {sorted(unknown)}; known: {sorted(known)}")
        disabled = [r.name for r in args.run_cfg.runs if r.name in args.only_runs and not r.enabled]
        if disabled:
            raise SystemExit(
                f"--only-runs names disabled run(s) {disabled}; set enabled: true first, "
                "otherwise the rebuild would delete their frames and write nothing back"
            )
    return args


def load_existing_dataset(out_root: Path) -> tuple[dict, dict[str, list[str]]]:
    """Current run_manifest and ImageSets, for a partial rebuild to merge into."""
    manifest_path = out_root / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"--only-runs needs an existing dataset, but {manifest_path} is missing. "
            "Run a full build first."
        )
    manifest = json.loads(manifest_path.read_text())
    sets: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        path = out_root / "ImageSets" / f"{split}.txt"
        sets[split] = path.read_text().split() if path.exists() else []
    return manifest, sets


def drop_run_frames(out_root: Path, manifest: dict, run_names: list[str], dry_run: bool) -> int:
    """Delete the on-disk frames of the named runs. Returns how many were removed.

    Deliberately driven by the manifest's recorded id ranges rather than by re-deriving
    them: the manifest is the only record of which frame id belonged to which run, and if
    it is stale the rebuild must fail loudly rather than delete another run's frames.
    """
    removed = 0
    for entry in manifest.get("runs", []):
        if entry["run"] not in run_names:
            continue
        for idx in range(entry["frame_start"], entry["frame_end"] + 1):
            frame_id = f"{idx:06d}"
            for tree in ("training", "testing"):
                for sub, ext in (("velodyne", "bin"), ("label_2", "txt"), ("calib", "txt"), ("image_2", "png")):
                    path = out_root / tree / sub / f"{frame_id}.{ext}"
                    if path.exists():
                        if not dry_run:
                            path.unlink()
                        removed += 1
    return removed


def main() -> None:
    args = parse_args()
    cfg = args.run_cfg
    start = time.time()
    pp.configure_from_run_config(cfg)

    pp.log_section("config")
    pp.log_kv("model_cfg", cfg.model_cfg)
    pp.log_kv("point_cloud_range", f"{pp.POINT_CLOUD_RANGE.tolist()} (used by training; export keeps full clouds)")
    pp.log_kv("car_box_lwh_m", f"{pp.CAR_L:.4f}/{pp.CAR_W:.4f}/{pp.CAR_H:.4f}")
    pp.log_kv("gt_box_bottom_z", "odom-derived opponent ground")
    pp.log_kv("min_points_in_opponent_box", pp.MIN_POINTS_IN_OPP)
    pp.log_kv("empty_train_fraction", cfg.empty_train_fraction)
    pp.log_kv("empty_val_fraction", cfg.empty_val_fraction)
    pp.log_kv("cache_enabled", cfg.cache_enabled)
    pp.log_kv("cache_dir", cfg.cache_dir)

    out_root = cfg.data_root
    out_root.mkdir(parents=True, exist_ok=True)

    kept_manifest: list[dict[str, Any]] = []
    kept_sets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    id_base = 0
    if args.only_runs:
        # Partial rebuild. New frames go PAST the current highest id rather than reusing
        # the run's old range: the fixes that motivate a partial rebuild change box
        # geometry, which changes which frames clear the >=5-point gate, so the run's
        # frame count changes and its old range no longer fits.
        manifest, sets = load_existing_dataset(out_root)
        rebuilt = set(args.only_runs)
        kept_manifest = [e for e in manifest["runs"] if e["run"] not in rebuilt]
        dropped_ids = {
            f"{i:06d}"
            for e in manifest["runs"]
            if e["run"] in rebuilt
            for i in range(e["frame_start"], e["frame_end"] + 1)
        }
        for split in kept_sets:
            kept_sets[split] = [f for f in sets[split] if f not in dropped_ids]
        id_base = max((e["frame_end"] for e in manifest["runs"]), default=-1) + 1
        removed = drop_run_frames(out_root, manifest, args.only_runs, args.dry_run)
        pp.log_kvs(
            "partial rebuild",
            [
                ("rebuilding", ", ".join(args.only_runs)),
                ("keeping", f"{len(kept_manifest)} run(s), {sum(len(v) for v in kept_sets.values())} frames"),
                ("deleted_files", removed),
                ("new_frame_ids_start_at", id_base),
                ("dataset.clean", "ignored (would destroy the runs being kept)"),
            ],
        )
    elif not args.dry_run and cfg.clean_dataset:
        clean_output(out_root)
    if not args.dry_run:
        ensure_dirs(out_root)

    train_ids, val_ids, test_ids = [], [], []
    totals: dict[str, Any] = {k: 0 for k in COUNTER_KEYS}
    totals["labels_per_frame"] = Counter()
    # Frame ids are global and sequential, so each run owns one contiguous range.
    # Recording it is the only way a frame id can later be traced back to its run --
    # needed for per-track evaluation and for sampling frames to render.
    manifest_runs: list[dict[str, Any]] = []
    for run in cfg.runs:
        if not run.enabled:
            pp.log_section(f"run: {run.name}")
            pp.log_kv("SKIPPED", "enabled: false in the run config")
            continue
        if args.only_runs and run.name not in args.only_runs:
            pp.log_section(f"run: {run.name}")
            pp.log_kv("SKIPPED", "not in --only-runs; its existing frames are kept as-is")
            continue
        frame_start = id_base + totals["out"]
        run_train, run_val, run_test, stats = build_one_run(cfg, run, frame_start, args.dry_run)
        train_ids.extend(run_train)
        val_ids.extend(run_val)
        test_ids.extend(run_test)
        for key, value in stats.items():
            totals[key] += value
        manifest_runs.append(
            {
                "run": run.name,
                "track": pp_run_manifest.track_for(run.name),
                "frames": stats["out"],
                "train": len(run_train),
                "val": len(run_val),
                "test": len(run_test),
                "frame_start": frame_start,
                "frame_end": frame_start + stats["out"] - 1,
            }
        )

    # Merge with whatever a partial rebuild kept; both lists are empty on a full build,
    # so this is the same code path either way.
    all_train = sorted(kept_sets["train"] + train_ids)
    all_val = sorted(kept_sets["val"] + val_ids)
    all_test = sorted(kept_sets["test"] + test_ids)
    all_manifest = sorted(kept_manifest + manifest_runs, key=lambda e: e["frame_start"])
    frames_total = sum(e["frames"] for e in all_manifest)

    overlap = set(all_train) & set(all_val) | set(all_train) & set(all_test) | set(all_val) & set(all_test)
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} frame id(s) appear in more than one split after merging, e.g. "
            f"{sorted(overlap)[:5]} -- refusing to write a dataset that would leak between splits"
        )

    if not args.dry_run:
        write_imageset(out_root / "ImageSets" / "train.txt", all_train)
        write_imageset(out_root / "ImageSets" / "val.txt", all_val)
        write_imageset(out_root / "ImageSets" / "test.txt", all_test)
        (out_root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "source": "written by build_pp_kitti.py",
                    "run_config": str(cfg.path),
                    "data_root": str(out_root),
                    "frames_total": frames_total,
                    "partial_rebuild_of": args.only_runs or None,
                    "runs": all_manifest,
                },
                indent=2,
            )
            + "\n"
        )

    pp.log_section("done")
    pp.log_kv("frames", f"{totals['out']} ({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)")
    pp.log_kv("labels", f"{totals['label']} across {totals['frames_with_label']} labelled frames")
    pp.log_kv("labels_per_frame", format_labels_per_frame(totals["labels_per_frame"]))
    pp.log_kv("empty_low_point_train", totals["empty_low_point_train"])
    pp.log_kv("empty_low_point_val", totals["empty_low_point_val"])
    pp.log_kv("skipped_split_guard_gap", totals["guard"])
    pp.log_kv("skipped_outside_ego_opponent_pose_time_overlap", totals["time"] + totals["no_pose_time"])
    pp.log_kv("skipped_empty_sparse_raw_cloud", f"{totals['empty']} (<100 pts)")
    pp.log_kv("skipped_low_points_in_opponent_box", f"{totals['low_points']} (<{pp.MIN_POINTS_IN_OPP} lidar points)")
    pp.log_kv("dontcare_boxes", f"{totals['dontcare_boxes']} (1..{pp.MIN_POINTS_IN_OPP - 1} points, not scored either way)")
    pp.log_kv("dontcare_footprint", f"{totals['dontcare_footprint']} (box held no points but a cluster stood in its footprint: box misplaced, not empty track)")
    pp.log_kv("ambiguous_frames_not_used_as_negative", totals["ambiguous_frames_not_used_as_negative"])
    pp.log_kv("elapsed_seconds", f"{time.time() - start:.1f}")
    if not args.dry_run:
        pp.log_kv("next", "python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos ...")


if __name__ == "__main__":
    main()
