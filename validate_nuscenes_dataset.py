#!/usr/bin/env python3
"""Validate the rear-radar nuScenes dataset against the official nuScenes devkit.

    python3 radar_openpcdet/validate_nuscenes_dataset.py            # full check
    python3 radar_openpcdet/validate_nuscenes_dataset.py --skip-eval

Everything is read back through ``nuscenes-devkit`` (vendored under ``vendor/pylib``),
not through the builder's own code, so a check that passes means a devkit user sees the
same dataset the builder meant to write:

1. schema      both versions load; every token resolves; prev/next chains are
               consistent; every sample has exactly one RADAR_REAR keyframe; every
               instance's first/last/nbr_annotations match its chain.
2. radar       every .pcd loads with ``RadarPointCloud.from_file`` under the devkit's
               *default* filters and keeps every point; mapped back through
               calibrated_sensor its xyz equals the sidecar's ego-frame xyz, and its
               compensated velocity equals the sidecar's v_abs_x / v_abs_y.
3. labels      each annotation, taken to the ego frame by the devkit, holds exactly
               ``num_radar_pts`` sidecar returns (oriented BEV footprint, the builder's
               membership rule); boxes are upright and the car size is the configured
               one.
4. splits      scenes are run-blocked; split sizes and background fractions.
4b. sweeps     every non-key scan sits on its scene's sample_data chain, at most
               ``sweeps.max_lag_s`` before its keyframe, with its files; and background
               keyframes get as many previous scans as positives do (the stacked-scan
               shortcut the 2026-09-28 build had: positives 98% full stacks, background
               4-19%).
5. label QA    per run, how tightly the opponent's moving returns sit on the box, and
               the near-edge gate: a radar only lights the face of the car pointing at
               it, and the opponent is behind us, so the returns' leading edge in box
               coordinates must sit near +L/2. Clutter and multipath only push it
               outward, so a shortfall means the box is displaced along its heading.
               This is what ``box_center_forward_offset_m`` is measured against.
               The URDF gate then checks that each offset implies a box centre the
               AV-24's own landmarks admit, given where that opponent reports its pose
               from (``opponent_pose_origin``). Failing it is a warning: the offset is
               carrying something beyond body geometry.
6. manifest    no selected scan is below the ego-speed gate; front-sector targets
               leave (almost) no returns in their box -- the coverage audit.
7. eval        the devkit's detection eval, pointed at RADAR_REAR, scores the ground
               truth as its own prediction at AP = 1.0 -- proof the tables drive the
               official metric end to end.

Writes ``validation_report.json`` beside the data; exits non-zero on any error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from repo_config import DEFAULT_CONFIG, load_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor" / "pylib"))
sys.path.insert(0, str(HERE))

import av24_geometry  # noqa: E402

from nuscenes.nuscenes import NuScenes  # noqa: E402
from nuscenes.utils.data_classes import RadarPointCloud  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

CHANNEL = "RADAR_REAR"
FEATURE_DIR = "RADAR_REAR_FRGEN21"
# Longitudinal half-window for the near-edge measurement. Without it a barrier on the
# box's own axis anywhere out to 100 m sets the maximum and the gate measures clutter.
NEAR_EDGE_WINDOW_M = 10.0


def sidecar_path(root: Path, filename: str) -> Path:
    return root / filename.replace(f"samples/{CHANNEL}/", f"samples/{FEATURE_DIR}/").replace(".pcd", ".npy")


class Checks:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        if len(self.errors) < 200:
            self.errors.append(message)

    def expect(self, ok: bool, message: str) -> None:
        if not ok:
            self.error(message)


def check_chain(nusc: NuScenes, table: str, first: str, checks: Checks, label: str) -> list[dict]:
    records, token, prev = [], first, ""
    while token:
        record = nusc.get(table, token)
        checks.expect(record["prev"] == prev, f"{label}: {table} {token} prev {record['prev']!r} != {prev!r}")
        records.append(record)
        prev, token = token, record["next"]
    return records


def ego_boxes(nusc: NuScenes, sample: dict) -> tuple[dict, list]:
    sd = nusc.get("sample_data", sample["data"][CHANNEL])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    boxes = []
    for ann_token in sample["anns"]:
        box = nusc.get_box(ann_token)
        box.translate(-np.array(pose["translation"]))
        box.rotate(Quaternion(pose["rotation"]).inverse)
        boxes.append((nusc.get("sample_annotation", ann_token), box))
    return sd, boxes


def points_in_bev_mask(xy: np.ndarray, center: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    d = xy - center[:2]
    c, s = math.cos(-yaw), math.sin(-yaw)
    return (np.abs(c * d[:, 0] - s * d[:, 1]) <= length / 2) & (np.abs(s * d[:, 0] + c * d[:, 1]) <= width / 2)


def points_in_bev_box(xy: np.ndarray, center: np.ndarray, yaw: float, length: float, width: float) -> int:
    return int(points_in_bev_mask(xy, center, yaw, length, width).sum())


def run_eval(nusc: NuScenes, version_splits: list[str], splits: dict, checks: Checks, report: dict) -> None:
    """Devkit detection eval with ground truth as the prediction; expects AP = 1."""
    import nuscenes.eval.common.loaders as loaders
    import nuscenes.eval.detection.evaluate as evaluate
    from nuscenes.eval.detection.constants import DETECTION_NAMES
    from nuscenes.eval.detection.data_classes import DetectionConfig

    # The devkit hardcodes nuScenes' own scene names and the LIDAR_TOP channel; point
    # both at this dataset. Nothing else about the metric is changed.
    ours = lambda verbose=False: {k: v for k, v in splits.items()}
    loaders.create_splits_scenes = ours
    evaluate.create_splits_scenes = ours

    def add_center_dist(nusc, eval_boxes):
        for sample_token in eval_boxes.sample_tokens:
            sd = nusc.get("sample_data", nusc.get("sample", sample_token)["data"][CHANNEL])
            pose = nusc.get("ego_pose", sd["ego_pose_token"])
            for box in eval_boxes[sample_token]:
                box.ego_translation = tuple(np.subtract(box.translation, pose["translation"]))
        return eval_boxes

    evaluate.add_center_dist = add_center_dist
    config = DetectionConfig(
        # The constructor insists on every nuScenes class; only car is scored below.
        # 105 m covers the 100 m labelling range (the nuScenes default for car is 50).
        class_range={name: (105 if name == "car" else 50) for name in DETECTION_NAMES},
        dist_fcn="center_distance",
        dist_ths=[0.5, 1.0, 2.0, 4.0],
        dist_th_tp=2.0,
        min_recall=0.1,
        min_precision=0.1,
        max_boxes_per_sample=500,
        mean_ap_weight=5,
    )
    config.class_names = ["car"]
    for split in version_splits:
        scene_names = set(splits[split])
        results = defaultdict(list)
        for sample in nusc.sample:
            if nusc.get("scene", sample["scene_token"])["name"] not in scene_names:
                continue
            results[sample["token"]]  # every sample appears, empty or not
            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                results[sample["token"]].append(
                    {
                        "sample_token": sample["token"],
                        "translation": ann["translation"],
                        "size": ann["size"],
                        "rotation": ann["rotation"],
                        "velocity": list(nusc.box_velocity(ann_token)[:2]),
                        "detection_name": "car",
                        "detection_score": 1.0,
                        "attribute_name": nusc.get("attribute", ann["attribute_tokens"][0])["name"]
                        if ann["attribute_tokens"] else "",
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            meta = {"use_camera": False, "use_lidar": False, "use_radar": True, "use_map": False, "use_external": False}
            path.write_text(json.dumps({"meta": meta, "results": results}))
            metrics, _ = evaluate.DetectionEval(nusc, config, str(path), split, tmp, verbose=False).evaluate()
        ap = metrics.mean_dist_aps["car"]
        report[f"eval_{split}_gt_as_prediction_car_ap"] = ap
        checks.expect(abs(ap - 1.0) < 1e-6, f"devkit eval on {split}: GT-as-prediction car AP {ap} != 1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataroot", type=Path, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.dataroot or Path(config["dataset"]["output_root"])
    length, width, height = map(float, config["dataset"]["car_size_lwh"])
    n_features = len(config["dataset"]["feature_names"])
    names = config["dataset"]["feature_names"]
    col = {n: i for i, n in enumerate(names)}
    gate = config["dataset"]["lateral_qa_gate"]
    dealias = config["dataset"]["doppler_dealias"]
    dataset_report = json.loads((root / "dataset_report.json").read_text())
    splits = json.loads((root / "splits.json").read_text())
    checks = Checks()
    report: dict = {"dataroot": str(root), "splits": {}, "runs": {}}
    RadarPointCloud.default_filters()

    scene_run: dict[str, str] = {}
    for version, version_splits in (("v1.0-trainval", ["train", "val"]), ("v1.0-test", ["test"])):
        nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
        print(f"[{version}] {len(nusc.scene)} scenes, {len(nusc.sample)} samples, {len(nusc.sample_annotation)} annotations")
        split_of = {name: split for split in version_splits for name in splits[split]}
        checks.expect(set(split_of) == {s["name"] for s in nusc.scene},
                      f"{version}: scenes and splits.json disagree")
        per_split = defaultdict(Counter)
        doppler = defaultdict(Counter)
        residuals = defaultdict(list)
        leading_edges = defaultdict(list)

        # 1. schema -------------------------------------------------------------
        for scene in nusc.scene:
            run = nusc.get("log", scene["log_token"])["logfile"]
            scene_run[scene["name"]] = run
            chain = check_chain(nusc, "sample", scene["first_sample_token"], checks, scene["name"])
            checks.expect(len(chain) == scene["nbr_samples"], f"{scene['name']}: nbr_samples mismatch")
            checks.expect(chain[-1]["token"] == scene["last_sample_token"], f"{scene['name']}: last_sample_token")
            stamps = [s["timestamp"] for s in chain]
            checks.expect(all(np.diff(stamps) > 0), f"{scene['name']}: sample timestamps not increasing")
            for sample in chain:
                checks.expect(sample["scene_token"] == scene["token"], f"sample {sample['token']} scene_token")
                checks.expect(list(sample["data"]) == [CHANNEL], f"sample {sample['token']} channels {list(sample['data'])}")
        for instance in nusc.instance:
            chain = check_chain(nusc, "sample_annotation", instance["first_annotation_token"], checks, instance["token"])
            checks.expect(len(chain) == instance["nbr_annotations"], f"instance {instance['token']} nbr_annotations")
            checks.expect(chain[-1]["token"] == instance["last_annotation_token"], f"instance {instance['token']} last")
            checks.expect(len({nusc.get("sample", a["sample_token"])["scene_token"] for a in chain}) == 1,
                          f"instance {instance['token']} spans scenes")

        # 4b. sweeps -------------------------------------------------------------
        sweep_cfg = config["dataset"]["sweeps"]
        max_lag_us = float(sweep_cfg["max_lag_s"]) * 1e6
        depth_wanted = int(sweep_cfg.get("check_depth", 3))
        scene_of_sample = {s["token"]: s["scene_token"] for s in nusc.sample}
        by_scene = defaultdict(list)
        for sd in nusc.sample_data:
            by_scene[scene_of_sample[sd["sample_token"]]].append(sd)
        sweep_count = 0
        for scene_token, records in by_scene.items():
            records.sort(key=lambda r: r["timestamp"])
            name = nusc.get("scene", scene_token)["name"]
            for i, sd in enumerate(records):
                checks.expect(sd["prev"] == (records[i - 1]["token"] if i else ""),
                              f"{name}: sample_data {sd['token']} prev is not the previous scan")
                checks.expect(sd["next"] == (records[i + 1]["token"] if i + 1 < len(records) else ""),
                              f"{name}: sample_data {sd['token']} next is not the next scan")
                if sd["is_key_frame"]:
                    continue
                sweep_count += 1
                key = nusc.get("sample", sd["sample_token"])
                lag = key["timestamp"] - sd["timestamp"]
                checks.expect(0 < lag <= max_lag_us + 1,
                              f"{name}: sweep {sd['token']} is {lag * 1e-6:.3f} s before its keyframe")
                checks.expect((root / sd["filename"]).is_file(), f"{sd['filename']} missing")
                sidecar = sidecar_path(root, sd["filename"])
                checks.expect(sidecar.is_file(), f"{sidecar.name} missing")
                if sidecar.is_file():
                    shape = np.load(sidecar, mmap_mode="r").shape
                    checks.expect(len(shape) == 2 and shape[1] == n_features, f"{sidecar.name}: shape {shape}")
        stacks = defaultdict(lambda: defaultdict(list))
        for sample in nusc.sample:
            sd = nusc.get("sample_data", sample["data"][CHANNEL])
            depth, prev = 0, sd["prev"]
            while prev and depth < depth_wanted:
                record = nusc.get("sample_data", prev)
                if sd["timestamp"] - record["timestamp"] > max_lag_us:
                    break
                depth, prev = depth + 1, record["prev"]
            split = split_of[nusc.get("scene", sample["scene_token"])["name"]]
            stacks[split]["positive" if sample["anns"] else "background"].append(depth)
        for split, kinds in stacks.items():
            summary = {kind: {"keyframes": len(d), "mean_previous_scans": float(np.mean(d)),
                              "full_stack_fraction": float(np.mean(np.asarray(d) == depth_wanted))}
                       for kind, d in kinds.items()}
            report.setdefault("sweep_stacks", {})[split] = {"depth": depth_wanted, **summary}
            if "positive" in summary and "background" in summary:
                gap = summary["positive"]["full_stack_fraction"] - summary["background"]["full_stack_fraction"]
                checks.expect(gap <= float(sweep_cfg.get("max_full_stack_gap", 0.05)),
                              f"{split}: {depth_wanted} previous scans for "
                              f"{100 * summary['positive']['full_stack_fraction']:.1f}% of positives but "
                              f"{100 * summary['background']['full_stack_fraction']:.1f}% of background "
                              "(the stack depth gives the label away)")
        report.setdefault("sweeps_written", {})[version] = sweep_count

        # 2-3. radar files and labels --------------------------------------------
        for n, sample in enumerate(nusc.sample):
            scene = nusc.get("scene", sample["scene_token"])
            split = split_of[scene["name"]]
            sd, boxes = ego_boxes(nusc, sample)
            path = root / sd["filename"]
            pc = RadarPointCloud.from_file(str(path))
            features = np.load(sidecar_path(root, sd["filename"]))
            checks.expect(features.ndim == 2 and features.shape[1] == n_features, f"{path.name}: sidecar shape {features.shape}")
            checks.expect(pc.points.shape == (18, len(features)),
                          f"{path.name}: devkit kept {pc.points.shape[1]} of {len(features)} points (default filters)")
            if pc.points.shape[1] == len(features):
                cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                rotation = Quaternion(cs["rotation"]).rotation_matrix
                xyz = pc.points[:3].T @ rotation.T + np.array(cs["translation"])
                err = float(np.abs(xyz - features[:, :3]).max()) if len(features) else 0.0
                checks.expect(err < 2e-3, f"{path.name}: pcd->ego xyz differs from sidecar by {err:.4f} m")
                v_comp = np.vstack([pc.points[8:10], np.zeros(len(features))]).T @ rotation.T
                verr = float(np.abs(v_comp[:, :2] - features[:, [col["v_abs_x"], col["v_abs_y"]]]).max()) if len(features) else 0.0
                checks.expect(verr < 2e-3, f"{path.name}: compensated velocity differs from sidecar by {verr:.4f} m/s")
            per_split[split]["samples"] += 1
            # Folded Doppler: stationary returns reading v_abs = -period (see doppler_dealias).
            run_name = scene_run[scene["name"]]
            v_abs_all = features[:, col["v_abs"]]
            doppler[run_name]["points"] += len(features)
            doppler[run_name]["folded"] += int((np.abs(v_abs_all + dealias["period_mps"]) < dealias["static_tolerance_mps"]).sum())
            doppler[run_name]["beyond_period"] += int((np.abs(v_abs_all) > dealias["period_mps"] - 5.0).sum())
            per_split[split]["points"] += len(features)
            per_split[split]["background" if not boxes else "positive"] += 1
            for ann, box in boxes:
                yaw = float(box.orientation.yaw_pitch_roll[0])
                upright = abs(box.orientation.yaw_pitch_roll[1]) < 1e-4 and abs(box.orientation.yaw_pitch_roll[2]) < 1e-4
                checks.expect(upright, f"{ann['token']}: box not upright in ego frame")
                checks.expect(np.allclose(box.wlh, [width, length, height], atol=1e-4), f"{ann['token']}: size {box.wlh}")
                inside = points_in_bev_box(features[:, :2].astype(np.float64), box.center, yaw, length, width)
                checks.expect(inside == ann["num_radar_pts"],
                              f"{ann['token']}: {inside} returns in the devkit box, annotation says {ann['num_radar_pts']}")
                checks.expect(inside >= int(config["dataset"]["min_points_in_box"]), f"{ann['token']}: only {inside} returns")
                in_box_moving = int(((np.abs(features[:, col["v_abs"]]) >= config["dataset"]["moving_speed_mps"])
                                     & (points_in_bev_mask(features[:, :2].astype(np.float64), box.center, yaw, length, width))).sum())
                doppler[run_name]["boxes"] += 1
                doppler[run_name]["boxes_without_moving_returns"] += int(in_box_moving == 0)
                checks.expect(in_box_moving >= int(config["dataset"]["min_moving_points_in_box"]),
                              f"{ann['token']}: {in_box_moving} moving returns in the box; its returns are clutter")
                per_split[split]["annotations"] += 1
                # 5. label QA: opponent's moving returns vs the box, across its width
                d = features[:, :2].astype(np.float64) - box.center[:2]
                c, s = math.cos(-yaw), math.sin(-yaw)
                along, across = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]
                moving = ((np.abs(features[:, col["v_abs"]]) >= gate["moving_speed_mps"])
                          & (np.abs(along) <= length / 2 + gate["search_along_m"])
                          & (np.abs(across) <= width / 2 + gate["search_across_m"]))
                heading = abs(math.degrees(math.atan2(math.sin(yaw), math.cos(yaw))))
                if heading < 25.0 or heading > 155.0:  # tail-on or head-on only
                    corridor = (np.abs(across) <= width / 2 + 0.25) & (np.abs(along) <= NEAR_EDGE_WINDOW_M)
                    if corridor.sum() >= 2:
                        leading_edges[scene_run[scene["name"]]].append(float(along[corridor].max()))
                if moving.sum() >= gate["min_moving_returns"]:
                    residuals[scene_run[scene["name"]]].append(
                        (float(np.median(along[moving])), float(np.median(across[moving])), float(np.hypot(*box.center[:2]))))
            if (n + 1) % 2000 == 0:
                print(f"  checked {n + 1}/{len(nusc.sample)} samples", flush=True)

        # 4. splits --------------------------------------------------------------
        for split in version_splits:
            c = per_split[split]
            runs = sorted({scene_run[name] for name in splits[split]})
            report["splits"][split] = {
                "version": version, "scenes": len(splits[split]), "runs": runs, "samples": c["samples"],
                "positives": c["positive"], "background": c["background"], "annotations": c["annotations"],
                "background_fraction": c["background"] / max(c["samples"], 1), "points": c["points"],
            }
            checks.expect(c["samples"] > 0, f"{split}: empty")
        for run, rows in residuals.items():
            a = np.asarray(rows)
            report["runs"][run] = {
                "positives_with_moving_returns": len(a),
                "along_median_m": float(np.median(a[:, 0])),
                "across_median_m": float(np.median(a[:, 1])),
                "across_abs_p50_m": float(np.median(np.abs(a[:, 1]))),
                "across_abs_p90_m": float(np.percentile(np.abs(a[:, 1]), 90)),
                "frac_across_over_1m": float(np.mean(np.abs(a[:, 1]) > 1.0)),
            }
        for run, c in doppler.items():
            report.setdefault("doppler", {})[run] = {
                "points": c["points"], "folded_points": c["folded"],
                "points_beyond_period_minus_5": c["beyond_period"],
                "boxes": c["boxes"], "boxes_without_moving_returns": c["boxes_without_moving_returns"],
            }
            checks.expect(c["folded"] <= 1e-4 * c["points"],
                          f"{run}: {c['folded']} returns at v_abs = -{dealias['period_mps']} m/s (folded Doppler); "
                          "check doppler_dealias")
        for run, values in leading_edges.items():
            report.setdefault("near_edge_values", {})[run] = values
        # 7. eval ------------------------------------------------------------------
        if not args.skip_eval:
            run_eval(nusc, version_splits, splits, checks, report)

    runs_by_split = {split: set(v["runs"]) for split, v in report["splits"].items()}
    for a in runs_by_split:
        for b in runs_by_split:
            if a < b:
                checks.expect(not (runs_by_split[a] & runs_by_split[b]), f"splits {a} and {b} share runs")
    report["splits_are_run_blocked"] = not any(
        runs_by_split[a] & runs_by_split[b] for a in runs_by_split for b in runs_by_split if a < b)
    report["lateral_qa"] = {run: r["lateral_qa"] for run, r in dataset_report["runs"].items()}
    report["along_qa"] = {run: r.get("along_qa") for run, r in dataset_report["runs"].items()}

    # 5b. near-edge and URDF gates ------------------------------------------------
    tolerance = float(config["dataset"].get("near_edge_tolerance_m", 1.0))
    near_edge = {"expected_leading_edge_m": length / 2, "tolerance_m": tolerance, "by_run": {}}
    for run, values in sorted(report.pop("near_edge_values", {}).items()):
        median = float(np.median(values))
        shortfall = median - length / 2
        near_edge["by_run"][run] = {"frames": len(values), "median_leading_edge_m": median,
                                    "shortfall_m": shortfall, "passes": abs(shortfall) <= tolerance}
        checks.expect(abs(shortfall) <= tolerance,
                      f"{run}: opponent's illuminated face at {median:+.2f} m in box coordinates, a correct "
                      f"box puts it at {length / 2:+.2f} m; measure box_center_forward_offset_m")
    report["near_edge_gate"] = near_edge
    rig = av24_geometry.Av24Rig.load(config["dataset"]["av24_urdf"])
    low, high = av24_geometry.box_center_bounds_in_ram(rig, length)
    origins = {
        "center_of_gravity": float(rig.translation(av24_geometry.RAM, av24_geometry.COG)[0]),
        "novatel_midpoint": float(av24_geometry.novatel_midpoint_in(rig)[0]),
        # laguna_2026_07_25: IU's own ESKF reports its top GNSS antenna.
        "top_antenna": float(rig.translation(av24_geometry.RAM, "top_antenna")[0]),
    }
    urdf_gate = {"box_center_window_m": [low, high], "pose_origin_x_in_rear_axle_middle": origins, "by_run": {}}
    for run_entry in config["runs"]:
        run = run_entry["name"]
        origin = run_entry.get("opponent_pose_origin", "center_of_gravity")
        # pp puts the centre (OPP_BOX_CENTER_IN_RAM - COG_IN_RAM) ahead of the reported
        # pose; the run's offset moves it further along the heading. OPP_BOX_CENTER_IN_RAM
        # is the pinned value the builder checks against the shared checkout (it was
        # length / 2 until 2026-09-27).
        box_origin = float(config["openpcdet"]["expected_box_center_x_in_ram_m"])
        ahead_of_pose = box_origin - origins["center_of_gravity"] + float(run_entry.get("box_center_forward_offset_m", 0.0))
        implied = origins[origin] + ahead_of_pose
        inside = low <= implied <= high
        urdf_gate["by_run"][run] = {"opponent_pose_origin": origin, "box_center_ahead_of_reported_pose_m": ahead_of_pose,
                                    "implied_box_center_x_m": implied, "inside_urdf_window": inside}
        if not inside:
            checks.warnings.append(
                f"{run}: box_center_forward_offset_m implies a box centre {implied:+.3f} m ahead of rear_axle_middle "
                f"(pose origin {origin}), outside the [{low:+.3f}, {high:+.3f}] window av24.urdf admits; "
                "the offset is carrying more than body geometry")
    report["urdf_geometry_gate"] = urdf_gate

    # 6. manifest -----------------------------------------------------------------
    manifest = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    min_speed = float(config["dataset"]["min_ego_speed_mps"])
    slow = [m for m in manifest if m["selected"] and not m.get("ego_speed_mps", -1.0) >= min_speed]
    checks.expect(not slow, f"{len(slow)} selected scans below min_ego_speed_mps={min_speed}")
    front = [m for m in manifest if m.get("target_center_xyz") is not None and m["target_center_xyz"][0] >= 0]
    contaminated = sum(m.get("points_in_box", 0) > 0 for m in front)
    report["front_sector_scans"] = len(front)
    report["front_contamination_rate"] = contaminated / len(front) if front else 0.0
    if report["front_contamination_rate"] > 0.01:
        checks.warnings.append(f"front-sector rear-return contamination {100 * report['front_contamination_rate']:.2f}%")
    # Background must have the whole opponent out of reach, not just its centre.
    reach = float(config["dataset"]["coverage"]["max_range_m"]) + length
    near_background = [m for m in manifest if m["selected"] and m["label_state"].startswith("far_negative")
                       and not (m.get("target_range_m", 0.0) > reach and m.get("points_in_box", 1) == 0)]
    checks.expect(not near_background,
                  f"{len(near_background)} background scans with the opponent within {reach:.1f} m or returns in its box")
    report["manifest_label_states"] = dict(Counter(m["label_state"] for m in manifest))
    report["errors"] = checks.errors
    report["warnings"] = checks.warnings
    report["passed"] = not checks.errors
    (root / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("splits", "runs", "near_edge_gate", "urdf_geometry_gate",
                                             "front_contamination_rate", "splits_are_run_blocked",
                                             "sweeps_written", "sweep_stacks")}, indent=2))
    for k in [k for k in report if k.startswith("eval_")]:
        print(f"{k}: {report[k]}")
    print(f"\n{'PASSED' if report['passed'] else 'FAILED'}: {len(checks.errors)} errors")
    for message in checks.errors[:20]:
        print("  " + message)
    for message in checks.warnings:
        print("  warning: " + message)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
