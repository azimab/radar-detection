#!/usr/bin/env python3
"""BEV images of the rear-radar nuScenes dataset: boxes over the radar returns.

    # N positives per run, cropped to +-8 m around the box (the scale at which you can
    # judge whether a box sits on its returns), plus one full-scan view each
    python3 radar_openpcdet/visualize_nuscenes.py --per-run 6

    # the scans the lateral QA gate dropped, re-labelled from the scan cache
    python3 radar_openpcdet/visualize_nuscenes.py --qa-dropped 6

Dataset images read boxes through the nuScenes devkit (global annotation -> ego frame)
and points from the ego-frame feature sidecar, so what is drawn is what a devkit user
and the training loader see. Output: ``<dataroot>/viz/<run>/``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from repo_config import DEFAULT_CONFIG, load_config  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor" / "pylib"))
sys.path.insert(0, str(HERE))

from nuscenes.nuscenes import NuScenes  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402

CHANNEL = "RADAR_REAR"
FEATURE_DIR = "RADAR_REAR_FRGEN21"


def corners(center: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    local = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]]) * [length / 2, width / 2]
    rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return local @ rot.T + center[:2]


def draw(ax, features: np.ndarray, boxes: list, title: str, zoom: float | None, v_col: int) -> None:
    xy, v = features[:, :2], features[:, v_col]
    moving = np.abs(v) >= 2.0
    ax.scatter(xy[~moving, 0], xy[~moving, 1], s=10, c="0.75", label="static returns (|v_abs| < 2 m/s)")
    sc = ax.scatter(xy[moving, 0], xy[moving, 1], s=26, c=v[moving], cmap="coolwarm", vmin=-15, vmax=15,
                    label="moving returns")
    for center, yaw, length, width, style, label in boxes:
        c = corners(center, yaw, length, width)
        ax.plot(c[:, 0], c[:, 1], style, lw=1.6, label=label)
        ax.plot([center[0], center[0] + math.cos(yaw) * length / 2],
                [center[1], center[1] + math.sin(yaw) * length / 2], style, lw=1.0)
    if zoom and boxes:
        cx, cy = boxes[0][0][:2]
        ax.set_xlim(cx - zoom, cx + zoom)
        ax.set_ylim(cy - zoom, cy + zoom)
    else:
        ax.plot(0, 0, "k^", ms=8)
        ax.set_xlim(-105, 5)
        ax.set_ylim(-55, 55)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]  (rear_axle_middle)")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=10)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="v_abs [m/s]")


def dataset_images(root: Path, config: dict, per_run: int, zoom: float) -> list[Path]:
    length, width, _ = map(float, config["dataset"]["car_size_lwh"])
    v_col = config["dataset"]["feature_names"].index("v_abs")
    written = []
    for version in ("v1.0-trainval", "v1.0-test"):
        nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
        by_run: dict[str, list] = {}
        for sample in nusc.sample:
            if sample["anns"]:
                run = nusc.get("log", nusc.get("scene", sample["scene_token"])["log_token"])["logfile"]
                by_run.setdefault(run, []).append(sample)
        for run, samples in sorted(by_run.items()):
            out = root / "viz" / run
            out.mkdir(parents=True, exist_ok=True)
            picks = [samples[int(i)] for i in np.linspace(0, len(samples) - 1, per_run)]
            for k, sample in enumerate(picks):
                sd = nusc.get("sample_data", sample["data"][CHANNEL])
                pose = nusc.get("ego_pose", sd["ego_pose_token"])
                features = np.load(root / sd["filename"].replace(CHANNEL + "/", FEATURE_DIR + "/").replace(".pcd", ".npy"))
                boxes = []
                for ann_token in sample["anns"]:
                    box = nusc.get_box(ann_token)
                    box.translate(-np.array(pose["translation"]))
                    box.rotate(Quaternion(pose["rotation"]).inverse)
                    boxes.append((box.center, box.orientation.yaw_pitch_roll[0], length, width, "k-", "label"))
                rng = float(np.hypot(*boxes[0][0][:2]))
                ann = nusc.get("sample_annotation", sample["anns"][0])
                stem = Path(sd["filename"]).stem
                views = [(zoom, "zoom")] + ([(None, "full")] if k == 0 else [])
                for z, tag in views:
                    fig, ax = plt.subplots(figsize=(8, 7))
                    draw(ax, features, boxes,
                         f"{run}  {tag}\n{stem}\nrange {rng:.1f} m, {ann['num_radar_pts']} returns in box", z, v_col)
                    ax.legend(loc="lower left", fontsize=7)
                    path = out / f"{stem}_{tag}.png"
                    fig.tight_layout()
                    fig.savefig(path, dpi=80)
                    plt.close(fig)
                    written.append(path)
    return written


def qa_dropped_images(root: Path, config: dict, per_run: int, zoom: float) -> list[Path]:
    """Re-label from the scan cache and draw the scans the lateral gate removed."""
    import build_nuscenes_dataset as build

    length, width, _ = map(float, config["dataset"]["car_size_lwh"])
    v_col = config["dataset"]["feature_names"].index("v_abs")
    report = json.loads((root / "dataset_report.json").read_text())
    geometry = build.Geometry(config)
    labeller = build.Labeller(config, geometry)
    written = []
    for run_entry in config["runs"]:
        spans = report["runs"][run_entry["name"]]["lateral_qa"]["spans"]
        if not spans:
            continue
        frames, _, _, _, _ = labeller.run_frames(run_entry)
        dropped = [f for f in frames if f.center is not None
                   and any(s["start_s"] <= f.stamp_ns * 1e-9 <= s["end_s"] for s in spans)]
        out = root / "viz" / run_entry["name"] / "qa_dropped"
        out.mkdir(parents=True, exist_ok=True)
        for frame in [dropped[int(i)] for i in np.linspace(0, len(dropped) - 1, min(per_run, len(dropped)))]:
            fig, ax = plt.subplots(figsize=(8, 7))
            residual = frame.meta.get("lateral_residual_m")
            draw(ax, frame.features, [(frame.center, frame.yaw, length, width, "r--", "GPS box (dropped)")],
                 f"{run_entry['name']}  QA-dropped\nstamp {frame.stamp_ns}  range {frame.meta['target_range_m']:.1f} m\n"
                 f"moving returns sit {residual if residual is None else round(residual, 2)} m across the box",
                 zoom, v_col)
            ax.legend(loc="lower left", fontsize=7)
            path = out / f"{frame.stamp_ns}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=80)
            plt.close(fig)
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataroot", type=Path, default=None)
    parser.add_argument("--per-run", type=int, default=6)
    parser.add_argument("--zoom", type=float, default=8.0, help="half-width of the cropped view, metres")
    parser.add_argument("--qa-dropped", type=int, default=0, help="also draw N QA-dropped scans per run")
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.dataroot or Path(config["dataset"]["output_root"])
    written = dataset_images(root, config, args.per_run, args.zoom) if args.per_run else []
    if args.qa_dropped:
        written += qa_dropped_images(root, config, args.qa_dropped, args.zoom)
    print(f"wrote {len(written)} images under {root / 'viz'}")


if __name__ == "__main__":
    main()
