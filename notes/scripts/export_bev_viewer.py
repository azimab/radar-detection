#!/usr/bin/env python3
"""Pack frames, labels and predictions for the BEV frame viewer (figures/bev_viewer).

For each test set this writes
  <key>.json  per-frame metadata: scene, time, label boxes, prediction boxes + scores,
              ego velocity "e" and each label's velocity (vx, vy appended to its row), both
              ground-plane, in m/s, on the ego frame's axes, by central difference of the
              ego poses and of the annotation track (null where a neighbour is > 0.5 s away)
  <key>.pts.txt  base64 of the current radar scan of every frame, int16 little-endian pairs
              (base64 text because artifact pages do not serve binary files):
              x in decimetres, and 2 * y in decimetres + 1 if the return is moving
              (|v_abs| >= 2 m/s), in the ego (rear_axle_middle) frame. Frame i's points
              are pairs [offset, offset + count) from the metadata.
and a datasets.json index. Only returns inside the model's range are kept.

    PYTHONPATH=vendor/RadarPillar/.deps/native:vendor/RadarPillar \
        python3 tools_eval/export_bev_viewer.py --out figures/bev_viewer
"""

from __future__ import annotations

import os
import argparse
import base64
import json
import pickle
from pathlib import Path

import numpy as np

from pcdet.datasets.radar.radar_dataset import (CHANNEL, SPLIT_VERSION, load_nuscenes_infos,
                                                quaternion_matrix)

OUT = os.path.join(os.environ.get("RADARPILLAR_ROOT", "vendor/RadarPillar"), "output/rear_models")
OLD = os.path.join(os.environ.get("RADAR_DATA_ROOT", "."), "data_nuscenes")
NEW = os.path.join(os.environ.get("RADAR_DATA_ROOT", "."), "data_nuscenes_2026_09_28")
V3S2 = f"{OUT}/rear_radarpillar_car_v3/rear_radarpillar_v3_seed2/eval/eval_all_default"
HOLD = f"{OUT}/rear_radarpillar_car_v3_lagunaholdout/rear_radarpillar_v3_laguna_seed1/eval/eval_all_default"
DATASETS = [
    {
        "key": "laguna_0725", "name": "Laguna 07-25", "detail": "IU opponent · never seen in training",
        "model": "v3 seed 2, epoch 35", "root": NEW, "split": "laguna_0725",
        "result": f"{V3S2}/laguna_0725_sweep/epoch_35/laguna_0725/result.pkl",
    },
    {
        "key": "laguna_holdout", "name": "Laguna 07-26 holdout", "detail": "Purdue opponent · same session as training",
        "model": "holdout model (44 scenes), epoch 35", "root": OLD, "split": "laguna_holdout",
        "result": f"{HOLD}/val_sweep/epoch_35/laguna_holdout/result.pkl",
    },
    {
        "key": "caltech_val", "name": "Caltech val", "detail": "Putnam Park · different track",
        "model": "v3 seed 2, epoch 35", "root": OLD, "split": "val",
        "result": f"{V3S2}/val_sweep/epoch_35/val/result.pkl",
        # Labels ~1.3 m along-track off the car for the whole scene; excluded from val_qa.
        "flagged_scenes": {"caltech_2026_05_05_rear-013": "Labels sit ~1.3 m ahead of the car in this scene (excluded from val_qa)"},
    },
]
ROI_X = (-105.0, 2.0)
ROI_Y = (-52.0, 52.0)
MAX_GAP_S = 0.5


def central_difference(cur, prev, nxt, pos, ts):
    """Velocity of ``cur`` from its neighbours, one-sided at a track end or gap, else None."""
    near = lambda e: e is not None and abs(ts(e) - ts(cur)) * 1e-6 <= MAX_GAP_S
    a = prev if near(prev) else cur
    b = nxt if near(nxt) else cur
    if a is b:
        return None
    return (np.asarray(pos(b), dtype=np.float64) - pos(a)) / ((ts(b) - ts(a)) * 1e-6)


class Velocities:
    """Ego and label-track velocities, rotated into each keyframe's ego axes."""

    def __init__(self, root: str, split: str):
        version = Path(root) / SPLIT_VERSION[split]
        table = lambda name: json.loads((version / f"{name}.json").read_text())
        self.pose = {p["token"]: p for p in table("ego_pose")}
        self.scans = {sd["token"]: sd for sd in table("sample_data") if f"samples/{CHANNEL}/" in sd["filename"]}
        self.keyframe = {sd["sample_token"]: sd for sd in self.scans.values() if sd["is_key_frame"]}
        self.sample_ts = {s["token"]: s["timestamp"] for s in table("sample")}
        self.anns = {a["token"]: a for a in table("sample_annotation")}
        self.by_sample = {}
        for a in self.anns.values():
            self.by_sample.setdefault(a["sample_token"], []).append(a)

    def __call__(self, info):
        """(ego [vx, vy] or None, one [vx, vy] or None per label box of ``info``)."""
        sd = self.keyframe[info["sample_token"]]
        pose = self.pose[sd["ego_pose_token"]]
        rot, t = quaternion_matrix(pose["rotation"]), np.asarray(pose["translation"])
        ego = central_difference(sd, self.scans.get(sd["prev"]), self.scans.get(sd["next"]),
                                 lambda s: self.pose[s["ego_pose_token"]]["translation"], lambda s: s["timestamp"])
        rows = []
        cands = self.by_sample.get(info["sample_token"], [])
        for g in info["annos"]["gt_boxes_lidar"]:
            # the annotation this label box came from: the one whose centre it is
            ann = min(cands, key=lambda a: np.linalg.norm(rot.T @ (np.asarray(a["translation"]) - t) - g[:3]))
            v = central_difference(ann, self.anns.get(ann["prev"]), self.anns.get(ann["next"]),
                                   lambda a: a["translation"], lambda a: self.sample_ts[a["sample_token"]])
            rows.append(v)
        flat = lambda v: None if v is None else [round(float(c), 2) for c in (rot.T @ v)[:2]]
        return flat(ego), [flat(v) for v in rows]


def boxes(arr, extra=None):
    out = []
    for i, b in enumerate(np.asarray(arr).reshape(-1, 7)):
        row = [round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[3]), 2), round(float(b[4]), 2),
               round(float(b[6]), 3)]
        if extra is not None:
            row.append(round(float(extra[i]), 3))
        out.append(row)
    return out


def export(ds: dict, out: Path) -> dict:
    infos = load_nuscenes_infos(ds["root"], ds["split"], ["Car"])
    preds = {str(a["frame_id"]): a for a in pickle.load(open(ds["result"], "rb"))}
    velocities = Velocities(ds["root"], ds["split"])
    scenes, frames, chunks, offset = [], [], [], 0
    scene_index, scene_t0 = {}, {}
    for info in infos:
        scene = info["scene"]
        if scene not in scene_index:
            scene_index[scene] = len(scenes)
            scene_t0[scene] = info["timestamp"]
            scenes.append({"name": scene, "note": ds.get("flagged_scenes", {}).get(scene)})
        pts = np.load(Path(ds["root"]) / info["point_cloud"]["path"])
        keep = (pts[:, 0] >= ROI_X[0]) & (pts[:, 0] <= ROI_X[1]) & (pts[:, 1] >= ROI_Y[0]) & (pts[:, 1] <= ROI_Y[1])
        pts = pts[keep]
        x = np.round(pts[:, 0] * 10).astype(np.int16)
        y = (np.round(pts[:, 1] * 10).astype(np.int32) * 2 + (np.abs(pts[:, 4]) >= 2.0)).astype(np.int16)
        chunks.append(np.stack([x, y], axis=1).ravel())
        a = preds.get(info["frame_id"])
        gt = info["annos"]["gt_boxes_lidar"]
        ego_v, gt_v = velocities(info)
        frames.append({
            "s": scene_index[scene],
            "t": round((info["timestamp"] - scene_t0[scene]) * 1e-6, 2),
            "o": offset, "n": int(len(pts)),
            "e": ego_v,
            "g": [row + [int(n)] + (v or [None, None])
                  for row, n, v in zip(boxes(gt), info["annos"]["num_points_in_gt"], gt_v)],
            "p": boxes(a["boxes_lidar"], a["score"]) if a is not None else [],
        })
        offset += len(pts)
    raw = np.concatenate(chunks).astype("<i2").tobytes()
    (out / f"{ds['key']}.pts.txt").write_bytes(base64.b64encode(raw))
    meta = {k: ds[k] for k in ("key", "name", "detail", "model")}
    meta.update({"scenes": scenes, "frames": frames, "points": offset})
    (out / f"{ds['key']}.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"{ds['key']}: {len(frames)} frames, {len(scenes)} scenes, {offset:,} points, "
          f"{(out / (ds['key'] + '.pts.txt')).stat().st_size / 1e6:.1f} MB points, "
          f"{(out / (ds['key'] + '.json')).stat().st_size / 1e6:.1f} MB json")
    return {k: ds[k] for k in ("key", "name", "detail", "model")} | {"frames": len(frames)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/bev_viewer")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = [export(ds, out) for ds in DATASETS]
    (out / "datasets.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
