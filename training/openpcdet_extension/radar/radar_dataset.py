"""Native OpenPCDet dataset for the rear-radar nuScenes build.

Reads the nuScenes v1.0 tables that ``radar_openpcdet/build_nuscenes_dataset.py``
writes, directly, with numpy -- the training environment needs no nuscenes-devkit.
Boxes are taken from global annotations into the ego ``rear_axle_middle`` frame (the
nuScenes ego frame) as ``[x, y, z, dx, dy, dz, heading]``, the frame the model trains
in, so nothing here converts to or from a camera frame. There is no calibration, no
image, no truncation/occlusion, and no difficulty tier, because none of those concepts
exist for a radar-only dataset.

With ``NUM_SWEEPS > 1`` each sample also stacks up to ``NUM_SWEEPS - 1`` earlier scans
of the same scene (causal only: this runs online), found by walking ``sample_data``
``prev`` links, and appends a per-point ``time_lag`` column (seconds, 0 for the current
scan). Previous scans are kept in their own ego frame by default: the opponent follows
at nearly our speed, so it stays put in the ego frame while static clutter smears and is
separated by Doppler. ``SWEEP_EGO_COMPENSATION`` moves them into the current ego frame
instead, for the ablation.

Points come from the ego-frame feature sidecar next to each radar ``.pcd``, not the
``.pcd`` itself: the model uses all 11 FRGen21 features, and nuScenes' 18-field radar
format has no slot for ``snr``, ``confidence`` or ``power``.

Evaluation is likewise native: matching is done with the compiled ``iou3d_nms``
op the model already uses for NMS, and reported as a single-class AP plus the
localisation errors that actually matter for tracking an opponent.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

from ..dataset import DatasetTemplate
from .radar_augment import augment_radar_sample

CHANNEL = "RADAR_REAR"
FEATURE_DIR = "RADAR_REAR_FRGEN21"
# laguna_train / laguna_holdout: within-Laguna diagnostic split (16 held-out scenes in four
# 4-scene blocks, one guard scene dropped either side), to separate track shift from capacity.
SPLIT_VERSION = {"train": "v1.0-trainval", "val": "v1.0-trainval", "test": "v1.0-test",
                 "laguna_train": "v1.0-trainval", "laguna_holdout": "v1.0-trainval",
                 "val_qa": "v1.0-trainval", "laguna_0725": "v1.0-trainval"}
# nuScenes category -> class name the model predicts.
CATEGORY_TO_CLASS = {"vehicle.car": "Car"}


def quaternion_matrix(wxyz):
    w, x, y, z = wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rigid_transform(pose):
    """4x4 ego->global transform of a nuScenes ``ego_pose`` record."""
    transform = np.eye(4)
    transform[:3, :3] = quaternion_matrix(pose["rotation"])
    transform[:3, 3] = pose["translation"]
    return transform


def load_nuscenes_infos(root, split, class_names, num_sweeps=1, max_sweep_lag_s=0.5):
    """One info per keyframe sample of ``split``, in (scene, time) order.

    With ``num_sweeps > 1`` each info also lists, under ``sweeps``, up to
    ``num_sweeps - 1`` earlier scans of the same scene no older than ``max_sweep_lag_s``,
    newest first, each with its ``time_lag`` and the 4x4 transform from its ego frame
    into the current one.
    """
    root = Path(root)
    splits = json.loads((root / "splits.json").read_text())
    if split not in splits:
        raise KeyError(f"{split!r} not in {root / 'splits.json'} ({sorted(splits)})")
    version = root / SPLIT_VERSION[split]
    table = lambda name: json.loads((version / f"{name}.json").read_text())
    scenes = {s["token"]: s for s in table("scene")}
    wanted = {token for token, s in scenes.items() if s["name"] in set(splits[split])}
    if len(wanted) != len(splits[split]):
        raise ValueError(f"{split}: splits.json names scenes missing from {version}")
    channel_data = [sd for sd in table("sample_data") if f"samples/{CHANNEL}/" in sd["filename"]]
    sample_data = {sd["sample_token"]: sd for sd in channel_data if sd["is_key_frame"]}
    # Every scan of the channel, keyframe or sweep, for walking ``prev`` links.
    scan_by_token = {sd["token"]: sd for sd in channel_data}
    sample_scene = {s["token"]: s["scene_token"] for s in table("sample")}
    ego_pose = {p["token"]: p for p in table("ego_pose")}
    instances = {i["token"]: i for i in table("instance")}
    categories = {c["token"]: c["name"] for c in table("category")}
    annotations = {}
    for ann in table("sample_annotation"):
        annotations.setdefault(ann["sample_token"], []).append(ann)

    infos = []
    samples = sorted((s for s in table("sample") if s["scene_token"] in wanted),
                     key=lambda s: (scenes[s["scene_token"]]["name"], s["timestamp"]))
    for sample in samples:
        sd = sample_data[sample["token"]]
        pose = ego_pose[sd["ego_pose_token"]]
        r_ego = quaternion_matrix(pose["rotation"])
        t_ego = np.asarray(pose["translation"], dtype=np.float64)
        boxes, names, npts = [], [], []
        for ann in annotations.get(sample["token"], []):
            name = CATEGORY_TO_CLASS.get(categories[instances[ann["instance_token"]]["category_token"]])
            if name is None or name not in class_names:
                continue
            center = r_ego.T @ (np.asarray(ann["translation"], dtype=np.float64) - t_ego)
            r_box = r_ego.T @ quaternion_matrix(ann["rotation"])
            width, length, height = ann["size"]  # nuScenes wlh
            boxes.append([*center, length, width, height, np.arctan2(r_box[1, 0], r_box[0, 0])])
            names.append(name)
            npts.append(ann["num_radar_pts"])
        stem = Path(sd["filename"]).stem
        sweeps = []
        prev = scan_by_token.get(sd["prev"])
        while prev is not None and len(sweeps) < num_sweeps - 1:
            time_lag = (sd["timestamp"] - prev["timestamp"]) * 1e-6
            # Never cross into another scene, and stop at a recording gap rather than
            # stacking a stale scan (dropped frames make the chain sparser, not older).
            if sample_scene[prev["sample_token"]] != sample["scene_token"] or time_lag > max_sweep_lag_s:
                break
            prev_pose = ego_pose[prev["ego_pose_token"]]
            sweeps.append(
                {
                    "path": f"samples/{FEATURE_DIR}/{Path(prev['filename']).stem}.npy",
                    "time_lag": time_lag,
                    "to_current_ego": np.linalg.solve(rigid_transform(pose), rigid_transform(prev_pose)),
                }
            )
            prev = scan_by_token.get(prev["prev"])
        infos.append(
            {
                "frame_id": stem,
                "sample_token": sample["token"],
                "scene": scenes[sample["scene_token"]]["name"],
                "timestamp": sample["timestamp"],
                "point_cloud": {"path": f"samples/{FEATURE_DIR}/{stem}.npy", "radar_pcd": sd["filename"]},
                "sweeps": sweeps,
                "annos": {
                    "name": np.array(names, dtype="<U3"),
                    "gt_boxes_lidar": np.array(boxes, dtype=np.float32).reshape(-1, 7),
                    "num_points_in_gt": np.array(npts, dtype=np.int32),
                },
            }
        )
    return infos


class RadarDataset(DatasetTemplate):
    """Rear-radar samples from a nuScenes v1.0 dataroot.

    Layout under ``DATA_PATH``::

        splits.json                                   scene names per split
        v1.0-trainval/*.json, v1.0-test/*.json        nuScenes tables
        samples/RADAR_REAR_FRGEN21/<stem>.npy         (N, NUM_POINT_FEATURES) float32, ego frame
        dataset_report.json                           feature_names, checked against the config

    Sweep stacking (all optional; the defaults reproduce single-scan input exactly)::

        NUM_SWEEPS               scans per sample, current included (default 1)
        MAX_SWEEP_LAG_S          oldest previous scan to stack, seconds (default 0.5)
        SWEEP_EGO_COMPENSATION   move previous scans into the current ego frame (default False)
    """

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        # DATA_PATH in the cfgs uses ${RADAR_DATA_ROOT}; train.py/test.py pass root_path=None
        root_path = Path(os.path.expandvars(str(root_path if root_path is not None else dataset_cfg.DATA_PATH)))
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger,
        )
        self.num_point_features = int(self.dataset_cfg.NUM_POINT_FEATURES)
        self.num_sweeps = int(self.dataset_cfg.get("NUM_SWEEPS", 1))
        self.max_sweep_lag_s = float(self.dataset_cfg.get("MAX_SWEEP_LAG_S", 0.5))
        self.sweep_ego_compensation = bool(self.dataset_cfg.get("SWEEP_EGO_COMPENSATION", False))
        if self.num_sweeps < 1:
            raise ValueError(f"NUM_SWEEPS must be >= 1, got {self.num_sweeps}")
        report = json.loads((self.root_path / "dataset_report.json").read_text())["dataset"]
        src_features = list(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        # ``time_lag`` is computed here when stacking, never stored: it must be the last
        # src column exactly when NUM_SWEEPS > 1, and the columns before it must match
        # the stored ones exactly.
        stacking = self.num_sweeps > 1
        if stacking != ("time_lag" in src_features) or (stacking and src_features[-1] != "time_lag"):
            raise ValueError(
                f"NUM_SWEEPS={self.num_sweeps} needs 'time_lag' as the last src_feature_list "
                f"entry if and only if NUM_SWEEPS > 1, got {src_features}"
            )
        expected = src_features[:-1] if stacking else src_features
        if len(src_features) != self.num_point_features:
            raise ValueError(
                f"NUM_POINT_FEATURES={self.num_point_features} != {len(src_features)} src features"
            )
        self.stored_features = expected
        if report["feature_names"] != expected:
            raise ValueError(
                f"dataset features {report['feature_names']} != config src_feature_list {expected}; "
                "a column-order mismatch would silently feed the model the wrong features"
            )
        self.infos = []
        self.set_split(self.dataset_cfg.DATA_SPLIT[self.mode])

    # -- splits -----------------------------------------------------------------

    def set_split(self, split):
        self.split = split
        self.infos = load_nuscenes_infos(
            self.root_path, split, self.class_names,
            num_sweeps=self.num_sweeps, max_sweep_lag_s=self.max_sweep_lag_s,
        )
        self.sample_id_list = [info["frame_id"] for info in self.infos]
        if not self.infos:
            raise ValueError(f"split {split!r} has no samples under {self.root_path}")
        if self.logger is not None:
            positives = sum(len(i["annos"]["name"]) > 0 for i in self.infos)
            self.logger.info(
                f"RadarDataset split {split}: {len(self.infos)} samples ({positives} with a car)"
            )
            if self.num_sweeps > 1:
                stacked = np.bincount([len(i["sweeps"]) for i in self.infos], minlength=self.num_sweeps)
                self.logger.info(
                    f"RadarDataset sweeps: up to {self.num_sweeps - 1} previous within "
                    f"{self.max_sweep_lag_s} s, ego compensation {self.sweep_ego_compensation}; "
                    f"samples by previous-scan count {stacked.tolist()}"
                )

    # -- io ---------------------------------------------------------------------

    def load_scan(self, relative_path):
        path = self.root_path / relative_path
        points = np.load(path)
        if points.ndim != 2 or points.shape[1] != len(self.stored_features):
            raise ValueError(
                f"{path}: shape {points.shape} does not match "
                f"{len(self.stored_features)} stored features {self.stored_features}"
            )
        return points.astype(np.float32, copy=False)

    def get_points(self, info):
        points = self.load_scan(info["point_cloud"]["path"])
        if self.num_sweeps == 1:
            return points
        # Current scan first, then previous scans newest first, each tagged with its lag.
        stacked = [np.hstack([points, np.zeros((len(points), 1), dtype=np.float32)])]
        for sweep in info["sweeps"]:
            prev = self.load_scan(sweep["path"])
            if self.sweep_ego_compensation:
                prev = self.to_current_ego(prev, sweep["to_current_ego"])
            lag = np.full((len(prev), 1), sweep["time_lag"], dtype=np.float32)
            stacked.append(np.hstack([prev, lag]))
        return np.concatenate(stacked, axis=0)

    def to_current_ego(self, points, transform):
        """Move a previous scan into the current ego frame: positions and the
        ego-frame absolute velocity vector. ``v_rel``/``v_abs`` are scalars measured
        from where the sensor was, so they are left as recorded."""
        points = points.copy()
        rotation = transform[:3, :3]
        points[:, :3] = points[:, :3] @ rotation.T + transform[:3, 3]
        vx = self.stored_features.index("v_abs_x")
        vy = self.stored_features.index("v_abs_y")
        velocity = np.zeros((len(points), 3))
        velocity[:, :2] = points[:, [vx, vy]]
        points[:, [vx, vy]] = (velocity @ rotation.T)[:, :2]
        return points

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = self.infos[index]
        annos = info["annos"]
        input_dict = {
            "points": self.get_points(info),
            "frame_id": info["frame_id"],
            "gt_boxes": np.asarray(annos["gt_boxes_lidar"], dtype=np.float32).reshape(-1, 7),
            # An empty box list yields a float64 array unless the dtype is forced.
            "gt_names": np.asarray(annos["name"], dtype=str).reshape(-1),
        }
        # Doppler-consistent augmentation (radar_augment.py); the generic DATA_AUGMENTOR
        # stays off. torch.initial_seed() differs per worker and per epoch (workers are not
        # persistent), so seeding from it gives each worker its own stream.
        if self.training and self.dataset_cfg.get("RADAR_AUGMENT", {}).get("ENABLED", False):
            rng = np.random.default_rng([torch.initial_seed() % 2**32, index])
            input_dict["points"], input_dict["gt_boxes"] = augment_radar_sample(
                input_dict["points"], input_dict["gt_boxes"],
                list(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list),
                self.dataset_cfg.RADAR_AUGMENT, rng,
            )
        return self.prepare_data(data_dict=input_dict)

    # -- predictions ------------------------------------------------------------

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        annos = []
        for index, pred in enumerate(pred_dicts):
            boxes = pred["pred_boxes"].cpu().numpy()
            scores = pred["pred_scores"].cpu().numpy()
            labels = pred["pred_labels"].cpu().numpy()
            annos.append(
                {
                    "frame_id": batch_dict["frame_id"][index],
                    "name": np.array([class_names[i - 1] for i in labels]),
                    "score": scores,
                    "boxes_lidar": boxes,
                    "pred_labels": labels,
                }
            )
        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            for anno in annos:
                lines = [
                    "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.6f} {} {:.4f}".format(
                        *box, name, score
                    )
                    for box, name, score in zip(anno["boxes_lidar"], anno["name"], anno["score"])
                ]
                (output_path / f"{anno['frame_id']}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else "")
                )
        return annos

    # -- evaluation -------------------------------------------------------------

    def evaluation(self, det_annos, class_names, **kwargs):
        from .radar_eval import evaluate

        gt = {
            info["frame_id"]: (
                np.asarray(info["annos"]["gt_boxes_lidar"], dtype=np.float32).reshape(-1, 7),
                np.asarray(info["annos"]["num_points_in_gt"]).reshape(-1),
            )
            for info in self.infos
        }
        return evaluate(
            det_annos,
            gt,
            iou_thresholds=tuple(self.dataset_cfg.get("EVAL_IOU_THRESHOLDS", [0.3, 0.5, 0.7])),
            score_threshold=float(self.dataset_cfg.get("EVAL_SCORE_THRESHOLD", 0.3)),
            range_bins=tuple(
                tuple(pair) for pair in self.dataset_cfg.get("EVAL_RANGE_BINS", [[0, 20], [20, 40], [40, 60], [60, 105]])
            ),
        )
