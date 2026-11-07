#!/usr/bin/env python3
"""Pack an OpenPCDet RadarPillar checkpoint into a laptop bundle, and prove it matches.

Run on the lab machines (it needs the OpenPCDet checkout; the bundle does not):

    PYTHONPATH=training/vendor/RadarPillar/.deps/native:training/vendor/RadarPillar \\
        python3 live/export_model.py --check 200

The bundle (``live/models/<name>.pt``, ~2 MB) holds the weights, every model and
post-processing constant, the feature settings from config/pipeline.yaml and the radar
mounting from the AV-24 URDF. It loads with ``torch.load(weights_only=True)``.

``--check N`` runs N samples of the cfg's eval split through both OpenPCDet and
``rear_radar_model`` and compares voxels, raw head outputs and (on a GPU, where
OpenPCDet's NMS runs) final boxes. On CPU, OpenPCDet's anchors are built by patching
``Tensor.cuda`` to a no-op; NMS is then checked against a CPU re-scoring instead.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RADARPILLAR = Path(os.environ.get("RADARPILLAR_ROOT", REPO_ROOT / "training" / "vendor" / "RadarPillar"))
for path in (RADARPILLAR / ".deps" / "native", RADARPILLAR, REPO_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("RADAR_DATA_ROOT", str(REPO_ROOT))

import av24_geometry  # noqa: E402
import rear_radar_model as rrm  # noqa: E402
from repo_config import load_config  # noqa: E402

DEFAULT_CFG = "rear_radarpillar_car_v5"
# v5 seed 1: best 3-epoch window on val_qa is centred on epoch 15 (74.5 smoothed
# centre-1 m AP; epoch 14 is the single best at 75.7).
DEFAULT_CKPT = (RADARPILLAR / "output/rear_models/rear_radarpillar_car_v5/rear_radarpillar_v5_seed1/ckpt/"
                "checkpoint_epoch_15.pth")
FEATURE_KEYS = ["coverage", "velocity_compensation_sign", "doppler_dealias", "min_ego_speed_mps",
                "ego_odom_topic", "ego_odom_child_frame", "feature_names"]


def load_cfg(name: str):
    from pcdet.config import cfg, cfg_from_yaml_file

    cwd = os.getcwd()
    os.chdir(RADARPILLAR / "tools")
    try:
        cfg_from_yaml_file(f"cfgs/rear_models/{name}.yaml", cfg)
    finally:
        os.chdir(cwd)
    cfg.DATA_CONFIG.DATA_PATH = os.path.expandvars(cfg.DATA_CONFIG.DATA_PATH)
    return cfg


def hyperparameters(cfg) -> dict:
    data, model = cfg.DATA_CONFIG, cfg.MODEL
    vox = next(p for p in data.DATA_PROCESSOR if p.NAME == "transform_points_to_voxels")
    pc_range = [float(v) for v in data.POINT_CLOUD_RANGE]
    voxel_size = [float(v) for v in vox.VOXEL_SIZE]
    grid = [int(round((pc_range[i + 3] - pc_range[i]) / voxel_size[i])) for i in range(3)]
    vfe, head, bb, post = model.VFE, model.DENSE_HEAD, model.BACKBONE_2D, model.POST_PROCESSING
    # The re-implementation covers exactly this configuration; refuse anything else.
    expect = {
        "MODEL.NAME": (model.NAME, "PointPillar"), "VFE.NAME": (vfe.NAME, "PillarVFE"),
        "WITH_DISTANCE": (vfe.WITH_DISTANCE, False), "USE_NORM": (vfe.USE_NORM, True),
        "USE_ABSOLUTE_XYZ": (vfe.get("USE_ABSOLUTE_XYZ", vfe.get("USE_ABSLOTE_XYZ")), True),
        "NUM_FILTERS": (len(vfe.NUM_FILTERS), 1), "HEAD": (head.NAME, "AnchorHeadSingle"),
        "DIR": (head.USE_DIRECTION_CLASSIFIER, True), "SINCOS": (head.TARGET_ASSIGNER_CONFIG.BOX_CODER, "ResidualCoder"),
        "anchor sets": (len(head.ANCHOR_GENERATOR_CONFIG), 1), "classes": (len(cfg.CLASS_NAMES), 1),
        "MULTI_CLASSES_NMS": (post.NMS_CONFIG.MULTI_CLASSES_NMS, False),
        "OUTPUT_RAW_SCORE": (post.OUTPUT_RAW_SCORE, False),
        "SWEEP_EGO_COMPENSATION": (bool(data.get("SWEEP_EGO_COMPENSATION", False)), False),
        "encoding": (list(data.POINT_FEATURE_ENCODING.used_feature_list), list(data.POINT_FEATURE_ENCODING.src_feature_list)),
    }
    for vfe_flag in ("USE_VELOCITY_DECOMPOSITION", "USE_REL_VELOCITY_DECOMPOSITION", "USE_VELOCITY_OFFSET",
                     "USE_REL_VELOCITY_OFFSET", "FEATURE_ORDER"):
        expect[vfe_flag] = (bool(vfe.get(vfe_flag, False)), False)
    bad = {k: v for k, (v, want) in expect.items() if v != want}
    if bad:
        raise SystemExit(f"rear_radar_model does not implement this cfg: {bad}")
    a = head.ANCHOR_GENERATOR_CONFIG[0]
    if len(a.anchor_sizes) != 1 or len(a.anchor_bottom_heights) != 1 or not a.get("align_center", False):
        raise SystemExit("rear_radar_model expects one anchor size, one height and align_center")
    return {
        "class_names": list(cfg.CLASS_NAMES), "num_class": 1,
        "feature_names": list(data.POINT_FEATURE_ENCODING.src_feature_list),
        "num_point_features": len(data.POINT_FEATURE_ENCODING.src_feature_list),
        "point_cloud_range": pc_range, "voxel_size": voxel_size, "grid_size": grid,
        "max_points_per_voxel": int(vox.MAX_POINTS_PER_VOXEL), "max_voxels": int(vox.MAX_NUMBER_OF_VOXELS["test"]),
        "num_sweeps": int(data.get("NUM_SWEEPS", 1)), "max_sweep_lag_s": float(data.get("MAX_SWEEP_LAG_S", 0.5)),
        "vfe_filters": int(vfe.NUM_FILTERS[0]),
        "backbone": {"layer_nums": list(bb.LAYER_NUMS), "layer_strides": list(bb.LAYER_STRIDES),
                     "num_filters": list(bb.NUM_FILTERS), "upsample_strides": list(bb.UPSAMPLE_STRIDES),
                     "upsample_filters": list(bb.NUM_UPSAMPLE_FILTERS)},
        "feature_map_stride": int(a.feature_map_stride),
        "anchor": {"size": [float(v) for v in a.anchor_sizes[0]], "rotations": [float(v) for v in a.anchor_rotations],
                   "bottom_height": float(a.anchor_bottom_heights[0])},
        "anchors_per_location": len(a.anchor_rotations),
        "num_dir_bins": int(head.NUM_DIR_BINS), "dir_offset": float(head.DIR_OFFSET),
        "dir_limit_offset": float(head.DIR_LIMIT_OFFSET),
        "post": {"score_thresh": float(post.SCORE_THRESH), "nms_thresh": float(post.NMS_CONFIG.NMS_THRESH),
                 "nms_pre_maxsize": int(post.NMS_CONFIG.NMS_PRE_MAXSIZE),
                 "nms_post_maxsize": int(post.NMS_CONFIG.NMS_POST_MAXSIZE)},
        "eval_score_threshold": float(data.get("EVAL_SCORE_THRESHOLD", 0.3)),
    }


def geometry(dataset_cfg: dict, data_path: Path) -> dict:
    rig = av24_geometry.Av24Rig.load(dataset_cfg["av24_urdf"])
    t_ram_sensor = rig.transform("rear_axle_middle", "radar_rear")
    cog = dataset_cfg["ego_odom_child_frame"]
    sensor_in_cog = (rig.transform(cog, "rear_axle_middle") @ np.append(t_ram_sensor[:3, 3], 1.0))[:3]
    # The mounting the training set was written with must be the one we ship.
    calib = json.loads((data_path / "v1.0-trainval" / "calibrated_sensor.json").read_text())
    for record in calib:
        built = np.eye(4)
        w, x, y, z = record["rotation"]  # nuScenes stores w, x, y, z
        built[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        built[:3, 3] = record["translation"]
        if not np.allclose(built, t_ram_sensor, atol=1e-6):
            raise SystemExit(f"URDF radar_rear mounting differs from {data_path} calibrated_sensor {record['token']}")
    return {"t_ram_sensor": t_ram_sensor.tolist(), "sensor_in_cog": sensor_in_cog.tolist()}


def export(args) -> Path:
    cfg = load_cfg(args.cfg)
    config = load_config(REPO_ROOT / "config" / "pipeline.yaml")
    dataset_cfg = config["dataset"]
    hp = hyperparameters(cfg)
    if hp["feature_names"][:-1] != list(dataset_cfg["feature_names"]):
        raise SystemExit(f"cfg features {hp['feature_names']} vs builder {dataset_cfg['feature_names']} + time_lag")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["model_state"].items()
             if k != "global_step"}
    net = rrm.PointPillar(hp)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
    bundle = {
        "bundle_version": rrm.BUNDLE_VERSION,
        "state_dict": net.state_dict(),
        "model": hp,
        "features": {k: dataset_cfg[k] for k in FEATURE_KEYS},
        "geometry": geometry(dataset_cfg, Path(cfg.DATA_CONFIG.DATA_PATH)),
        "provenance": {"cfg": args.cfg, "checkpoint": str(Path(args.ckpt).resolve()),
                       "epoch": int(ckpt.get("epoch", -1)), "dataset": cfg.DATA_CONFIG.DATA_PATH,
                       "note": args.note},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), epoch {bundle['provenance']['epoch']}")
    return out


def check(args, bundle_path: Path) -> None:
    """Same samples through OpenPCDet and rear_radar_model."""
    gpu = torch.cuda.is_available()
    if not gpu:
        torch.Tensor.cuda = lambda self, *a, **k: self  # OpenPCDet builds its anchors with .cuda()
    from pcdet.datasets.radar.radar_dataset import RadarDataset
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    cfg = load_cfg(args.cfg)
    logger = common_utils.create_logger()
    dataset = RadarDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False,
                           root_path=Path(cfg.DATA_CONFIG.DATA_PATH), logger=logger)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(args.ckpt), logger=logger, to_cpu=not gpu)
    model = (model.cuda() if gpu else model).eval()
    det = rrm.RearRadarDetector.load(bundle_path, device="cuda" if gpu else "cpu")

    rng = np.random.default_rng(0)
    picks = rng.choice(len(dataset), size=min(args.check, len(dataset)), replace=False)
    worst = {"cls": 0.0, "box": 0.0, "dir": 0.0, "box_final": 0.0, "score_final": 0.0}
    count_mismatch = voxel_mismatch = 0
    for n, i in enumerate(picks):
        data = dataset[int(i)]
        # voxels: same pillars with the same points (OpenPCDet's order is spconv's)
        vox, num, coords = rrm.voxelize(data["points"], hp(det, "voxel_size"), hp(det, "point_cloud_range"),
                                        hp(det, "max_points_per_voxel"), hp(det, "max_voxels"))
        theirs = {tuple(c): k for k, c in enumerate(data["voxel_coords"])}
        if len(theirs) != len(coords) or any(
            tuple(c) not in theirs
            or int(data["voxel_num_points"][theirs[tuple(c)]]) != int(num[k])
            or not np.allclose(np.sort(data["voxels"][theirs[tuple(c)], :num[k]], axis=0), np.sort(vox[k, :num[k]], axis=0))
            for k, c in enumerate(coords)
        ):
            voxel_mismatch += 1
        batch = dataset.collate_batch([data])
        load_data_to_gpu(batch)  # a CPU copy when Tensor.cuda is patched
        with torch.no_grad():
            for module in model.module_list:
                batch = module(batch)
            head = model.dense_head.forward_ret_dict
            mine = det.raw_outputs(data["points"])
            for key, ours in zip(("cls", "box", "dir"), mine):
                ref = head[{"cls": "cls_preds", "box": "box_preds", "dir": "dir_cls_preds"}[key]]
                worst[key] = max(worst[key], float((ref - ours).abs().max()))
            if gpu:
                pred, _ = model.post_processing(batch)
                ref_boxes, ref_scores = pred[0]["pred_boxes"].cpu().numpy(), pred[0]["pred_scores"].cpu().numpy()
            else:
                ref_boxes = ref_scores = None
            boxes, scores = det.detect(data["points"])
        if ref_boxes is not None:
            keep_ref, keep = ref_scores >= 0.3, scores >= 0.3
            if keep_ref.sum() != keep.sum():
                count_mismatch += 1
            elif keep.any():
                worst["box_final"] = max(worst["box_final"], float(np.abs(ref_boxes[keep_ref] - boxes[keep]).max()))
                worst["score_final"] = max(worst["score_final"], float(np.abs(ref_scores[keep_ref] - scores[keep]).max()))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(picks)} {worst}", flush=True)
    print(f"checked {len(picks)} samples of {cfg.DATA_CONFIG.DATA_SPLIT['test']} on {'GPU' if gpu else 'CPU'}")
    print(f"  max |diff| raw head outputs: cls {worst['cls']:.2e}, box {worst['box']:.2e}, dir {worst['dir']:.2e}")
    print(f"  samples whose voxels differ: {voxel_mismatch}")
    if gpu:
        print(f"  samples with a different number of boxes >= 0.3: {count_mismatch}")
        print(f"  max |diff| final boxes {worst['box_final']:.2e}, scores {worst['score_final']:.2e}")
    ok = max(worst["cls"], worst["box"], worst["dir"]) < 1e-3 and voxel_mismatch == 0 and count_mismatch == 0
    print("PARITY OK" if ok else "PARITY FAILED")
    if not ok:
        raise SystemExit(1)


def hp(det: rrm.RearRadarDetector, key: str):
    return det.hp[key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", default=DEFAULT_CFG)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--out", default=str(HERE / "models" / "rear_radarpillar_v5.pt"))
    parser.add_argument("--note", default="v5 seed 1 epoch 15; val_qa centre-1m AP 74.5 (3-epoch smoothed)")
    parser.add_argument("--check", type=int, default=0, metavar="N", help="compare N eval samples with OpenPCDet")
    args = parser.parse_args()
    out = export(args)
    if args.check:
        check(args, out)


if __name__ == "__main__":
    main()
