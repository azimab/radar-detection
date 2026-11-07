"""Rear-radar RadarPillar inference without OpenPCDet: plain PyTorch + NumPy, CPU or GPU.

The trained network is a stock OpenPCDet PointPillar (PillarVFE -> PointPillarScatter ->
BaseBEVBackbone -> AnchorHeadSingle). This file re-implements its eval-mode forward pass
and post-processing so that the checkpoint runs on a laptop with nothing but ``torch``:
no CUDA extensions, no spconv, no pcdet install. Module names match OpenPCDet's, so the
checkpoint's ``model_state`` loads as is.

Input features are made by the dataset builder's own ``scan_features`` (imported, not
copied), and scans are stacked the way ``RadarDataset.get_points`` stacks them.
``export_model.py`` writes the bundle this loads and checks it against OpenPCDet.

    det = RearRadarDetector.load("live/models/rear_radarpillar_v5.pt")
    result = det.step(t, raw, v_cog, w_cog)   # raw: SOURCE_FIELDS rows, radar frame
"""

from __future__ import annotations

import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_nuscenes_dataset as builder  # noqa: E402  (scan_features, pointcloud_array)

BUNDLE_VERSION = 1


# --------------------------------------------------------------------------------------
# network: OpenPCDet PointPillar, eval mode
# --------------------------------------------------------------------------------------


class PFNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return F.relu(x).max(dim=1)[0]


class PillarVFE(nn.Module):
    def __init__(self, num_point_features: int, num_filters: int, voxel_size, pc_range):
        super().__init__()
        # USE_ABSOLUTE_XYZ, no distance, no velocity decomposition: point features +
        # offsets from the pillar's point mean (3) and from the pillar centre (3).
        self.pfn_layers = nn.ModuleList([PFNLayer(num_point_features + 6, num_filters)])
        self.voxel_size = [float(v) for v in voxel_size]
        self.offset = [self.voxel_size[i] / 2 + float(pc_range[i]) for i in range(3)]

    def forward(self, voxels: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """voxels (V, P, C), num_points (V,), coords (V, 3) as z, y, x."""
        xyz = voxels[:, :, :3]
        mean = xyz.sum(dim=1, keepdim=True) / num_points.type_as(voxels).view(-1, 1, 1)
        f_cluster = xyz - mean
        centre = torch.stack(
            [coords[:, 2 - i].to(voxels.dtype) * self.voxel_size[i] + self.offset[i] for i in range(3)], dim=1
        )
        f_center = xyz - centre[:, None, :]
        features = torch.cat([voxels, f_cluster, f_center], dim=-1)
        pad = torch.arange(voxels.shape[1], device=voxels.device)[None, :] < num_points[:, None]
        features = features * pad.unsqueeze(-1).type_as(features)
        return self.pfn_layers[0](features)


class BaseBEVBackbone(nn.Module):
    def __init__(self, c_in: int, layer_nums, layer_strides, num_filters, upsample_strides, upsample_filters):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        channels = [c_in, *num_filters[:-1]]
        for i, n in enumerate(layer_nums):
            layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(channels[i], num_filters[i], 3, stride=layer_strides[i], padding=0, bias=False),
                nn.BatchNorm2d(num_filters[i], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(n):
                layers += [
                    nn.Conv2d(num_filters[i], num_filters[i], 3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[i], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ]
            self.blocks.append(nn.Sequential(*layers))
            stride = upsample_strides[i]
            if stride < 1:
                raise NotImplementedError("fractional upsample strides are not used by this model")
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(num_filters[i], upsample_filters[i], stride, stride=stride, bias=False),
                nn.BatchNorm2d(upsample_filters[i], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ups = []
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        return torch.cat(ups, dim=1)


class AnchorHeadSingle(nn.Module):
    def __init__(self, c_in: int, num_anchors: int, num_class: int, num_dir_bins: int, code_size: int = 7):
        super().__init__()
        self.conv_cls = nn.Conv2d(c_in, num_anchors * num_class, 1)
        self.conv_box = nn.Conv2d(c_in, num_anchors * code_size, 1)
        self.conv_dir_cls = nn.Conv2d(c_in, num_anchors * num_dir_bins, 1)

    def forward(self, x: torch.Tensor):
        return tuple(conv(x).permute(0, 2, 3, 1).contiguous() for conv in (self.conv_cls, self.conv_box, self.conv_dir_cls))


class PointPillar(nn.Module):
    """Raw network outputs; decoding and NMS are in RearRadarDetector."""

    def __init__(self, hp: dict):
        super().__init__()
        self.grid = [int(v) for v in hp["grid_size"]]  # x, y, z
        self.vfe = PillarVFE(hp["num_point_features"], hp["vfe_filters"], hp["voxel_size"], hp["point_cloud_range"])
        self.num_bev = hp["vfe_filters"] * self.grid[2]
        b = hp["backbone"]
        self.backbone_2d = BaseBEVBackbone(self.num_bev, b["layer_nums"], b["layer_strides"], b["num_filters"],
                                           b["upsample_strides"], b["upsample_filters"])
        self.dense_head = AnchorHeadSingle(sum(b["upsample_filters"]), hp["anchors_per_location"],
                                           hp["num_class"], hp["num_dir_bins"])

    def forward(self, voxels, num_points, coords):
        pillar = self.vfe(voxels, num_points, coords)
        nx, ny, nz = self.grid
        canvas = pillar.new_zeros(pillar.shape[1], nz * ny * nx)
        index = (coords[:, 0] + coords[:, 1] * nx + coords[:, 2]).long()
        canvas[:, index] = pillar.t()
        bev = canvas.view(1, pillar.shape[1] * nz, ny, nx)
        return self.dense_head(self.backbone_2d(bev))


# --------------------------------------------------------------------------------------
# pre/post-processing (OpenPCDet DataProcessor, AnchorGenerator, ResidualCoder, NMS)
# --------------------------------------------------------------------------------------


def voxelize(points: np.ndarray, voxel_size, pc_range, max_points: int, max_voxels: int):
    """spconv PointToVoxel semantics: voxels in order of first point, points in input order."""
    voxel_size = np.asarray(voxel_size, np.float32)
    pc_range = np.asarray(pc_range, np.float32)
    grid = np.round((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int64)
    cell = np.floor((points[:, :3] - pc_range[:3]) / voxel_size).astype(np.int64)
    valid = np.all((cell >= 0) & (cell < grid), axis=1)
    points, cell = points[valid], cell[valid]
    if not len(points):
        return (np.zeros((0, max_points, points.shape[1]), np.float32), np.zeros(0, np.int32),
                np.zeros((0, 3), np.int32))
    linear = (cell[:, 2] * grid[1] + cell[:, 1]) * grid[0] + cell[:, 0]
    uniq, first, inverse = np.unique(linear, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")[:max_voxels]  # voxel ids by first appearance
    rank = np.full(len(uniq), -1, np.int64)
    rank[order] = np.arange(len(order))
    vid = rank[inverse.reshape(-1)]
    keep = vid >= 0
    points, vid = points[keep], vid[keep]
    # slot of each point within its voxel, in input order
    sort = np.argsort(vid, kind="stable")
    counts = np.bincount(vid, minlength=len(order))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    slot = np.empty(len(vid), np.int64)
    slot[sort] = np.arange(len(vid)) - np.repeat(starts, counts)
    keep = slot < max_points
    voxels = np.zeros((len(order), max_points, points.shape[1]), np.float32)
    voxels[vid[keep], slot[keep]] = points[keep]
    lin = uniq[order]
    coords = np.stack([lin // (grid[0] * grid[1]), (lin // grid[0]) % grid[1], lin % grid[0]], axis=1)
    return voxels, np.minimum(counts, max_points).astype(np.int32), coords.astype(np.int32)


def make_anchors(hp: dict) -> np.ndarray:
    """(H * W * A, 7) in the head's output order (y, x, rotation), align_center=True."""
    r = hp["point_cloud_range"]
    stride = hp["feature_map_stride"]
    nx, ny = hp["grid_size"][0] // stride, hp["grid_size"][1] // stride
    a = hp["anchor"]
    l, w, h = a["size"]
    xs = r[0] + (r[3] - r[0]) / nx * (np.arange(nx) + 0.5)
    ys = r[1] + (r[4] - r[1]) / ny * (np.arange(ny) + 0.5)
    rots = np.asarray(a["rotations"], np.float64)
    yy, xx, rr = np.meshgrid(ys, xs, rots, indexing="ij")
    out = np.stack([xx, yy, np.full_like(xx, a["bottom_height"] + h / 2), np.full_like(xx, l),
                    np.full_like(xx, w), np.full_like(xx, h), rr], axis=-1)
    return out.reshape(-1, 7).astype(np.float32)


def limit_period(val, offset, period):
    return val - torch.floor(val / period + offset) * period


def decode_boxes(box_preds: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    xa, ya, za, la, wa, ha, ra = anchors.unbind(-1)
    xt, yt, zt, lt, wt, ht, rt = box_preds.unbind(-1)
    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    return torch.stack([xt * diagonal + xa, yt * diagonal + ya, zt * ha + za,
                        torch.exp(lt) * la, torch.exp(wt) * wa, torch.exp(ht) * ha, rt + ra], dim=-1)


def bev_corners(box: np.ndarray) -> np.ndarray:
    x, y, l, w, yaw = box[0], box[1], box[3], box[4], box[6]
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[l / 2, w / 2], [-l / 2, w / 2], [-l / 2, -w / 2], [l / 2, -w / 2]])
    return local @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def _clip(subject: list, a: np.ndarray, b: np.ndarray) -> list:
    """Sutherland-Hodgman: keep the part of polygon ``subject`` left of edge a->b."""
    side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    out = []
    for i, p in enumerate(subject):
        q = subject[i - 1]
        sp, sq = side(p), side(q)
        if sp >= 0:
            if sq < 0:
                out.append(q + (p - q) * (sq / (sq - sp)))
            out.append(p)
        elif sq >= 0:
            out.append(q + (p - q) * (sq / (sq - sp)))
    return out


def _area(poly) -> float:
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly)
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def bev_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Rotated BEV IoU of two (7,) boxes, as iou3d_nms's boxes_iou_bev."""
    ca, cb = bev_corners(a), bev_corners(b)
    poly = list(ca)
    for i in range(4):
        poly = _clip(poly, cb[i], cb[(i + 1) % 4])
        if not poly:
            return 0.0
    inter = _area(poly)
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / max(union, 1e-8)


def nms_bev(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy rotated-BEV NMS. Boxes must already be sorted by descending score."""
    keep = []
    suppressed = np.zeros(len(boxes), bool)
    centres, radius = boxes[:, :2], 0.5 * np.hypot(boxes[:, 3], boxes[:, 4])
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(i)
        near = np.where(~suppressed & (np.linalg.norm(centres - centres[i], axis=1) < radius + radius[i]))[0]
        for j in near[near > i]:
            if bev_iou(boxes[i], boxes[j]) > threshold:
                suppressed[j] = True
    return np.asarray(keep, np.int64)


# --------------------------------------------------------------------------------------
# detector: features -> stack -> network -> boxes
# --------------------------------------------------------------------------------------


@dataclass
class Detections:
    boxes: np.ndarray     # (N, 7) x, y, z, l, w, h, yaw in rear_axle_middle
    scores: np.ndarray    # (N,)
    features: np.ndarray  # current scan, (M, 11) builder feature columns
    stacked: np.ndarray   # model input, (K, 12): features + time_lag
    previous: int         # previous scans stacked
    ego_speed: float      # m/s, at the scan's time
    timings_ms: dict


class RearRadarDetector:
    def __init__(self, bundle: dict, device: str | None = None):
        if bundle.get("bundle_version") != BUNDLE_VERSION:
            raise ValueError(f"bundle version {bundle.get('bundle_version')}, expected {BUNDLE_VERSION}")
        self.hp = bundle["model"]
        self.features_cfg = bundle["features"]
        self.geometry = {k: np.asarray(v, np.float64) for k, v in bundle["geometry"].items()}
        self.info = bundle.get("provenance", {})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = PointPillar(self.hp)
        self.net.load_state_dict(bundle["state_dict"])
        self.net.to(self.device).eval()
        self.anchors = torch.from_numpy(make_anchors(self.hp)).to(self.device)
        self.num_sweeps = int(self.hp["num_sweeps"])
        self.max_lag = float(self.hp["max_sweep_lag_s"])
        self.history: deque[tuple[float, np.ndarray]] = deque(maxlen=max(self.num_sweeps - 1, 0))

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> "RearRadarDetector":
        return cls(torch.load(path, map_location="cpu", weights_only=True), device)

    # ------------------------------------------------------------------ inputs
    def features(self, raw: np.ndarray, v_cog: np.ndarray, w_cog: np.ndarray) -> np.ndarray:
        """``raw``: builder.SOURCE_FIELDS columns in the radar's own frame."""
        raw = raw[np.isfinite(raw).all(axis=1)]
        features, _ = builder.scan_features(raw, self.geometry["t_ram_sensor"], self.geometry["sensor_in_cog"],
                                            v_cog, w_cog, self.features_cfg)
        return features

    def stack(self, t: float, features: np.ndarray) -> tuple[np.ndarray, int]:
        """Current scan, then previous scans newest first, as RadarDataset.get_points does."""
        parts = [np.hstack([features, np.zeros((len(features), 1), np.float32)])]
        previous = 0
        for t_prev, prev in reversed(self.history):
            lag = t - t_prev
            # The dataset stops at the first scan older than MAX_SWEEP_LAG_S.
            if lag > self.max_lag or lag <= 0:
                break
            parts.append(np.hstack([prev, np.full((len(prev), 1), lag, np.float32)]))
            previous += 1
        return np.concatenate(parts, axis=0), previous

    def reset(self) -> None:
        self.history.clear()

    # ------------------------------------------------------------------ model
    @torch.no_grad()
    def raw_outputs(self, points: np.ndarray):
        """Network outputs for model-input points, as OpenPCDet's dense head gives them."""
        hp = self.hp
        r = hp["point_cloud_range"]
        # DataProcessor.mask_points_and_boxes_outside_range (x and y only, inclusive)
        inside = (points[:, 0] >= r[0]) & (points[:, 0] <= r[3]) & (points[:, 1] >= r[1]) & (points[:, 1] <= r[4])
        voxels, num, coords = voxelize(points[inside], hp["voxel_size"], r, hp["max_points_per_voxel"],
                                       hp["max_voxels"])
        if not len(voxels):
            return None
        to = lambda a: torch.from_numpy(a).to(self.device)
        return self.net(to(voxels), to(num), to(coords))

    @torch.no_grad()
    def detect(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = self.raw_outputs(points)
        if out is None:
            return np.zeros((0, 7), np.float32), np.zeros((0,), np.float32)
        cls, box, direction = (o.view(-1, o.shape[-1] // self.hp["anchors_per_location"]) for o in out)
        scores = torch.sigmoid(cls.max(dim=-1)[0])
        post = self.hp["post"]
        keep = scores >= post["score_thresh"]
        if not keep.any():
            return np.zeros((0, 7), np.float32), np.zeros((0,), np.float32)
        scores, idx = scores[keep], torch.nonzero(keep)[:, 0]
        scores, order = torch.topk(scores, k=min(post["nms_pre_maxsize"], len(scores)))
        idx = idx[order]
        boxes = decode_boxes(box[idx], self.anchors[idx])
        period = 2 * math.pi / self.hp["num_dir_bins"]
        dir_labels = direction[idx].argmax(dim=-1).to(boxes.dtype)
        rot = limit_period(boxes[:, 6] - self.hp["dir_offset"], self.hp["dir_limit_offset"], period)
        boxes[:, 6] = rot + self.hp["dir_offset"] + period * dir_labels
        boxes, scores = boxes.cpu().numpy(), scores.cpu().numpy()
        keep = nms_bev(boxes, scores, post["nms_thresh"])[: post["nms_post_maxsize"]]
        return boxes[keep], scores[keep]

    def step(self, t: float, raw: np.ndarray, v_cog: np.ndarray, w_cog: np.ndarray) -> Detections:
        """One scan in arrival order. ``t`` in seconds; twist in center_of_gravity, body frame."""
        import time

        t0 = time.perf_counter()
        features = self.features(raw, np.asarray(v_cog, np.float64), np.asarray(w_cog, np.float64))
        t1 = time.perf_counter()
        if self.num_sweeps > 1:
            stacked, previous = self.stack(t, features)
            self.history.append((t, features))
        else:
            stacked, previous = features, 0
        boxes, scores = self.detect(stacked)
        t2 = time.perf_counter()
        return Detections(boxes, scores, features, stacked, previous, float(np.linalg.norm(v_cog[:2])),
                          {"features": (t1 - t0) * 1e3, "model": (t2 - t1) * 1e3})
