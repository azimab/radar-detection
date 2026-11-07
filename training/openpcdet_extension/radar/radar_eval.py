"""Single-class detection metrics in lidar coordinates.

There are no camera quantities (2D box height, occlusion, truncation) to tier
objects by, so every GT box counts. IoU is a harsh match criterion for a target
represented by 2-20 radar returns, so a center-distance criterion is reported
alongside it.

AP is 40-point interpolated, computed over score-sorted detections with greedy
one-to-one matching, which is the standard R40 definition without difficulty
tiers.
"""

import numpy as np


def _boxes_iou_bev(boxes_a, boxes_b):
    """(N,7) x (M,7) -> (N,M) BEV IoU, using the compiled op when a GPU is present."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    try:
        import torch

        from ...ops.iou3d_nms import iou3d_nms_utils

        if torch.cuda.is_available():
            a = torch.from_numpy(np.ascontiguousarray(boxes_a)).float().cuda()
            b = torch.from_numpy(np.ascontiguousarray(boxes_b)).float().cuda()
            return iou3d_nms_utils.boxes_iou_bev(a, b).cpu().numpy()
    except Exception:
        pass
    from .rotate_iou_torch import rotate_iou_gpu_eval

    # rotate_iou takes [x, y, dx, dy, angle] and rotates corners the other way.
    a = np.stack([boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 3], boxes_a[:, 4], -boxes_a[:, 6]], 1)
    b = np.stack([boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 3], boxes_b[:, 4], -boxes_b[:, 6]], 1)
    return rotate_iou_gpu_eval(a.astype(np.float32), b.astype(np.float32), -1)


def _height_overlap(boxes_a, boxes_b):
    a_lo = (boxes_a[:, 2] - boxes_a[:, 5] / 2.0)[:, None]
    a_hi = (boxes_a[:, 2] + boxes_a[:, 5] / 2.0)[:, None]
    b_lo = (boxes_b[:, 2] - boxes_b[:, 5] / 2.0)[None, :]
    b_hi = (boxes_b[:, 2] + boxes_b[:, 5] / 2.0)[None, :]
    return np.maximum(np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo), 0.0)


def _boxes_iou_3d(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    bev = _boxes_iou_bev(boxes_a, boxes_b)
    area_a = (boxes_a[:, 3] * boxes_a[:, 4])[:, None]
    area_b = (boxes_b[:, 3] * boxes_b[:, 4])[None, :]
    # Recover the BEV intersection area from the IoU, then extrude it.
    inter_bev = bev * (area_a + area_b) / (1.0 + bev)
    inter = inter_bev * _height_overlap(boxes_a, boxes_b)
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5])[:, None]
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5])[None, :]
    return inter / np.maximum(vol_a + vol_b - inter, 1e-6)


def wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _match(frames, criterion, threshold):
    """Greedy score-ordered one-to-one matching. Returns (scores, tp_flags, n_gt)."""
    scores, flags = [], []
    n_gt = 0
    for frame in frames:
        gt = frame["gt"]
        n_gt += len(gt)
        order = np.argsort(-frame["score"])
        taken = np.zeros(len(gt), dtype=bool)
        cost = frame[criterion]
        for pi in order:
            scores.append(float(frame["score"][pi]))
            if len(gt) == 0:
                flags.append(False)
                continue
            candidates = np.where(~taken)[0]
            if len(candidates) == 0:
                flags.append(False)
                continue
            if criterion == "iou_bev" or criterion == "iou_3d":
                best = candidates[np.argmax(cost[pi, candidates])]
                hit = cost[pi, best] >= threshold
            else:
                best = candidates[np.argmin(cost[pi, candidates])]
                hit = cost[pi, best] <= threshold
            if hit:
                taken[best] = True
            flags.append(bool(hit))
    return np.asarray(scores), np.asarray(flags, dtype=bool), n_gt


def _ap_r40(scores, tp_flags, n_gt):
    """40-point interpolated average precision, in percent."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = np.cumsum(tp_flags[order])
    fp = np.cumsum(~tp_flags[order])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1)
    total = 0.0
    for i in range(1, 41):
        target = i / 40.0
        mask = recall >= target
        total += precision[mask].max() if mask.any() else 0.0
    return 100.0 * total / 40.0


def build_frames(det_annos, gt_by_frame):
    """Pair detections with ground truth and precompute every match criterion once.

    Split out from ``evaluate`` so a caller sweeping score thresholds over the same
    detections pays the IoU cost once instead of once per threshold.
    """
    frames = []
    for anno in det_annos:
        frame_id = str(anno["frame_id"])
        if frame_id not in gt_by_frame:
            continue
        gt, num_points = gt_by_frame[frame_id]
        pred = np.asarray(anno["boxes_lidar"], dtype=np.float32).reshape(-1, 7)
        score = np.asarray(anno["score"], dtype=np.float32).reshape(-1)
        if len(pred) and len(gt):
            distance = np.hypot(
                pred[:, None, 0] - gt[None, :, 0], pred[:, None, 1] - gt[None, :, 1]
            )
        else:
            distance = np.full((len(pred), len(gt)), np.inf)
        frames.append(
            {
                "frame_id": frame_id,
                "gt": gt,
                "num_points": num_points,
                "pred": pred,
                "score": score,
                "iou_bev": _boxes_iou_bev(pred, gt),
                "iou_3d": _boxes_iou_3d(pred, gt),
                "dist": distance,
            }
        )
    return frames


def evaluate(det_annos, gt_by_frame, *, iou_thresholds, score_threshold, range_bins,
             frames=None):
    if frames is None:
        frames = build_frames(det_annos, gt_by_frame)

    result = {}
    lines = ["", "Radar detection metrics (single class, lidar frame)", "-" * 58]
    for threshold in iou_thresholds:
        for criterion, label in (("iou_bev", "BEV"), ("iou_3d", "3D")):
            scores, flags, n_gt = _match(frames, criterion, threshold)
            ap = _ap_r40(scores, flags, n_gt)
            result[f"Car_{label.lower()}_AP_R40@{threshold}"] = ap
            lines.append(f"  {label:3s} AP_R40 @ IoU {threshold:.1f} : {ap:7.3f}")
    for radius in (1.0, 2.0):
        scores, flags, n_gt = _match(frames, "dist", radius)
        ap = _ap_r40(scores, flags, n_gt)
        result[f"Car_center_AP_R40@{radius}m"] = ap
        lines.append(f"  AP_R40 @ centre dist {radius:.0f} m : {ap:7.3f}")

    # Operating-point statistics at a fixed score, which is what a tracker sees.
    tp = fp = fn = 0
    errors, lateral, longitudinal, headings, matched_ranges = [], [], [], [], []
    background_frames = background_false = 0
    per_bin = {bin_: [0, 0] for bin_ in range_bins}
    for frame in frames:
        keep = np.where(frame["score"] >= score_threshold)[0]
        order = keep[np.argsort(-frame["score"][keep])]
        gt = frame["gt"]
        taken = np.zeros(len(gt), dtype=bool)
        if len(gt) == 0:
            background_frames += 1
            if len(order):
                background_false += 1
        for pi in order:
            candidates = np.where(~taken)[0]
            if len(candidates) == 0 or not len(gt):
                fp += 1
                continue
            best = candidates[np.argmax(frame["iou_bev"][pi, candidates])]
            if frame["iou_bev"][pi, best] < 0.5:
                fp += 1
                continue
            taken[best] = True
            tp += 1
            delta = frame["pred"][pi, :2] - gt[best, :2]
            yaw = float(gt[best, 6])
            c, s = np.cos(-yaw), np.sin(-yaw)
            longitudinal.append(abs(c * delta[0] - s * delta[1]))
            lateral.append(abs(s * delta[0] + c * delta[1]))
            errors.append(float(np.hypot(*delta)))
            headings.append(abs(np.degrees(wrap_pi(float(frame["pred"][pi, 6]) - yaw))))
            matched_ranges.append(float(np.hypot(gt[best, 0], gt[best, 1])))
        fn += int((~taken).sum())
        for gi in range(len(gt)):
            r = float(np.hypot(gt[gi, 0], gt[gi, 1]))
            for bin_ in range_bins:
                if bin_[0] <= r < bin_[1]:
                    per_bin[bin_][0] += 1
                    per_bin[bin_][1] += int(taken[gi])
                    break

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result.update(
        {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "background_frames": background_frames,
            "background_false_alarm_rate": background_false / max(background_frames, 1),
        }
    )
    lines += [
        "",
        f"  at score >= {score_threshold:.2f}, BEV IoU >= 0.5",
        f"    P={precision:.3f}  R={recall:.3f}  F1={result['f1']:.3f}  "
        f"(TP {tp} / FP {fp} / FN {fn})",
        f"    background false-alarm rate: {result['background_false_alarm_rate']:.3f} "
        f"({background_false}/{background_frames} empty frames)",
    ]
    if errors:
        for name, values in (
            ("BEV position", errors),
            ("longitudinal", longitudinal),
            ("lateral", lateral),
        ):
            result[f"{name.replace(' ', '_')}_error_median_m"] = float(np.median(values))
            lines.append(f"    {name:>13s} error: median {np.median(values):.3f} m  "
                         f"p95 {np.percentile(values, 95):.3f} m")
        result["heading_error_median_deg"] = float(np.median(headings))
        lines.append(f"    {'heading':>13s} error: median {np.median(headings):.3f} deg")
    lines.append("  recall by range:")
    for bin_, (n, hit) in per_bin.items():
        rate = hit / n if n else float("nan")
        result[f"recall_{bin_[0]}_{bin_[1]}m"] = rate
        lines.append(f"    {bin_[0]:3d}-{bin_[1]:3d} m: n_gt={n:5d}  recall={rate:.4f}")
    lines.append("")
    return "\n".join(lines), result
