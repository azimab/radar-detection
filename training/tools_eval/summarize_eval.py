#!/usr/bin/env python3
"""Summarise a checkpoint sweep: which epoch is best, and how it behaves.

``test.py --eval_all`` writes one ``result.pkl`` of detection annotations per
checkpoint but only logs the headline numbers. This re-reads each one, scores it
with the dataset's own evaluator, sweeps the score threshold to find the best
operating point, and names the best epoch.

Every metric here comes from ``pcdet.datasets.radar.radar_eval`` -- the same code
the model's own evaluation path runs -- so a sweep and an in-training number
cannot disagree about what AP means.

Epoch selection. The primary metric is AP_R40 at a 1 m centre distance, because
the consumer is a Kalman tracker that uses box centres. BEV IoU 0.5 is harsh on a
4.9 m box whose error is mostly along-track. It is still reported. Single-epoch
val AP swings by ~4.5 AP, so a smoothed pick is reported as well: the best mean
over ``--smooth_window`` consecutive epochs. The window's epochs are the natural
input to ``average_checkpoints.py``.

``--from_summary`` re-runs only the selection on an existing summary JSON, without
re-scoring and without importing pcdet. ``--out_dir`` redirects the written summary,
for example when the sweep directory must stay untouched.
"""

import os
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

SELECTION_METRIC = "Car_center_AP_R40@1.0m"
IOU_METRIC = "Car_bev_AP_R40@0.5"


def rolling_best(rows, metric, window):
    """Best mean of ``metric`` over ``window`` consecutive epochs.

    Only runs of epochs with no gaps count, so a missing checkpoint is never
    averaged over. Returns None if no run is long enough. The reported epoch is the
    window's middle one (upper middle for an even window).
    """
    series = sorted((row["epoch"], row["metrics"].get(metric)) for row in rows)
    series = [(e, v) for e, v in series if v is not None]
    best = None
    for start in range(len(series) - window + 1):
        chunk = series[start:start + window]
        if chunk[-1][0] - chunk[0][0] != window - 1:
            continue
        mean = float(np.mean([v for _, v in chunk]))
        if best is None or mean > best["mean"]:
            best = {
                "epoch": chunk[window // 2][0],
                "mean": mean,
                "window_epochs": [e for e, _ in chunk],
            }
    return best


def epoch_noise_sd(rows, metric):
    """Per-epoch noise estimate: sd of consecutive-epoch differences / sqrt(2)."""
    series = [v for _, v in sorted((r["epoch"], r["metrics"].get(metric)) for r in rows)
              if v is not None]
    if len(series) < 3:
        return None
    return float(np.std(np.diff(series), ddof=1) / np.sqrt(2))


def select(rows, window):
    """All selection results, as the (added) summary JSON fields."""
    best = max(rows, key=lambda row: row["metrics"].get(SELECTION_METRIC, -1))
    best_iou = max(rows, key=lambda row: row["metrics"].get(IOU_METRIC, -1))
    smooth = rolling_best(rows, SELECTION_METRIC, window)
    smooth_iou = rolling_best(rows, IOU_METRIC, window)
    return {
        "best_epoch": best["epoch"],
        "best_value": best["metrics"].get(SELECTION_METRIC),
        "best_epoch_bev_iou0.5": best_iou["epoch"],
        "best_value_bev_iou0.5": best_iou["metrics"].get(IOU_METRIC),
        "smoothing_window": window,
        "smoothed_best_epoch": smooth["epoch"] if smooth else None,
        "smoothed_best_value": smooth["mean"] if smooth else None,
        "smoothed_best_window_epochs": smooth["window_epochs"] if smooth else None,
        "smoothed_best_epoch_bev_iou0.5": smooth_iou["epoch"] if smooth_iou else None,
        "smoothed_best_value_bev_iou0.5": smooth_iou["mean"] if smooth_iou else None,
        "smoothed_best_window_epochs_bev_iou0.5":
            smooth_iou["window_epochs"] if smooth_iou else None,
        "epoch_noise_sd": {
            SELECTION_METRIC: epoch_noise_sd(rows, SELECTION_METRIC),
            IOU_METRIC: epoch_noise_sd(rows, IOU_METRIC),
        },
    }


def print_selection(sel, rows_by_epoch):
    window = sel["smoothing_window"]
    iou_at_best = rows_by_epoch[sel["best_epoch"]]["metrics"].get(IOU_METRIC, float("nan"))
    print("\n" + "=" * 70)
    print(f"BEST EPOCH by {SELECTION_METRIC}: {sel['best_epoch']}  ({sel['best_value']:.4f})"
          f"  [{IOU_METRIC} there: {iou_at_best:.4f}]")
    print(f"best epoch by {IOU_METRIC} (previous rule): {sel['best_epoch_bev_iou0.5']}  "
          f"({sel['best_value_bev_iou0.5']:.4f})")
    for suffix, metric in (("", SELECTION_METRIC), ("_bev_iou0.5", IOU_METRIC)):
        epoch = sel["smoothed_best_epoch" + suffix]
        if epoch is None:
            print(f"SMOOTHED ({window}-epoch mean) by {metric}: not enough consecutive epochs")
            continue
        print(f"SMOOTHED ({window}-epoch mean) by {metric}: epoch {epoch}  "
              f"({sel['smoothed_best_value' + suffix]:.4f}, "
              f"window {sel['smoothed_best_window_epochs' + suffix]})")
    print("per-epoch noise sd estimate: " + ", ".join(
        f"{k}={v:.2f}" if v is not None else f"{k}=n/a"
        for k, v in sel["epoch_noise_sd"].items()))
    win = sel["smoothed_best_window_epochs"]
    if win and len(win) > 1:
        print(f"  to weight-average that window: tools_eval/average_checkpoints.py "
              f"--run_dir <run_dir> --epochs {win[0]}-{win[-1]}")


def sweep_score_thresholds(frames, thresholds, iou_threshold):
    """Precision/recall/F1 across score thresholds, reusing one set of IoU matrices."""
    rows = []
    for threshold in thresholds:
        tp = fp = fn = 0
        for frame in frames:
            gt = frame["gt"]
            keep = np.where(frame["score"] >= threshold)[0]
            order = keep[np.argsort(-frame["score"][keep])]
            taken = np.zeros(len(gt), dtype=bool)
            for pi in order:
                candidates = np.where(~taken)[0]
                if not len(gt) or not len(candidates):
                    fp += 1
                    continue
                best = candidates[np.argmax(frame["iou_bev"][pi, candidates])]
                if frame["iou_bev"][pi, best] < iou_threshold:
                    fp += 1
                    continue
                taken[best] = True
                tp += 1
            fn += int((~taken).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append({
            "score_thresh": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "tp": tp, "fp": fp, "fn": fn,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="output/<exp>/<cfg>/<run_tag>")
    parser.add_argument("--eval_tag", default="val_sweep")
    parser.add_argument("--split", default="val", help="splits.json key; also the eval subdirectory")
    parser.add_argument("--result_split", default=None,
                        help="eval subdirectory to read result.pkl from, if not --split; frames "
                             "outside --split are skipped, so a subset split re-scores a superset's "
                             "predictions without new inference")
    parser.add_argument("--data_path", default=os.path.join(os.environ.get("RADAR_DATA_ROOT", "."), "data_nuscenes"),
                        help="nuScenes dataroot the checkpoints were evaluated on")
    parser.add_argument("--match_iou", type=float, default=0.5)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    parser.add_argument("--out_name", default=None)
    parser.add_argument("--smooth_window", type=int, default=3,
                        help="consecutive epochs averaged for the smoothed pick")
    parser.add_argument("--out_dir", default=None,
                        help="write the summary here instead of the sweep directory")
    parser.add_argument("--from_summary", default=None,
                        help="existing summary JSON: redo only the epoch selection")
    args = parser.parse_args()
    if args.smooth_window < 1:
        parser.error("--smooth_window must be >= 1")

    eval_root = Path(args.run_dir) / "eval" / "eval_all_default" / args.eval_tag
    out_dir = Path(args.out_dir) if args.out_dir else eval_root
    out = out_dir / (args.out_name or f"summary_{args.split}.json")

    if args.from_summary:
        if out.resolve() == Path(args.from_summary).resolve():
            raise SystemExit("--from_summary would overwrite its input; pass --out_dir or --out_name")
        previous = json.loads(Path(args.from_summary).read_text())
        rows = previous["epochs"]
        sel = select(rows, args.smooth_window)
        previous.update({"selection_metric": SELECTION_METRIC, **sel})
        out_dir.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(previous, indent=2))
        print_selection(sel, {row["epoch"]: row for row in rows})
        print(f"wrote {out}")
        return

    from pcdet.datasets.radar import radar_eval
    from pcdet.datasets.radar.radar_dataset import load_nuscenes_infos

    result_files = sorted(
        eval_root.glob(f"epoch_*/{args.result_split or args.split}/result.pkl"),
        key=lambda path: int(path.parent.parent.name.split("_")[-1]),
    )
    if not result_files:
        raise SystemExit(f"no result.pkl found under {eval_root}")

    # The same loader the dataset uses, so GT here is exactly what the model was scored on.
    infos = load_nuscenes_infos(args.data_path, args.split, ["Car"])
    gt_by_frame = {
        info["frame_id"]: (
            np.asarray(info["annos"]["gt_boxes_lidar"], dtype=np.float32).reshape(-1, 7),
            np.asarray(info["annos"]["num_points_in_gt"]).reshape(-1),
        )
        for info in infos
    }

    thresholds = np.arange(0.1, 0.95, 0.05)
    rows, reports = [], {}
    for result_file in result_files:
        epoch = int(result_file.parent.parent.name.split("_")[-1])
        with open(result_file, "rb") as stream:
            det_annos = pickle.load(stream)
        frames = radar_eval.build_frames(det_annos, gt_by_frame)
        report, metrics = radar_eval.evaluate(
            det_annos, gt_by_frame, frames=frames,
            iou_thresholds=(0.3, 0.5, 0.7), score_threshold=args.score_thresh,
            range_bins=((0, 20), (20, 40), (40, 60), (60, 105)),
        )
        sweep = sweep_score_thresholds(frames, thresholds, args.match_iou)
        reports[epoch] = report
        rows.append({
            "epoch": epoch,
            "metrics": metrics,
            "threshold_sweep": sweep,
            "best_f1_operating_point": max(sweep, key=lambda row: row["f1"]),
        })
        print(
            f"epoch {epoch:>3}  "
            f"ctr_R40@1m={metrics[SELECTION_METRIC]:7.3f}  "
            f"bev_R40@.5={metrics['Car_bev_AP_R40@0.5']:7.3f}  "
            f"3d_R40@.5={metrics['Car_3d_AP_R40@0.5']:7.3f}  "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
            f"F1={metrics['f1']:.3f}  "
            f"pos_err_med={metrics.get('BEV_position_error_median_m', float('nan')):.3f} m",
            flush=True,
        )

    sel = select(rows, args.smooth_window)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_dir": args.run_dir,
        "split": args.split,
        "result_split": args.result_split or args.split,
        "eval_tag": args.eval_tag,
        # Centre distance rather than BEV IoU 0.5: the downstream Kalman tracker uses
        # box centres, and IoU 0.5 is harsh on along-track error of a 4.9 m box.
        # Neither is 3D, on purpose: FRGen21 elevation is too noisy to voxelise, GT
        # box height comes from a ground-surface model, and the anchor has a single
        # bottom height, so 3D IoU mostly measures those choices rather than the
        # detector.
        "selection_metric": SELECTION_METRIC,
        **sel,
        "velocity_metrics": "not applicable - the model has no velocity head",
        "epochs": rows,
    }, indent=2))

    print_selection(sel, {row["epoch"]: row for row in rows})
    print(reports[sel["best_epoch"]])
    best = next(row for row in rows if row["epoch"] == sel["best_epoch"])
    operating = best["best_f1_operating_point"]
    print(f"  best F1 operating point: score>={operating['score_thresh']:.2f}  "
          f"P={operating['precision']:.3f} R={operating['recall']:.3f} "
          f"F1={operating['f1']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
