#!/usr/bin/env python3
"""Slide figures: v3 on the new Laguna session (2026-07-25, IU opponent) vs the
same-session Laguna holdout (2026-07-26, Purdue opponent).

Both models are shown at their final checkpoint (epoch 35), so no checkpoint is chosen
on the data it is scored on. The v3 seed shown is the best of three on laguna_0725 at
that checkpoint (seed 2; seeds 1/3 score 88.9 / 91.0 centre-1 m AP vs its 91.4). The
holdout model is the single laguna_train model (44 scenes of 07-26).

Every metric comes from pcdet.datasets.radar.radar_eval, the evaluator the sweeps use.

    PYTHONPATH=vendor/RadarPillar/.deps/native:vendor/RadarPillar \
        python3 tools_eval/plot_laguna_0725.py --out figures/laguna_0725
"""

from __future__ import annotations

import os
import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from pcdet.datasets.radar import radar_eval as E  # noqa: E402
from pcdet.datasets.radar.radar_dataset import load_nuscenes_infos  # noqa: E402

OUT = os.path.join(os.environ.get("RADARPILLAR_ROOT", "vendor/RadarPillar"), "output/rear_models")
EPOCH = 35
SETS = {
    # key: (label, detail, dataroot, split, result.pkl, summary json)
    "new": (
        "Laguna 07-25 (new session)",
        "IU opponent · never seen in training",
        os.path.join(os.environ.get("RADAR_DATA_ROOT", "."), "data_nuscenes_2026_09_28"),
        "laguna_0725",
        f"{OUT}/rear_radarpillar_car_v3/rear_radarpillar_v3_seed2/eval/eval_all_default/"
        f"laguna_0725_sweep/epoch_{EPOCH}/laguna_0725/result.pkl",
        f"{OUT}/rear_radarpillar_car_v3/rear_radarpillar_v3_seed2/eval/eval_all_default/"
        "laguna_0725_sweep/summary_laguna_0725.json",
    ),
    "holdout": (
        "Laguna 07-26 holdout",
        "Purdue opponent · same session as training",
        os.path.join(os.environ.get("RADAR_DATA_ROOT", "."), "data_nuscenes"),
        "laguna_holdout",
        f"{OUT}/rear_radarpillar_car_v3_lagunaholdout/rear_radarpillar_v3_laguna_seed1/eval/"
        f"eval_all_default/val_sweep/epoch_{EPOCH}/laguna_holdout/result.pkl",
        f"{OUT}/rear_radarpillar_car_v3_lagunaholdout/rear_radarpillar_v3_laguna_seed1/eval/"
        "eval_all_default/val_sweep/summary_laguna_holdout.json",
    ),
}
BINS = [(0, 20), (20, 40), (40, 60), (60, 105)]
MIN_N = 20  # a band with fewer GT boxes than this is not scored

# Reference palette (dataviz skill, light mode): categorical slots 1-2 validated as a
# pair; text and surface tokens.
C = {
    "new": "#2a78d6",
    "holdout": "#eb6834",
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#8a8984",
    "grid": "#e6e5e1",
    "static": "#c9c8c2",
}
FOOT = ("v3 PointPillars (rear FRGen21 radar, 4 stacked scans), final checkpoint (epoch 35). 07-25: model trained on all "
        "68 scenes of 07-26, best of 3 seeds (88.9 / 91.4 / 91.0). Holdout: same recipe trained on 44 of those scenes, "
        "scored on 16 others.\nAP = R40 average precision; centre-1 m = a detection counts if its centre is within "
        "1 m of the car's.")


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"], "font.size": 13, "font.family": "DejaVu Sans",
        "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"],
        "ytick.color": C["ink2"], "axes.titlecolor": C["ink"], "text.color": C["ink"],
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": C["grid"], "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False,
    })


def headline(fig, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.955, title, fontsize=21, weight="bold", color=C["ink"], va="top")
    fig.text(0.04, 0.895, subtitle, fontsize=13.5, color=C["ink2"], va="top")
    fig.text(0.04, 0.025, FOOT, fontsize=9.5, color=C["muted"], va="bottom", wrap=True)


def load(key):
    _, _, root, split, result, summary = SETS[key]
    gt = {
        i["frame_id"]: (np.asarray(i["annos"]["gt_boxes_lidar"], np.float32).reshape(-1, 7),
                        np.asarray(i["annos"]["num_points_in_gt"]).reshape(-1), i)
        for i in load_nuscenes_infos(root, split, ["Car"])
    }
    det = pickle.load(open(result, "rb"))
    metrics = {r["epoch"]: r["metrics"] for r in json.load(open(summary))["epochs"]}[EPOCH]
    return gt, det, metrics


def match(gt, det, criterion, threshold, lo=0.0, hi=1e9):
    """Scores, TP flags and GT count, with GT and detections both restricted to [lo, hi) m."""
    g2, d2 = {}, []
    for k, (b, n, _) in gt.items():
        m = (np.hypot(b[:, 0], b[:, 1]) >= lo) & (np.hypot(b[:, 0], b[:, 1]) < hi)
        g2[k] = (b[m], n[m])
    for a in det:
        b = np.asarray(a["boxes_lidar"]).reshape(-1, 7)
        m = (np.hypot(b[:, 0], b[:, 1]) >= lo) & (np.hypot(b[:, 0], b[:, 1]) < hi)
        d2.append({**a, "boxes_lidar": b[m], "score": np.asarray(a["score"])[m]})
    return E._match(E.build_frames(d2, g2), criterion, threshold)


# ------------------------------------------------------------------------------------
# figure 1: accuracy by distance
# ------------------------------------------------------------------------------------
def fig_distance(data, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.74, bottom=0.2)
    groups = [f"{lo}–{hi} m" for lo, hi in BINS] + ["All ranges"]
    width, gap = 0.36, 0.03
    for j, key in enumerate(("new", "holdout")):
        gt, det, _ = data[key]
        for i, (lo, hi) in enumerate(BINS + [(0, 105)]):
            sc, fl, n = match(gt, det, "dist", 1.0, lo, hi)
            x = i + (j - 0.5) * (width + gap) + (0.35 if i == len(BINS) else 0)
            if n < MIN_N:
                ax.text(x, 2, f"n={n}\nnot scored", ha="center", va="bottom", fontsize=10, color=C["muted"])
                continue
            ap = E._ap_r40(sc, fl, n)
            ax.bar(x, ap, width, color=C[key], edgecolor=C["surface"], linewidth=2, zorder=3)
            ax.text(x, ap + 1.2, f"{ap:.0f}", ha="center", va="bottom", fontsize=13, color=C["ink"],
                    weight="bold")
            ax.text(x, -2, f"n={n:,}", ha="center", va="top", fontsize=10, color=C["muted"])
    ax.axvline(len(BINS) - 0.5 + 0.17, color=C["grid"], linewidth=1.2)
    ax.set_xticks([i + (0.35 if i == len(BINS) else 0) for i in range(len(groups))])
    ax.set_xticklabels(groups, fontsize=13, color=C["ink"])
    ax.tick_params(axis="x", pad=28, length=0)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Centre-1 m AP")
    ax.grid(axis="x", visible=False)
    for key in ("new", "holdout"):
        ax.bar(0, 0, color=C[key], label=f"{SETS[key][0]}  ({SETS[key][1]})")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1, fontsize=12, handlelength=1.2)
    headline(fig, "Close range transfers to a new session; long range does not",
             "Detection accuracy by distance to the opponent. n = labelled cars in each band.")
    fig.savefig(out / "1_accuracy_by_distance.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------
# figure 2: precision-recall
# ------------------------------------------------------------------------------------
def pr_curve(scores, flags, n):
    order = np.argsort(-scores)
    tp = np.cumsum(flags[order])
    fp = np.cumsum(~flags[order])
    return tp / max(n, 1), tp / np.maximum(tp + fp, 1), scores[order]


def fig_pr(data, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.76, bottom=0.2, wspace=0.08)
    panels = [("dist", 1.0, "Match: centre within 1 m"), ("iou_bev", 0.5, "Match: top-down box overlap ≥ 0.5 (IoU)")]
    for ax, (crit, thr, title) in zip(axes, panels):
        for key in ("new", "holdout"):
            gt, det, _ = data[key]
            sc, fl, n = match(gt, det, crit, thr)
            rec, prec, s = pr_curve(sc, fl, n)
            ap = E._ap_r40(sc, fl, n)
            k = np.searchsorted(-s, -0.3, side="right") - 1  # last detection with score >= 0.3
            ax.plot(rec, prec, color=C[key], linewidth=2.2,
                    label=f"{SETS[key][0]}   AP {ap:.0f}\n    at score ≥ 0.3:  precision {prec[k]:.2f}, recall {rec[k]:.2f}")
            ax.plot(rec[k], prec[k], "o", color=C[key], markersize=10, markeredgecolor=C["surface"],
                    markeredgewidth=2, zorder=5)
        ax.set_title(title, fontsize=14, loc="left", pad=10)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0.5, 1.01)
        ax.set_xlabel("Recall  (share of cars found)")
        ax.legend(loc="lower left", fontsize=11, labelspacing=0.9)
    axes[0].set_ylabel("Precision  (share of detections that are real)")
    headline(fig, "Precision–recall on the new Laguna session",
             "Each curve sweeps the detection score threshold; the dot is the deployed threshold (0.3). "
             "Precision axis starts at 0.5.")
    fig.savefig(out / "2_precision_recall.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------
# figure 3: metric table
# ------------------------------------------------------------------------------------
def fig_table(data, out: Path) -> None:
    rows = [
        ("Centre-1 m AP", "Car_center_AP_R40@1.0m", "{:.1f}", True),
        ("Centre-2 m AP", "Car_center_AP_R40@2.0m", "{:.1f}", True),
        ("Top-down IoU 0.5 AP", "Car_bev_AP_R40@0.5", "{:.1f}", True),
        ("Precision  (score ≥ 0.3, IoU 0.5 match)", "precision", "{:.2f}", True),
        ("Recall  (score ≥ 0.3, IoU 0.5 match)", "recall", "{:.2f}", True),
        ("F1  (score ≥ 0.3, IoU 0.5 match)", "f1", "{:.2f}", True),
        ("False alarms on empty scans", "background_false_alarm_rate", "{:.1%}", False),
        ("Position error, median", "BEV_position_error_median_m", "{:.2f} m", False),
        ("Heading error, median", "heading_error_median_deg", "{:.1f}°", False),
    ]
    fig = plt.figure(figsize=(13.33, 7.5))
    headline(fig, "Metric summary", "Final checkpoint (epoch 35). Scene counts: 68 (07-25) vs 16 (07-26 holdout).")
    ax = fig.add_axes([0.04, 0.14, 0.92, 0.66])
    ax.axis("off")
    cols = [0.0, 0.52, 0.78]
    ax.text(cols[0], 1.0, "Metric", fontsize=12, color=C["muted"], weight="bold", transform=ax.transAxes)
    for c, key in zip(cols[1:], ("new", "holdout")):
        ax.add_patch(plt.Rectangle((c - 0.005, 0.985), 0.012, 0.03, color=C[key], transform=ax.transAxes))
        ax.text(c + 0.015, 1.0, SETS[key][0], fontsize=12, color=C["ink"], weight="bold", transform=ax.transAxes)
    step = 0.95 / len(rows)
    for i, (label, key, fmt, _) in enumerate(rows):
        y = 0.9 - i * step
        ax.plot([0, 1], [y + step * 0.55] * 2, color=C["grid"], linewidth=0.8, transform=ax.transAxes)
        ax.text(cols[0], y, label, fontsize=14, color=C["ink"], transform=ax.transAxes, va="center")
        for c, k in zip(cols[1:], ("new", "holdout")):
            ax.text(c + 0.015, y, fmt.format(data[k][2][key]), fontsize=15, color=C["ink"],
                    transform=ax.transAxes, va="center")
    fig.savefig(out / "3_metric_summary.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------
# figure 4: example scans
# ------------------------------------------------------------------------------------
def corners(b):
    x, y, _, l, w, _, yaw = b[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    return pts @ np.array([[c, s], [-s, c]]) + [x, y]


def pick_examples(gt, det):
    """Three hits at ~15/30/50 m and three long-range misses, deterministically."""
    by_frame = {str(a["frame_id"]): a for a in det}
    rows = []
    for fid, (b, _, info) in gt.items():
        if not len(b) or fid not in by_frame:
            continue
        a = by_frame[fid]
        pb = np.asarray(a["boxes_lidar"]).reshape(-1, 7)
        ps = np.asarray(a["score"])
        keep = ps >= 0.3
        d = np.hypot(pb[keep, 0] - b[0, 0], pb[keep, 1] - b[0, 1]) if keep.any() else np.array([np.inf])
        rows.append((fid, float(np.hypot(b[0, 0], b[0, 1])), float(d.min()), info))
    hits, misses = [], []
    for target in (15, 30, 50):
        cand = [r for r in rows if r[2] <= 1.0]
        hits.append(min(cand, key=lambda r: abs(r[1] - target)))
    far = sorted((r for r in rows if r[1] >= 40 and r[2] > 1.0), key=lambda r: r[1])
    offset = [r for r in far if r[2] < 4.0]  # detected, but more than 1 m off
    none = [r for r in far if r[2] >= 4.0]  # nothing near it
    for group in (offset, none, none[len(none) // 2:]):
        if group:
            misses.append(group[len(group) // 3])
    return hits, misses


def fig_examples(data, out: Path) -> None:
    gt, det, _ = data["new"]
    root = Path(SETS["new"][2])
    by_frame = {str(a["frame_id"]): a for a in det}
    hits, misses = pick_examples(gt, det)
    fig, axes = plt.subplots(2, 3, figsize=(13.33, 7.5))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.74, bottom=0.13, wspace=0.12, hspace=0.32)
    for ax, (fid, rng, dmin, info), kind in zip(axes.ravel(), hits + misses, ["hit"] * 3 + ["miss"] * 3):
        b = gt[fid][0][0]
        pts = np.load(root / info["point_cloud"]["path"])
        moving = np.abs(pts[:, 4]) >= 2.0
        half = 9.0
        win = (np.abs(pts[:, 0] - b[0]) < half * 1.6) & (np.abs(pts[:, 1] - b[1]) < half)
        ax.scatter(pts[win & ~moving, 0], pts[win & ~moving, 1], s=14, color=C["static"], linewidths=0)
        ax.scatter(pts[win & moving, 0], pts[win & moving, 1], s=22, color=C["ink2"], linewidths=0)
        ax.add_patch(Polygon(corners(b), closed=True, fill=False, edgecolor=C["ink"], linewidth=2,
                             linestyle=(0, (4, 2.5))))
        a = by_frame[fid]
        pb = np.asarray(a["boxes_lidar"]).reshape(-1, 7)
        for p, s in zip(pb, np.asarray(a["score"])):
            if s < 0.3 or abs(p[0] - b[0]) > half * 1.6 or abs(p[1] - b[1]) > half:
                continue
            ax.add_patch(Polygon(corners(p), closed=True, fill=False, edgecolor=C["new"], linewidth=2.2))
            ax.text(p[0], p[1] + p[4] / 2 + 0.6, f"{s:.2f}", color=C["new"], fontsize=10.5, ha="center",
                    weight="bold")
        ax.set_xlim(b[0] - half * 1.6, b[0] + half * 1.6)
        ax.set_ylim(b[1] - half, b[1] + half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(C["grid"])
        verdict = ("Found" if kind == "hit" else
                   ("Found, but >1 m off" if dmin < 4.0 else "Missed"))
        ax.set_title(f"{verdict}  ·  {rng:.0f} m behind", fontsize=13, loc="left",
                     color=C["ink"], weight="bold")
        ax.text(0.98, 0.03, f"{int(gt[fid][1][0])} returns in box   ·   ego car →", transform=ax.transAxes,
                fontsize=9.5, color=C["muted"], ha="right")
    handles = [
        plt.Line2D([], [], color=C["ink"], linewidth=2, linestyle=(0, (4, 2.5)), label="Ground truth"),
        plt.Line2D([], [], color=C["new"], linewidth=2.2, label="Detection (score shown)"),
        plt.Line2D([], [], marker="o", linestyle="", color=C["ink2"], markersize=6, label="Moving radar return"),
        plt.Line2D([], [], marker="o", linestyle="", color=C["static"], markersize=6, label="Static return (clutter)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.035, 0.855), ncol=4, fontsize=11.5)
    headline(fig, "Example scans from the new Laguna session",
             "Top row: detections at close, mid and long range. Bottom row: long-range failures. Top-down view, 18 m tall windows.")
    fig.savefig(out / "4_example_scans.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/laguna_0725")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    data = {key: load(key) for key in SETS}
    fig_distance(data, out)
    fig_pr(data, out)
    fig_table(data, out)
    fig_examples(data, out)
    print(f"wrote {sorted(p.name for p in out.glob('*.png'))} to {out}")


if __name__ == "__main__":
    main()
