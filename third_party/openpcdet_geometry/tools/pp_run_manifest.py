#!/usr/bin/env python3
"""Map KITTI frame indices back to the run and track they came from.

`build_pp_kitti.py` numbers frames globally and sequentially -- `build_one_run` is
called with `frame_start = totals["out"]`, so each run owns one contiguous index
range, in run-config order.  Nothing recorded those ranges, which means a frame id
alone cannot answer "which track is this?".  That question is needed twice: to sample
frames per run for rendering, and to break evaluation down per track instead of
reporting one number dominated by whichever run is largest.

Going forward `build_pp_kitti.py` writes the manifest itself.  This tool reconstructs
it for a dataset that was already built, by parsing the build job's stdout, and is the
fallback if a manifest is ever lost.

    python3 pp_run_manifest.py --build-log ../logs/pp_multitrack_run1-6527900.out \
        --data-root ../data/kitti_multitrack

The reconstruction is checked against the dataset before it is written: the summed
frame count must match the number of velodyne files, and every id in ImageSets must
fall inside exactly one run's range.  A manifest that disagrees with the data on disk
is worse than no manifest, because everything downstream would silently mislabel.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Track is not recorded anywhere either; it is only ever encoded in the run name.
# Matched longest-prefix-first so `pp_` cannot shadow a longer name.
TRACK_PREFIXES = [
    ("pp_", "putnam"),
    ("caltech_", "putnam"),  # Caltech was the opponent at Putnam Park, not a track
    ("iu_", "putnam"),  # likewise IU
    ("ims_", "ims"),
    ("lvms_", "lvms"),
    ("laguna_", "laguna"),
]

RUN_RE = re.compile(r"^\[run: (?P<name>\S+)\]\s*$")
WRITTEN_RE = re.compile(
    r"^\s*frames_written:\s*(?P<total>\d+)\s*"
    r"\((?P<train>\d+) train, (?P<val>\d+) val, (?P<test>\d+) test\)\s*$"
)


def track_for(run_name: str) -> str:
    for prefix, track in sorted(TRACK_PREFIXES, key=lambda kv: -len(kv[0])):
        if run_name.startswith(prefix):
            return track
    raise ValueError(
        f"cannot derive a track for run {run_name!r}; add its prefix to TRACK_PREFIXES"
    )


def parse_build_log(path: Path) -> list[dict]:
    """Pull (run name, frames written) pairs out of a build log, in order.

    The log interleaves a `[run: NAME]` banner with a later `frames_written:` line for
    the same run.  A run banner with no matching count means the build did not finish
    that run, which invalidates every subsequent range -- so that is a hard failure
    rather than a skip.
    """
    runs: list[dict] = []
    pending: str | None = None

    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        run_match = RUN_RE.match(line)
        if run_match:
            if pending is not None:
                raise ValueError(
                    f"{path}:{lineno}: run {run_match['name']!r} starts before "
                    f"{pending!r} reported frames_written; the build was incomplete"
                )
            pending = run_match["name"]
            continue

        written = WRITTEN_RE.match(line)
        if written and pending is not None:
            runs.append(
                {
                    "run": pending,
                    "track": track_for(pending),
                    "frames": int(written["total"]),
                    "train": int(written["train"]),
                    "val": int(written["val"]),
                    "test": int(written["test"]),
                }
            )
            pending = None

    if pending is not None:
        raise ValueError(f"{path}: run {pending!r} never reported frames_written")
    if not runs:
        raise ValueError(f"{path}: no runs found; is this a data-stage build log?")
    return runs


def assign_ranges(runs: list[dict]) -> list[dict]:
    """Turn per-run counts into the global [start, end] index ranges the builder used."""
    cursor = 0
    for run in runs:
        run["frame_start"] = cursor
        run["frame_end"] = cursor + run["frames"] - 1  # inclusive
        cursor += run["frames"]
    return runs


def verify(runs: list[dict], data_root: Path) -> dict:
    """Check the reconstruction against the dataset before trusting it."""
    total = sum(r["frames"] for r in runs)

    on_disk = sum(
        1
        for split in ("training", "testing")
        for _ in (data_root / split / "velodyne").glob("*.bin")
        if (data_root / split / "velodyne").is_dir()
    )
    if on_disk != total:
        raise ValueError(
            f"build log accounts for {total} frames but {on_disk} velodyne files "
            f"exist under {data_root}; the log does not describe this dataset"
        )

    # Every id in every split must land in exactly one run.  Ranges are contiguous by
    # construction, so this really checks that the highest id is covered -- which is
    # the failure mode if a run were missing from the log.
    counts: dict[str, dict[str, int]] = {r["run"]: {} for r in runs}
    for split in ("train", "val", "test"):
        ids_path = data_root / "ImageSets" / f"{split}.txt"
        if not ids_path.exists():
            continue
        for line in ids_path.read_text().split():
            idx = int(line)
            for run in runs:
                if run["frame_start"] <= idx <= run["frame_end"]:
                    counts[run["run"]][split] = counts[run["run"]].get(split, 0) + 1
                    break
            else:
                raise ValueError(
                    f"frame {line} in {split}.txt falls outside every run range"
                )

    # Cross-check against the per-run split counts the build log reported.  If these
    # disagree, the ranges are shifted and the manifest would mislabel silently.
    for run in runs:
        for split in ("train", "val", "test"):
            expected = run[split]
            actual = counts[run["run"]].get(split, 0)
            if expected != actual:
                raise ValueError(
                    f"run {run['run']}: build log says {expected} {split} frames, "
                    f"but {actual} ids in that range appear in {split}.txt"
                )

    return {"frames_total": total, "velodyne_files": on_disk}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-log", type=Path, required=True,
                        help="stdout of the data-stage build job")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="KITTI dataset root the log describes")
    parser.add_argument("--out", type=Path, default=None,
                        help="manifest path (default: <data-root>/run_manifest.json)")
    args = parser.parse_args()

    runs = assign_ranges(parse_build_log(args.build_log))
    stats = verify(runs, args.data_root)

    out = args.out or args.data_root / "run_manifest.json"
    manifest = {
        "source": "reconstructed from build log",
        "build_log": str(args.build_log),
        "data_root": str(args.data_root),
        "frames_total": stats["frames_total"],
        "runs": runs,
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n")

    width = max(len(r["run"]) for r in runs)
    print(f"verified {stats['frames_total']} frames against {args.data_root}")
    for run in runs:
        print(
            f"  {run['run']:<{width}}  {run['track']:<7}  "
            f"{run['frame_start']:06d}-{run['frame_end']:06d}  "
            f"{run['frames']:>6} frames"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
