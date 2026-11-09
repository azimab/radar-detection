#!/usr/bin/env python3
"""Seed a private odometry cache from the shared OpenPCDet checkout.

The rear-radar builder gets its ego/opponent timelines from
``build_pp_kitti.run_timelines``, which reads (and, unless told not to, writes)
an odometry cache under the run config's ``paths.data_root``. That path is a
shared checkout, so this pipeline points ``data_root`` at a local
directory instead and copies the caches it needs into it. Nothing in the shared
checkout is ever written.

One wrinkle needs an explicit decision rather than a silent workaround. The
Laguna cache was built before that run's bags were reorganised,
so its stored fingerprint names the pre-move path::

    cached: .../run3/no_lidar/no_lidar_0.mcap        (no longer exists)
    config: .../run3/rosbags/no_lidar/no_lidar_0.mcap

``pp.source_fingerprint`` hashes ``(path, size, mtime_ns)``, so the moved file
reads as a cache miss and forces a full re-decode of a 9 GB bag on every build.
This script restamps the path component -- and *only* the path component -- and
refuses to do so unless size and mtime_ns are byte-identical between the stored
row and the file the config now names. Identical size and nanosecond mtime after
a rename is a moved file, not a changed one.

    python3 seed_odom_cache.py --dry-run
    python3 seed_odom_cache.py
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from repo_config import DEFAULT_CONFIG, load_config


def stored_rows(path: Path) -> list[list[str]]:
    with np.load(path, allow_pickle=True) as data:
        return [list(map(str, row)) for row in np.asarray(data["source_fingerprint"], dtype=object).tolist()]


def expected_sources(run_config: Path, run_name: str) -> list[Path]:
    raw = yaml.safe_load(run_config.read_text())
    for run in raw["inputs"]["runs"]:
        if run["name"] != run_name:
            continue
        sources = [Path(os.path.expandvars(p)) for p in run.get("ego_odom", [])]
        for opponent in run.get("opponents", []):
            sources += [Path(os.path.expandvars(p)) for p in opponent.get("odom", [])]
        return sources
    raise KeyError(f"{run_name!r} not in {run_config}")


def restamp(cache: Path, run_config: Path, run_name: str, *, dry_run: bool) -> bool:
    rows = stored_rows(cache)
    wanted = expected_sources(run_config, run_name)
    by_basename = {Path(row[0]).name: row for row in rows}
    changed = False
    for source in wanted:
        row = by_basename.get(source.name)
        if row is None:
            raise RuntimeError(f"{cache.name}: no cached row for {source.name}")
        if row[0] == str(source):
            continue
        stat = source.stat()
        if row[1] != str(stat.st_size) or row[2] != str(stat.st_mtime_ns):
            raise RuntimeError(
                f"{cache.name}: {source} differs from the cached source "
                f"(size {row[1]} -> {stat.st_size}, mtime_ns {row[2]} -> {stat.st_mtime_ns}); "
                "this is a changed file, not a moved one -- delete the cache and rebuild it"
            )
        print(f"    restamp path only: {row[0]}\n                    -> {source}")
        print(f"      size={stat.st_size} mtime_ns={stat.st_mtime_ns} identical, so same bytes")
        row[0] = str(source)
        changed = True
    if changed and not dry_run:
        with np.load(cache, allow_pickle=True) as data:
            payload = {key: data[key] for key in data.files}
        payload["source_fingerprint"] = np.asarray(sorted(rows), dtype=object)
        np.savez(cache, **payload)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    repo = Path(config["openpcdet"]["repo"])
    run_config = repo / config["openpcdet"]["run_config"]
    local_root = Path(config["openpcdet"]["local_cache_root"])
    shared_cache = os.path.expandvars(yaml.safe_load(run_config.read_text())["paths"]["data_root"])
    shared_cache = (repo / shared_cache) if not str(shared_cache).startswith("/") else Path(shared_cache)
    shared_cache = shared_cache / "cache"
    local_cache = local_root / "cache"

    print(f"shared cache: {shared_cache}  (read-only to this pipeline)")
    print(f"local cache : {local_cache}")
    if not args.dry_run:
        local_cache.mkdir(parents=True, exist_ok=True)

    shared_runs = {run["name"] for run in yaml.safe_load(run_config.read_text())["inputs"]["runs"]}
    for run in config["runs"]:
        name = run["multitrack_run"]
        source = shared_cache / f"{name}_odom.npz"
        target = local_cache / f"{name}_odom.npz"
        if name not in shared_runs:
            # a run defined by a multitrack: block in config/runs: the shared checkout never decoded it, so `decode`
            # builds this cache itself on its first (cache-miss) pass
            print(f"\n  {name}\n    not in the shared run config; decode will build {target.name}")
            continue
        if not source.is_file():
            raise SystemExit(f"missing shared cache {source}")
        print(f"\n  {name}")
        print(f"    copy {source} -> {target}")
        if args.dry_run:
            rows = stored_rows(source)
            for wanted in expected_sources(run_config, name):
                row = {Path(r[0]).name: r for r in rows}.get(wanted.name)
                if row and row[0] != str(wanted):
                    print(f"    would restamp: {row[0]} -> {wanted}")
            continue
        shutil.copy2(source, target)
        if restamp(target, run_config, name, dry_run=False):
            print("    restamped")
        else:
            print("    fingerprint already matches")
    print("\ndone" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
