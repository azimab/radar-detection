#!/usr/bin/env python3
"""Install the rear-radar dataset, evaluator and configs into an OpenPCDet checkout.

Small and idempotent. It copies the ``pcdet/datasets/radar`` package and its
config files, and adds two lines to the dataset registry. It touches nothing
else -- no datasets, no checkpoints, no existing user changes.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

IMPORT_LINE = "from .radar.radar_dataset import RadarDataset\n"
REGISTRY_LINE = "    'RadarDataset': RadarDataset,\n"


RADAR_PACKAGE = ["__init__.py", "radar_dataset.py", "radar_eval.py", "rotate_iou_torch.py", "radar_augment.py"]
ARCHIVE = Path(__file__).resolve().parents[1] / "notes" / "configs_archive"


def plan(repo: Path, source: Path, archive: bool = False) -> dict[Path, Path]:
    """Every file to copy: the radar package, then every config in cfgs/datasets/ and
    cfgs/models/ (no list to keep in sync). With ``archive``, also the retired v1-v5
    configs in notes/configs_archive/, sorted into the same two folders."""
    package = repo / "pcdet" / "datasets" / "radar"
    datasets = repo / "tools" / "cfgs" / "dataset_configs"
    models = repo / "tools" / "cfgs" / "rear_models"
    targets = {source / "radar" / name: package / name for name in RADAR_PACKAGE}
    for cfg in sorted((source / "cfgs" / "datasets").glob("*.yaml")):
        targets[cfg] = datasets / cfg.name
    for cfg in sorted((source / "cfgs" / "models").glob("*.yaml")):
        targets[cfg] = models / cfg.name
    if archive:
        for cfg in sorted(ARCHIVE.glob("rear_radar*.yaml")):
            is_model = cfg.name.startswith("rear_radarpillar")
            targets[cfg] = (models if is_model else datasets) / cfg.name
    clash = [dst for dst in targets.values() if list(targets.values()).count(dst) > 1]
    if clash:
        raise RuntimeError(f"two configs install to the same place: {sorted(set(clash))}")
    return targets


def patched_registry(text: str, registry: Path) -> str:
    updated = text
    if IMPORT_LINE not in updated:
        anchor = "from .dataset import DatasetTemplate\n"
        if anchor not in updated:
            raise RuntimeError(f"registry import anchor not found in {registry}")
        updated = updated.replace(anchor, anchor + IMPORT_LINE, 1)
    if REGISTRY_LINE not in updated:
        anchor = "    'DatasetTemplate': DatasetTemplate,\n"
        if anchor not in updated:
            raise RuntimeError(f"registry mapping anchor not found in {registry}")
        updated = updated.replace(anchor, anchor + REGISTRY_LINE, 1)
    return updated


def install(repo: Path, source: Path, dry_run: bool, archive: bool = False) -> None:
    registry = repo / "pcdet" / "datasets" / "__init__.py"
    if not registry.is_file():
        raise FileNotFoundError(f"not an OpenPCDet checkout: {registry} is missing")
    targets = plan(repo, source, archive)
    for src in targets:
        if not src.is_file():
            raise FileNotFoundError(src)
    text = registry.read_text()
    updated = patched_registry(text, registry)

    print(f"repo: {repo}")
    for src, dst in targets.items():
        print(f"copy: {src} -> {dst}")
    print(
        f"patch registry: {registry}" if updated != text
        else f"registry already configured: {registry}"
    )
    if dry_run:
        return
    for src, dst in targets.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if updated != text:
        registry.write_text(updated)
    # pcdet/__init__.py imports pcdet/version.py, which upstream's full setup.py writes;
    # we only build the one CUDA op (setup_iou3d_nms.py), so write it here
    version = repo / "pcdet" / "version.py"
    if not version.is_file():
        version.write_text('__version__ = "0.3.0+radarpillar"\n')
        print(f"wrote {version}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path,
        default=Path(os.environ.get("RADARPILLAR_ROOT", Path(__file__).parent / "vendor" / "RadarPillar")),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--archive", action="store_true",
                        help="also install the retired v1-v5 configs from notes/configs_archive/")
    args = parser.parse_args()
    install(args.repo.resolve(), Path(__file__).parent / "openpcdet_extension", args.dry_run, args.archive)


if __name__ == "__main__":
    main()
