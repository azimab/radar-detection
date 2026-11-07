"""Load config/pipeline.yaml and the per-run files it lists, with machine-specific paths
left out of the files.

``${VAR}`` is expanded from the environment; an unset variable is an error, so a
missing setting fails loudly instead of building against a wrong path. ``${VAR:-default}``
falls back to ``default`` when VAR is unset. The path settings below may be relative,
and are then taken relative to this repository.

``runs:`` in pipeline.yaml is a list of run names; each is read from
``config/runs/<name>.yaml`` (``runs_dir`` overrides the folder). A run file's optional
``multitrack:`` block defines a run the shared lidar run config lacks; those blocks are
collected into ``config["openpcdet"]["extra_runs"]`` for the builder's private copy.

From the shell (the Slurm scripts use this)::

    python3 repo_config.py runs          # the configured run names, one per line
    python3 repo_config.py output-root   # absolute path of the dataset build
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "pipeline.yaml"

# (section, key) of every filesystem path the pipeline writes or reads from the config
PATH_KEYS = [
    ("dataset", "output_root"),
    ("dataset", "carry_splits_from"),
    ("dataset", "scan_cache_root"),
    ("dataset", "av24_urdf"),
    ("openpcdet", "repo"),
    ("openpcdet", "local_cache_root"),
]

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def sub(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            if name not in os.environ:
                if default is not None:
                    return default
                raise KeyError(f"config uses ${{{name}}} but the environment variable is not set")
            return os.environ[name]
        return os.path.expanduser(_VAR.sub(sub, value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _multitrack_entry(run: dict[str, Any], block: dict[str, Any], source: Path) -> dict[str, Any]:
    """A run file's ``multitrack:`` block as a shared-run-config entry.

    Fills the two fields the shared parser requires but this builder never reads:
    ``fused_lidar`` (defaults to ego_odom) and each opponent's ``time_offset_seconds``
    (the run's own opponent_time_offset_seconds is what applies).
    """
    if not isinstance(block, dict):
        raise ValueError(f"{source}: multitrack must be a mapping")
    entry = {"name": run["multitrack_run"], **block}
    if entry["name"] != run["multitrack_run"]:
        raise ValueError(f"{source}: multitrack.name {entry['name']!r} differs from multitrack_run")
    entry.setdefault("fused_lidar", list(entry.get("ego_odom") or []))
    for opponent in entry.get("opponents") or []:
        opponent.setdefault("time_offset_seconds", 0.0)
    return entry


def load_runs(config: dict[str, Any], names: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run entries in the listed order, plus the multitrack blocks they define."""
    runs_dir = Path(config.get("runs_dir") or "config/runs")
    if not runs_dir.is_absolute():
        runs_dir = HERE / runs_dir
    runs, extra = [], []
    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"runs: expected run names (config/runs/<name>.yaml), got {name!r}")
        source = runs_dir / f"{name}.yaml"
        if not source.is_file():
            raise FileNotFoundError(f"run {name!r} is listed in runs: but {source} does not exist")
        run = _expand(yaml.safe_load(source.read_text()))
        if run.get("name") != name:
            raise ValueError(f"{source}: name is {run.get('name')!r}; it must match the file name")
        if "multitrack_run" not in run:
            raise ValueError(f"{source}: multitrack_run is required")
        block = run.pop("multitrack", None)
        if block is not None:
            extra.append(_multitrack_entry(run, block, source))
        runs.append(run)
    seen = [run["name"] for run in runs]
    if len(set(seen)) != len(seen):
        raise ValueError(f"runs: listed twice: {sorted({n for n in seen if seen.count(n) > 1})}")
    return runs, extra


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _expand(yaml.safe_load(Path(path).read_text()))
    for section, key in PATH_KEYS:
        value = (config.get(section) or {}).get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            config[section][key] = str(HERE / value)
    config["runs"], extra = load_runs(config, config.get("runs") or [])
    config["openpcdet"]["extra_runs"] = extra
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Print values from the pipeline config.")
    parser.add_argument("what", choices=["runs", "output-root"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.what == "runs":
        print("\n".join(run["name"] for run in config["runs"]))
    else:
        print(config["dataset"]["output_root"])


if __name__ == "__main__":
    main()
