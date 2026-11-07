#!/usr/bin/env bash
set -euxo pipefail

PIPELINE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
RADARPILLAR_ROOT=${RADARPILLAR_ROOT:-${PIPELINE_ROOT}/vendor/RadarPillar}

RUN_TAG=${1:-smoke}
EPOCHS=${2:-1}
BATCH_SIZE=${3:-4}
# Optional. Empty SEED keeps the old behaviour: train.py's fixed seed 666 and
# extra_tag = RUN_TAG. With a SEED, extra_tag = ${RUN_TAG}_seed${SEED}.
SEED=${4:-}
# Model cfg name under tools/cfgs/rear_models/ (with or without .yaml).
CFG_NAME=$(basename "${5:-rear_radarpillar}" .yaml)
# Optional dataset build: a folder name under RADAR_DATA_ROOT (default: the repository
# root), or an absolute path. Empty = the build config/pipeline.yaml's dataset.output_root
# names, i.e. the one build_nuscenes_dataset.py label just wrote. It always overrides the
# cfg's DATA_PATH, so a new build never needs a new cfg.
DATASET=${6:-}
REPO_ROOT=$(dirname "${PIPELINE_ROOT}")
CFG_FILE=cfgs/rear_models/${CFG_NAME}.yaml

if [[ -n "${SEED}" ]]; then
    [[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer, got '${SEED}'" >&2; exit 2; }
    EXTRA_TAG=${RUN_TAG}_seed${SEED}
else
    EXTRA_TAG=${RUN_TAG}
fi

cd "${RADARPILLAR_ROOT}"

if [[ -z "${DATASET}" ]]; then
    DATASET_ROOT=$(python3 "${REPO_ROOT}/repo_config.py" output-root)
elif [[ "${DATASET}" = /* ]]; then
    DATASET_ROOT=${DATASET}
else
    DATASET_ROOT=${RADAR_DATA_ROOT:-${REPO_ROOT}}/${DATASET}
fi
[[ -n "${DATASET_ROOT}" ]] || { echo "could not resolve the dataset to train on" >&2; exit 2; }
echo "dataset: ${DATASET_ROOT}"

# Record which build this run trained on; eval_rear_radarpillar.slurm reads it back, so a
# run is scored on its own build even after pipeline.yaml moves on. A run tag already used
# for a different build is refused rather than mixing two datasets' checkpoints.
RUN_DIR=${RUN_OUTPUT_ROOT:-${RADARPILLAR_ROOT}/output}/rear_models/${CFG_NAME}/${EXTRA_TAG}
if [[ -f "${RUN_DIR}/dataset.txt" && "$(cat "${RUN_DIR}/dataset.txt")" != "${DATASET_ROOT}" ]]; then
    echo "${RUN_DIR} was trained on $(cat "${RUN_DIR}/dataset.txt"); pick a new run tag" >&2
    exit 2
fi

DATASET_ROOT="${DATASET_ROOT}" python3 - <<'PY'
import json
import os
from pathlib import Path

import torch

root = Path(os.environ["DATASET_ROOT"])
report = json.loads((root / "validation_report.json").read_text())
if not report.get("passed"):
    raise SystemExit(f"{root}/validation_report.json is not passing: {report['errors'][:3]}")

counts = {k: v["samples"] for k, v in report["splits"].items()}
if not counts.get("train") or not counts.get("val"):
    raise SystemExit(f"train/val must both be non-empty, got {counts}")
print("frames per split:", counts)
if not report.get("splits_are_run_blocked"):
    print("WARNING: splits share a source run; val is not independent of train")

# The near-edge gate measures where the opponent's illuminated face falls inside its
# own label. Training on boxes that fail it is a deliberate choice, not a default.
gate = report.get("near_edge_gate", {})
failed = {r: v for r, v in gate.get("by_run", {}).items() if not v["passes"]}
for run, value in gate.get("by_run", {}).items():
    print(f"near-edge {run}: measured {value['median_leading_edge_m']:+.2f} m, "
          f"expected {gate['expected_leading_edge_m']:+.2f} m, "
          f"shortfall {value['shortfall_m']:+.2f} m, "
          f"{'PASS' if value['passes'] else 'FAIL'}")
if failed and os.environ.get("ALLOW_DISPLACED_LABELS") != "1":
    raise SystemExit(
        f"near-edge gate failed for {sorted(failed)}. GT boxes are displaced along the "
        "target heading; measure box_center_forward_offset_m per run and rebuild. "
        "To train on them anyway, set ALLOW_DISPLACED_LABELS=1."
    )

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the allocated job")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch/CUDA:", torch.__version__, torch.version.cuda)
PY

# Seeds submitted together would all compile into the same build/ at once and corrupt it,
# so only one job builds; the others wait on the lock and then find the .so.
if ! compgen -G 'pcdet/ops/iou3d_nms/iou3d_nms_cuda*.so' >/dev/null; then
    (
        flock 9
        if ! compgen -G 'pcdet/ops/iou3d_nms/iou3d_nms_cuda*.so' >/dev/null; then
            rm -rf build
            python3 setup_iou3d_nms.py build_ext --inplace
        fi
    ) 9>.iou3d_build.lock
fi

mkdir -p "${RUN_DIR}"
echo "${DATASET_ROOT}" > "${RUN_DIR}/dataset.txt"

cd tools
test -f "${CFG_FILE}"
TRAIN_ARGS=(
    --cfg_file "${CFG_FILE}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --workers 4
    --extra_tag "${EXTRA_TAG}"
    --ckpt_save_interval 1
    --max_ckpt_save_num "${EPOCHS}"
    --skip_eval
    --fix_random_seed
)
# --set takes the rest of the command line (argparse REMAINDER), so it always goes last.
SET_ARGS=(--set DATA_CONFIG.DATA_PATH "${DATASET_ROOT}")

if [[ -z "${SEED}" ]]; then
    python3 train.py "${TRAIN_ARGS[@]}" "${SET_ARGS[@]}"
elif grep -q "'--seed'" train.py; then
    # train.py has native seed support (the proposed vendor patch).
    python3 train.py "${TRAIN_ARGS[@]}" --seed "${SEED}" "${SET_ARGS[@]}"
else
    # Stock train.py hard-codes common_utils.set_random_seed(666). Rebind that
    # function so the call uses SEED instead, then run train.py unmodified as
    # __main__. train.py looks the function up on the module at call time, so the
    # rebinding takes effect. Everything else about the run is identical.
    SEED="${SEED}" python3 - "${TRAIN_ARGS[@]}" "${SET_ARGS[@]}" <<'PY'
import os
import runpy
import sys

from pcdet.utils import common_utils

seed = int(os.environ["SEED"])
_original = common_utils.set_random_seed


def _seeded(_hard_coded):
    print(f"[run_rear_radarpillar_native] set_random_seed({_hard_coded}) -> {seed}", flush=True)
    _original(seed)


common_utils.set_random_seed = _seeded
sys.argv = ["train.py"] + sys.argv[1:]
sys.path.insert(0, os.getcwd())
runpy.run_path("train.py", run_name="__main__")
PY
fi
