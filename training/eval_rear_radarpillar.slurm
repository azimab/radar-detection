#!/usr/bin/env bash
#SBATCH --job-name=rear_radarpillar_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_2080ti
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/rear_radarpillar_eval-%j.out
#SBATCH --error=logs/rear_radarpillar_eval-%j.err

set -euxo pipefail

PIPELINE_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}  # submit from training/
RADARPILLAR_ROOT=${RADARPILLAR_ROOT:-${PIPELINE_ROOT}/vendor/RadarPillar}
# --export=NONE drops the submitting shell's variables; the builder writes data_nuscenes*/
# into the repository root, so that is the default (export it in ~/.bashrc to override)
export RADAR_DATA_ROOT=${RADAR_DATA_ROOT:-$(dirname "${PIPELINE_ROOT}")}
NATIVE_DEPS=${RADARPILLAR_ROOT}/.deps/native

# Usage: sbatch eval_rear_radarpillar.slurm [RUN_TAG] [BATCH_SIZE] [CFG] [SPLIT] [DATA_PATH]
#   RUN_TAG is the training extra_tag, so for a seeded run include the suffix,
#   e.g. rear_radarpillar_v3_seed1.
#   CFG: model cfg name under tools/cfgs/rear_models/ (default rear_radarpillar).
#   SPLIT: splits.json key to evaluate on. Default: the cfg's own DATA_SPLIT.test.
#     Anything else overrides DATA_SPLIT.test and writes to eval tag <SPLIT>_sweep, so
#     it never mixes with the cfg's own val_sweep.
#   DATA_PATH: build to score on: a folder name under RADAR_DATA_ROOT or an absolute path.
#     Default: the build the run trained on (its dataset.txt, written by the training
#     script), else the cfg's own DATA_PATH (runs trained before dataset.txt existed).
RUN_TAG=${1:-rear_radarpillar_v1}
BATCH_SIZE=${2:-4}
CFG_NAME=$(basename "${3:-rear_radarpillar}" .yaml)
SPLIT=${4:-}
DATA_PATH_OVERRIDE=${5:-}

module purge
module load gcc
module load cuda/13.0.0

export PYTHONPATH=${NATIVE_DEPS}:${RADARPILLAR_ROOT}
export TORCH_CUDA_ARCH_LIST=7.5


cd "${RADARPILLAR_ROOT}"

CFG_SPLIT=$(python3 -c "import sys; from pcdet.config import cfg, cfg_from_yaml_file; import os; os.chdir('tools'); cfg_from_yaml_file(sys.argv[1], cfg); print(cfg.DATA_CONFIG.DATA_SPLIT['test'])" "cfgs/rear_models/${CFG_NAME}.yaml")
DATA_PATH=$(python3 -c "import sys, os; from pcdet.config import cfg, cfg_from_yaml_file; os.chdir('tools'); cfg_from_yaml_file(sys.argv[1], cfg); print(os.path.expandvars(cfg.DATA_CONFIG.DATA_PATH))" "cfgs/rear_models/${CFG_NAME}.yaml")
EVAL_TAG=val_sweep
SET_ARGS=()
if [ -n "${SPLIT}" ] && [ "${SPLIT}" != "${CFG_SPLIT}" ]; then
    EVAL_TAG="${SPLIT}_sweep"
    SET_ARGS+=(DATA_CONFIG.DATA_SPLIT.test "${SPLIT}")
fi
RUN_DIR=${RADARPILLAR_ROOT}/output/rear_models/${CFG_NAME}/${RUN_TAG}
if [ -z "${DATA_PATH_OVERRIDE}" ] && [ -f "${RUN_DIR}/dataset.txt" ]; then
    DATA_PATH_OVERRIDE=$(cat "${RUN_DIR}/dataset.txt")
elif [ -n "${DATA_PATH_OVERRIDE}" ] && [ "${DATA_PATH_OVERRIDE#/}" = "${DATA_PATH_OVERRIDE}" ]; then
    DATA_PATH_OVERRIDE=${RADAR_DATA_ROOT}/${DATA_PATH_OVERRIDE}
fi
if [ -n "${DATA_PATH_OVERRIDE}" ]; then
    DATA_PATH="${DATA_PATH_OVERRIDE}"
    SET_ARGS+=(DATA_CONFIG.DATA_PATH "${DATA_PATH}")
fi
SPLIT=${SPLIT:-${CFG_SPLIT}}

echo "=================== verifying the rotated-IoU fallback kernel ==================="
python3 "${PIPELINE_ROOT}/tools_eval/verify_rotate_iou.py"

echo "=================== evaluating all checkpoints on ${SPLIT} (${DATA_PATH}) ==================="
cd tools
python3 test.py \
    --cfg_file "cfgs/rear_models/${CFG_NAME}.yaml" \
    --batch_size "${BATCH_SIZE}" \
    --workers 4 \
    --extra_tag "${RUN_TAG}" \
    --eval_all \
    --max_waiting_mins 1 \
    --eval_tag "${EVAL_TAG}" \
    ${SET_ARGS[@]+--set "${SET_ARGS[@]}"}

echo "=================== summarising sweep ==================="
python3 "${PIPELINE_ROOT}/tools_eval/summarize_eval.py" \
    --run_dir "${RUN_DIR}" \
    --eval_tag "${EVAL_TAG}" \
    --split "${SPLIT}" \
    --data_path "${DATA_PATH}"

echo "=================== DONE ==================="
