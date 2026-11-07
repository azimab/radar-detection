#!/usr/bin/env bash
#SBATCH --job-name=rear_radarpillar
#SBATCH --partition=gpu
# Use UVA's documented generic GPU request plus a hardware constraint.
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_2080ti
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/rear_radarpillar-%j.out
#SBATCH --error=logs/rear_radarpillar-%j.err

set -euxo pipefail

PIPELINE_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}  # submit from training/
RADARPILLAR_ROOT=${RADARPILLAR_ROOT:-${PIPELINE_ROOT}/vendor/RadarPillar}
# --export=NONE drops the submitting shell's variables. A bare DATASET name resolves under
# this (default: the repository root, where the builder writes); export it in ~/.bashrc.
export RADAR_DATA_ROOT=${RADAR_DATA_ROOT:-$(dirname "${PIPELINE_ROOT}")}
NATIVE_DEPS=${RADARPILLAR_ROOT}/.deps/native

# Usage: sbatch train_rear_radarpillar.slurm [RUN_TAG] [EPOCHS] [BATCH_SIZE] [SEED] [CFG] [DATASET]
#   SEED:    optional integer. Empty = train.py's fixed seed 666. When set, the
#            output extra_tag becomes ${RUN_TAG}_seed${SEED}.
#   CFG:     model cfg name under tools/cfgs/rear_models/ (default rear_radarpillar).
#   DATASET: build folder name under RADAR_DATA_ROOT, or an absolute path. Default: the
#            build config/pipeline.yaml's dataset.output_root names.
#   Leave an argument empty ("") to keep its default, e.g. ... v6 35 4 1 "" data_nuscenes_X
RUN_TAG=${1:-smoke}
EPOCHS=${2:-1}
BATCH_SIZE=${3:-4}
SEED=${4:-}
CFG_NAME=$(basename "${5:-rear_radarpillar}" .yaml)
DATASET=${6:-}

# The dataset (the cfg's DATA_PATH) is checked by run_rear_radarpillar_native.sh.
test -f "${RADARPILLAR_ROOT}/tools/cfgs/rear_models/${CFG_NAME}.yaml"
mkdir -p "${PIPELINE_ROOT}/logs"

module purge
module load gcc
module load cuda/13.0.0

export PYTHONPATH=${NATIVE_DEPS}:${RADARPILLAR_ROOT}
export TORCH_CUDA_ARCH_LIST=7.5
export MAX_JOBS=${SLURM_CPUS_PER_TASK}

command -v python3
command -v nvcc
python3 --version
nvcc --version

bash "${PIPELINE_ROOT}/run_rear_radarpillar_native.sh" \
    "${RUN_TAG}" "${EPOCHS}" "${BATCH_SIZE}" "${SEED}" "${CFG_NAME}" "${DATASET}"
