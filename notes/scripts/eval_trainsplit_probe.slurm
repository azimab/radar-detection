#!/usr/bin/env bash
#SBATCH --job-name=rear_trainsplit_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_2080ti
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/trainsplit_probe-%j.out
#SBATCH --error=logs/trainsplit_probe-%j.err
# Sanity probe: score checkpoints on the TRAIN split (a model that cannot fit its own
# training data points at a data/label bug, not at generalisation).
#   sbatch eval_trainsplit_probe.slurm <CKPT_DIR>
# CKPT_DIR holds the checkpoint_epoch_*.pth files to score.
set -euxo pipefail
CKPT_DIR=$(realpath "${1:?usage: sbatch eval_trainsplit_probe.slurm <CKPT_DIR>}")
PIPELINE_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}  # submit from training/
RADARPILLAR_ROOT=${RADARPILLAR_ROOT:-${PIPELINE_ROOT}/vendor/RadarPillar}
# --export=NONE drops the submitting shell's variables; the builder writes data_nuscenes*/
# into the repository root, so that is the default (export it in ~/.bashrc to override)
export RADAR_DATA_ROOT=${RADAR_DATA_ROOT:-$(dirname "${PIPELINE_ROOT}")}
module purge; module load gcc; module load cuda/13.0.0
export PYTHONPATH=${RADARPILLAR_ROOT}/.deps/native:${RADARPILLAR_ROOT}
cd "${RADARPILLAR_ROOT}/tools"
python3 test.py \
    --cfg_file cfgs/rear_models/rear_radarpillar_car_trainprobe.yaml \
    --batch_size 4 --workers 4 \
    --extra_tag trainsplit_probe \
    --ckpt_dir "${CKPT_DIR}" \
    --eval_all --max_waiting_mins 1 --eval_tag train_probe
python3 "${PIPELINE_ROOT}/tools_eval/summarize_eval.py" \
    --run_dir "${RADARPILLAR_ROOT}/output/rear_models/rear_radarpillar_car_trainprobe/trainsplit_probe" \
    --eval_tag train_probe --split train \
    --data_path "${RADAR_DATA_ROOT}/data_nuscenes"
echo "=== PROBE DONE ==="
