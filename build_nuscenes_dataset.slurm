#!/usr/bin/env bash
# Decode stage of build_nuscenes_dataset.py: bags -> cache/scans/<run>.npz. The bags are
# large (Caltech alone is 213 GB), so this belongs on a compute node, not a portal.
#
# Submit from the repository root, one job per run:
#   sbatch build_nuscenes_dataset.slurm <run>                  # one run, e.g. a new one
#   for r in $(python3 repo_config.py runs); do sbatch build_nuscenes_dataset.slurm "$r"; done
#
# Run names are the `runs:` list in config/pipeline.yaml. Decode a run once, and again
# only if its bags or radar_topic change: `label` refuses a cache whose bags differ, and
# calibration changes (offsets, clocks, gates) never need a re-decode.
#
# Then, anywhere (minutes, no bag I/O):
#   python3 build_nuscenes_dataset.py label --overwrite
#SBATCH --job-name=rear_radar_decode
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/decode-%j.out

set -euo pipefail
RUN=${1:?usage: sbatch build_nuscenes_dataset.slurm <run name from config/pipeline.yaml>}
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
python3 build_nuscenes_dataset.py decode --runs "${RUN}"
