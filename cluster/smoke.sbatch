#!/bin/bash
# GPU smoke test: simulator throughput, then the classical controller through the shared
# eval script. Submit from the repo root: sbatch cluster/smoke.sbatch
#SBATCH --job-name=peg-smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source cluster/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/gpu_smoke.py --num-envs 1024
python evaluate.py --policy classical --starts data/interim_test.pt \
    --num-envs 50 --sim-backend physx_cuda --noise-mm 0 2 4 6 \
    --out results/classical_v1_interim50_gpu.json
