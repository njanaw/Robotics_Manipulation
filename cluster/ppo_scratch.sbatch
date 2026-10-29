#!/bin/bash
# PPO from scratch (3): array tasks 1-3 are the dense-reward seeds, task 4 is the one-seed
# sparse-reward sanity check. Resubmitting the same array resumes each run from its newest
# checkpoint, so time limits only cost the iterations since the last save.
#     sbatch --array=1-4 cluster/ppo_scratch.sbatch
#SBATCH --job-name=ppo-scratch
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
source cluster/env.sh

if [ "$SLURM_ARRAY_TASK_ID" -le 3 ]; then
    SEED=$SLURM_ARRAY_TASK_ID
    REWARD=normalized_dense
else
    SEED=1
    REWARD=sparse
fi
RUN=ppo_scratch_${REWARD}_seed${SEED}

if [ -f runs/"$RUN"/final_ckpt.pt ]; then
    echo "$RUN already finished"
    exit 0
fi
FLAGS=(--seed "$SEED" --reward-mode "$REWARD" --exp-name "$RUN" --no-capture-video)
LATEST=$(ls -v runs/"$RUN"/ckpt_*.pt 2>/dev/null | tail -n 1 || true)
if [ -n "$LATEST" ]; then
    echo "resuming $RUN from $LATEST"
    FLAGS+=(--checkpoint "$LATEST")
fi
python ppo.py "${FLAGS[@]}"
