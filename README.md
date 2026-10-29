# Robotics_Manipulation
pose error contribution

## Setup

```bash
brew install molten-vk vulkan-loader  # macOS only: ManiSkill always builds a Vulkan renderer
uv sync
uv run pytest tests
```

## Residual PPO

```bash
uv run python ppo.py --base hold                               # clean residual (4), sparse reward by default
uv run python ppo.py --base hold --hole-offset-range 0 0.004   # noise-trained residual (4')
uv run python ppo.py                                           # PPO from scratch (3), dense reward by default
uv run python ppo.py <original flags> --checkpoint runs/<run>/ckpt_<i>.pt --exp-name <run>  # resume: repeat every original flag
```

Local smoke run: add `--sim-backend physx_cpu --num-envs 1 --num-eval-envs 1 --num-minibatches 1 --no-capture-video`.

If the tests fail on macOS with `ErrorIncompatibleDriver` or `failed to find a rendering device`, point SAPIEN at MoltenVK:
`export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`
(`evaluate.py` sets these itself).

## Evaluation and classical controller (Kevin)

```bash
uv run python evaluate.py --policy classical --starts data/interim_test.pt --out results/classical.json
uv run python evaluate.py --policy my_module:make_policy --policy-kwargs '{"checkpoint": "..."}' --starts data/test_states.pt --num-envs 100
```

- `evaluate.py` runs each start state once per hole offset (default 0, 2, 4, 6 mm). It reports `success_once` (primary) and `success_at_end` with Wilson 95% intervals, plus per-episode results in the JSON. Offset directions are seeded per (start index, magnitude), so every method sees identical offsets. A policy needs `reset()` and `act(obs)`, where `obs` is the batched `state_dict`; `common.flatten_state_dict(obs, use_torch=True)` gives the exact 43-dim `obs_mode="state"` vector.
- `classical.py` is controller (1). It aligns the peg 2 cm in front of the observed hole, matching the square peg to the nearest quarter roll. It then pushes along the hole axis under the stiffness controller. On a stall at the rim (no progress for 4 steps) it spiral searches in the hole plane, pausing whenever the peg starts to drop. An integral term cancels gravity sag. It is pure feedback on the observation, so the hole offset reaches it exactly as it reaches the learned methods.
- `ClassicalBase` wraps it in `residual.py`'s base interface (`reset(mask)`, `act(flat_obs)`) to replace `HoldBase`.
- `start_states.py` defines the start-state file. It is a `torch.save` dict holding: `seeds` (N,), `env_states` (stacked `get_state_dict()`), `peg_half_sizes` (N, 3) and `meta`. The seed rebuilds the peg and hole geometry on reconfigure; `reset_to` checks that.
- `data/interim_test.pt` holds 50 stand-in grasped states from `scripts/make_interim_start_states.py`. They come from servoing to the shipped planner's grasp pose, because mplib has no macOS wheels. Sut's planner states replace these for every reported number.

Classical v1 on the 50 stand-in states (CPU sim, 100 steps):

| Hole offset | Success (95% CI) | Mean steps to success |
|---|---|---|
| 0 mm | 1.00 [0.93, 1.00] | 34 |
| 2 mm | 1.00 [0.93, 1.00] | 34 |
| 4 mm | 0.96 [0.87, 0.99] | 46 |
| 6 mm | 0.92 [0.81, 0.97] | 49 |

## Cluster (ORCD Engaging)

```bash
bash cluster/setup_engaging.sh            # once, on the login node, from the repo root
sbatch cluster/smoke.sbatch               # GPU sim throughput + classical eval on the GPU backend
sbatch --array=1-4 cluster/ppo_scratch.sbatch   # PPO from scratch: dense seeds 1-3, sparse seed 1; resubmit to resume
```

On the GPU, ManiSkill's delta-pose controller takes a different IK path from the CPU one. If the smoke job's classical numbers fall well below the table above, check the rotation convention first.
