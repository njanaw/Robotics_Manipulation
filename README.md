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
