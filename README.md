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
uv run python ppo.py --base hold                               # clean residual on the stand-in base
uv run python ppo.py --base hold --hole-offset-range 0 0.004   # noise-trained residual (4')
uv run python ppo.py                                           # PPO from scratch (3)
uv run python ppo.py --base hold --checkpoint runs/<run>/ckpt_<i>.pt --exp-name <run>  # resume
```

Local smoke run: add `--sim-backend physx_cpu --num-envs 1 --num-eval-envs 1 --no-capture-video`.
