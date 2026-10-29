"""Cluster smoke test: GPU-parallel PegInsertionSideNoisy-v1 and simulation throughput.

    python scripts/gpu_smoke.py --num-envs 1024
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluate import make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    env = make_env(num_envs=args.num_envs, sim_backend="physx_cuda", obs_mode="state", hole_offset_range=(0.0, 0.004))
    u = env.unwrapped
    obs, _ = env.reset(seed=0)
    print("obs", tuple(obs.shape), "action", env.action_space.shape, "device", obs.device)
    off = u.hole_offsets[:, 1:].norm(dim=1)
    print(f"training hole noise: max {off.max().item() * 1000:.2f} mm, mean {off.mean().item() * 1000:.2f} mm")

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.steps):
        obs, rew, term, trunc, info = env.step(2 * torch.rand(env.action_space.shape, device=u.device) - 1)
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = args.num_envs * args.steps / dt
    print(f"{sps:,.0f} env steps/s -> 75M steps in {75e6 / sps / 3600:.2f} h of pure simulation")
    env.close()


if __name__ == "__main__":
    main()
