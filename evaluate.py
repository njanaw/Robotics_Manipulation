"""Shared evaluation: run a policy from saved start states under the hole pose offset sweep.

    python evaluate.py --policy classical --starts data/test_states.pt \
        --noise-mm 0 2 4 6 --num-envs 100 --out results/classical.json

A policy is any object with
    reset()       called at the start of each batch of episodes
    act(obs)      obs is the state_dict observation (batched); returns (num_envs, 7) actions
                  for pd_ee_delta_pose (normalized, gripper last)
--policy is "classical" or "module:factory", where factory(num_envs=..., env=..., **policy_kwargs)
builds the policy (e.g. a Diffusion Policy or residual adapter that loads a checkpoint).
mani_skill.utils.common.flatten_state_dict(obs, use_torch=True) gives exactly the obs_mode="state"
vector the learned policies were trained on.

Each noise magnitude m shifts the observed hole pose by exactly m in the hole plane, in a direction
drawn once per episode. Directions depend only on (start index, m, --seed), so every controller is
scored on identical offsets. Success is reported both ways ManiSkill does: success_once (inserted at
any step, the primary number) and success_at_end.
"""

import argparse
import importlib
import json
import math
import os
import platform
import time
from pathlib import Path

if platform.system() == "Darwin":
    # macOS only: point SAPIEN at Homebrew's MoltenVK (brew install molten-vk vulkan-loader)
    os.environ.setdefault("VK_ICD_FILENAMES", "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
    os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

import gymnasium as gym
import numpy as np
import torch

import noisy_env  # noqa: F401  registers PegInsertionSideNoisy-v1
import start_states

ENV_ID = "PegInsertionSideNoisy-v1"
CONTROL_MODE = "pd_ee_delta_pose"  # shared stiffness controller for every method


def make_env(num_envs=1, sim_backend="auto", obs_mode="state_dict", max_episode_steps=100, **kwargs):
    if sim_backend == "auto":
        sim_backend = "physx_cuda" if torch.cuda.is_available() else "physx_cpu"
    return gym.make(ENV_ID, num_envs=num_envs, obs_mode=obs_mode, control_mode=CONTROL_MODE,
                    sim_backend=sim_backend, max_episode_steps=max_episode_steps, **kwargs)


def sweep_offsets(n, magnitude, seed):
    """(n, 2) hole-plane (y, z) offsets of fixed magnitude in seeded uniformly random directions"""
    theta = np.random.default_rng(seed).uniform(0, 2 * np.pi, size=n)
    return magnitude * np.stack([np.cos(theta), np.sin(theta)], axis=1)


def set_hole_offsets(env, offsets):
    # resets into saved states skip _initialize_episode, so the sweep writes noisy_env's offsets directly
    u = env.unwrapped
    u.hole_offsets[:, 0] = 0
    u.hole_offsets[:, 1:] = torch.as_tensor(offsets, dtype=torch.float32, device=u.device)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def build_policy(spec, num_envs, env, policy_kwargs):
    if spec == "classical":
        from classical import ClassicalController
        return ClassicalController(num_envs, **policy_kwargs)
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)(num_envs=num_envs, env=env, **policy_kwargs)


def run_episodes(env, policy, starts, idx, offsets, max_steps):
    """Run one batch. idx may be shorter than num_envs; it is padded and the padding dropped."""
    u = env.unwrapped
    n_real = len(idx)
    pad = np.resize(idx, u.num_envs)
    start_states.reset_to(env, starts, pad)
    set_hole_offsets(env, np.resize(offsets, (u.num_envs, 2)))
    obs = u.get_obs()
    policy.reset()

    first_success = torch.full((u.num_envs,), -1, dtype=torch.long)
    success = torch.zeros(u.num_envs, dtype=torch.bool)
    for t in range(max_steps):
        action = policy.act(obs).to(u.device)
        obs, _, _, _, info = env.step(action)
        success = info["success"].cpu()
        first_success[(first_success < 0) & success] = t + 1
        if (first_success >= 0).all():
            break
    return dict(success_once=(first_success >= 0)[:n_real].numpy(),
                success_at_end=success[:n_real].numpy(),
                steps_to_success=first_success[:n_real].numpy())


def evaluate(policy_spec, starts_path, noise_mm, num_envs, sim_backend, max_steps, seed,
             limit=None, policy_kwargs=None):
    starts = start_states.load(starts_path)
    n = len(starts["seeds"]) if limit is None else min(limit, len(starts["seeds"]))
    env = make_env(num_envs=num_envs, sim_backend=sim_backend, max_episode_steps=max_steps + 1)
    policy = build_policy(policy_spec, num_envs, env, policy_kwargs or {})

    results = dict(policy=policy_spec, starts=str(starts_path), n=n, max_steps=max_steps,
                   seed=seed, policy_kwargs=policy_kwargs or {}, sweep=[])
    for m in noise_mm:
        offsets = sweep_offsets(len(starts["seeds"]), m / 1000.0, seed=[seed, int(round(m * 1000))])[:n]
        t0 = time.time()
        per_ep = {k: [] for k in ("success_once", "success_at_end", "steps_to_success")}
        for b in range(0, n, num_envs):
            idx = np.arange(b, min(b + num_envs, n))
            out = run_episodes(env, policy, starts, idx, offsets[idx], max_steps)
            for k in per_ep:
                per_ep[k].append(out[k])
        per_ep = {k: np.concatenate(v) for k, v in per_ep.items()}
        k_once = int(per_ep["success_once"].sum())
        steps = per_ep["steps_to_success"][per_ep["success_once"]]
        row = dict(
            noise_mm=m,
            success_once=k_once / n,
            success_once_ci95=wilson(k_once, n),
            success_at_end=float(per_ep["success_at_end"].mean()),
            mean_steps_to_success=float(steps.mean()) if len(steps) else None,
            wall_seconds=time.time() - t0,
            episodes={k: v.tolist() for k, v in per_ep.items()},
        )
        results["sweep"].append(row)
        lo, hi = row["success_once_ci95"]
        print(f"noise {m:g} mm: success {row['success_once']:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"at_end {row['success_at_end']:.3f}  steps {row['mean_steps_to_success']}  "
              f"({row['wall_seconds']:.0f}s)", flush=True)
    env.close()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="classical")
    ap.add_argument("--starts", required=True)
    ap.add_argument("--noise-mm", type=float, nargs="+", default=[0, 2, 4, 6])
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--sim-backend", default="auto")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0, help="seeds the offset directions")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N start states")
    ap.add_argument("--policy-kwargs", default="{}", help="JSON dict passed to the policy factory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = evaluate(args.policy, args.starts, args.noise_mm, args.num_envs, args.sim_backend,
                       args.max_steps, args.seed, args.limit, json.loads(args.policy_kwargs))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
