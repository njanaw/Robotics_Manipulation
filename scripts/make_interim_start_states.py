"""Stand-in grasped start states for local development (no mplib on macOS).

Uses the shipped planner's grasp pose (same OBB grasp and tail offset as
mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py) but reaches it
with the proportional servo instead of a planned trajectory. Sut's planner-generated
states replace these for every reported number.

    python scripts/make_interim_start_states.py --out data/interim_test.pt --first-seed 100000 -n 50
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sapien
import torch
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb

from evaluate import make_env  # first, so the macOS Vulkan setup runs before SAPIEN loads
import start_states
from classical import servo_action, split_pose

FINGER_LENGTH = 0.025


def servo_to(env, pose, gripper, max_steps, tol=0.003, rot_tol=0.03):
    u = env.unwrapped
    des_p, des_R = split_pose(np.concatenate([pose.p, pose.q]))
    for _ in range(max_steps):
        tcp_p, tcp_R = split_pose(u.agent.tcp.pose.raw_pose[0].cpu().numpy())
        if np.linalg.norm(des_p - tcp_p) < tol and (des_R * tcp_R.inv()).magnitude() < rot_tol:
            return True
        env.step(torch.from_numpy(servo_action(tcp_p, tcp_R, des_p, des_R, gripper))[None])
    return False


def grasp(env, seed):
    env.reset(seed=seed)
    u = env.unwrapped
    obb = get_actor_obb(u.peg)
    target_closing = u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    info = compute_grasp_info_by_obb(obb, approaching=np.array([0, 0, -1]),
                                     target_closing=target_closing, depth=FINGER_LENGTH)
    grasp_pose = u.agent.build_grasp_pose(np.array([0, 0, -1]), info["closing"], info["center"])
    grasp_pose = grasp_pose * sapien.Pose([-max(0.05, u.peg_half_sizes[0, 0].item() / 2 + 0.01), 0, 0])

    if not servo_to(env, grasp_pose * sapien.Pose([0, 0, -0.05]), 1.0, 60):
        return False
    if not servo_to(env, grasp_pose, 1.0, 40, tol=0.002):
        return False
    tcp_p, tcp_R = split_pose(u.agent.tcp.pose.raw_pose[0].cpu().numpy())
    for _ in range(10):  # close while holding the grasp pose
        env.step(torch.from_numpy(servo_action(tcp_p, tcp_R, tcp_p, tcp_R, -1.0))[None])
    return bool(u.agent.is_grasping(u.peg, max_angle=20)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("-n", type=int, default=50)
    args = ap.parse_args()

    env = make_env(num_envs=1, sim_backend="physx_cpu", max_episode_steps=1000)
    u = env.unwrapped
    seeds, states, sizes = [], [], []
    seed = args.first_seed
    while len(seeds) < args.n:
        if grasp(env, seed):
            seeds.append(seed)
            states.append(u.get_state_dict())
            sizes.append(u.peg_half_sizes.cpu().clone())
        else:
            print(f"seed {seed}: grasp failed, skipped")
        seed += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    start_states.save(args.out, seeds, start_states.stack_states(states), torch.cat(sizes),
                      meta=dict(source="interim servo grasp", first_seed=args.first_seed, tried=seed - args.first_seed))
    print(f"saved {len(seeds)} start states ({seed - args.first_seed} seeds tried) to {args.out}")


if __name__ == "__main__":
    main()
