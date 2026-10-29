"""
peg_insertion_planner.py implements a project-local, collision-aware motion-
planning runner for the custom ManiSkill 3 PegInsertion-sutnaw environment.

It reuses ManiSkill's Panda motion-planning solvers while adding custom
insertion logic without modifying installed package files.

The peg spawns at the gripper with a randomized orientation and follows
the TCP through scripted kinematic attachment. MPLib collision checking
accounts for the attached peg, fixture, and table.

To address an upstream segmentation fault in mplib==0.1.1, the project
overrides the dependency to mplib==0.2.1 and adapts incompatible planner
API calls locally.

Note: The attachment is scripted and does not demonstrate a physically
validated grasp using the robot's fingers.
"""
import argparse
import mplib
import numpy as np
import sapien

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.utils.structs.pose import to_sapien_pose

import gymnasium as gym
import argparse

from mani_skill.envs.tasks import PegInsertionSideEnv

from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

# import peg_insertion  # noqa: F401  (registers "PegInsertion-sutnaw")

ENV_ID = "PegInsertionSide-v1" #PegInsertionSide-v1 PegInsertion-sutnaw
# The task registers max_episode_steps=100, which is sized for a learned
# policy's per-step control, not for executing a full multi-stage joint-space
# plan (approach + align + refine + insert) in one episode. gym.make's own
# `max_episode_steps` kwarg overrides the registered TimeLimit wrapper; the
# task definition in peg_insertion.py itself is untouched.
PLANNING_MAX_EPISODE_STEPS = 300


class AttachedPegMotionPlanningSolver(PandaArmMotionPlanningSolver):
    """PandaArmMotionPlanningSolver whose screw motions also account for the
    carried peg's own collision box (attached via mplib's native
    attached-object support), not just the robot's links. Also bridges the
    mplib 0.1.x -> 0.2.1 API break described in the module docstring, for
    exactly the two calls that broke (set_base_pose, plan_screw's pose arg
    and removed kwargs) -- everything else is inherited unchanged.
    """

    def setup_planner(self):
        # Copy of BaseMotionPlanningSolver.setup_planner, with one fix:
        # 0.2.1's set_base_pose wants an mplib.Pose, not a raw [p..., q...]
        # array (that line would raise/misbehave otherwise).
        move_group = self.MOVE_GROUP if hasattr(self, "MOVE_GROUP") else "eef"
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=move_group,
        )
        planner.set_base_pose(mplib.Pose(p=self.base_pose.p, q=self.base_pose.q))
        planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        return planner

    def move_to_pose_with_screw(self, pose, dry_run: bool = False, refine_steps: int = 0):
        pose = to_sapien_pose(pose)
        self._update_grasp_visual(pose)
        pose = self._transform_pose_for_planning(pose)
        # Differs from the base class in two ways, both required for 0.2.1:
        # goal pose is an mplib.Pose (not a raw array), and there is no
        # use_point_cloud/use_attach toggle any more -- whatever point cloud
        # and attached box are currently registered on self.planner (set up
        # once per episode in solve_episode) are always considered.
        goal = mplib.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_screw(
            goal,
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                goal,
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)


def _add_fixture_and_table_obstacles(planner: AttachedPegMotionPlanningSolver, base_env) -> None:
    """Register the fixture's four frame pieces and the table as mplib point-
    cloud obstacles. Neither is part of the robot URDF, so mplib has no other
    way to know about them; `add_box_collision` samples each box's surface.
    """
    # Recompute the exact same four frame-piece boxes peg_insertion.py builds
    # in _load_scene, in the fixture's own local frame, then transform each
    # to world using the fixture's actual (always axis-aligned) pose.
    outer = base_env.fixture_outer_size / 2
    ox, oy = base_env.opening_size
    hz = base_env.fixture_height / 2
    pieces = [
        ([0, (oy / 2 + outer) / 2, hz], [outer, (outer - oy / 2) / 2, hz]),
        ([0, -(oy / 2 + outer) / 2, hz], [outer, (outer - oy / 2) / 2, hz]),
        ([(ox / 2 + outer) / 2, 0, hz], [(outer - ox / 2) / 2, oy / 2, hz]),
        ([-(ox / 2 + outer) / 2, 0, hz], [(outer - ox / 2) / 2, oy / 2, hz]),
    ]
    fixture_pose = to_sapien_pose(base_env.fixture.pose)
    for local_p, half in pieces:
        world_pose = fixture_pose * sapien.Pose(p=local_p)
        extents = np.array([2 * h for h in half])
        planner.add_box_collision(extents, world_pose)

    # Table collision box, reconstructed from TableSceneBuilder's own
    # hard-coded local box (see mani_skill/utils/scene_builder/table/
    # scene_builder.py): half_size=(2.418/2, 1.209/2, 0.9196429/2) at local
    # p=[0,0,0.9196429/2], composed onto the table actor's actual world pose
    # (so any table rotation/translation is handled correctly either way).
    table_actor_pose = to_sapien_pose(base_env.table_scene.table.pose)
    table_local_box_pose = sapien.Pose(p=[0, 0, 0.9196429 / 2])
    table_world_pose = table_actor_pose * table_local_box_pose
    table_extents = np.array([2.418, 1.209, 0.9196429])
    planner.add_box_collision(table_extents, table_world_pose)


def solve_episode(env, seed=0, debug=False, vis=False, stop_after_grasp=False):
    env.reset(seed=seed)
    base_env = env.unwrapped

    assert base_env.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], base_env.control_mode

    # AttachedPegMotionPlanningSolver (not plain PandaArmMotionPlanningSolver):
    # it plans identically here (nothing calls update_attached_box in this
    # function, so the "attached peg" collision feature is simply unused),
    # but it also carries the two mplib 0.1.x -> 0.2.1 API fixes
    # (set_base_pose, plan_screw) this project's mplib==0.2.1 override needs.
    # Plain PandaArmMotionPlanningSolver still calls the old signatures and
    # crashes with TypeError: set_base_pose() incompatible function arguments.
    planner = AttachedPegMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=base_env.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )

    try:
        finger_length = 0.025

        # Find a grasp pose from the built-in peg's oriented bounding box.
        obb = get_actor_obb(base_env.peg)
        approaching = np.array([0, 0, -1])
        target_closing = (
            base_env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1]
            .cpu()
            .numpy()
        )

        peg_init_pose = base_env.peg.pose

        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=finger_length,
        )

        closing = grasp_info["closing"]
        center = grasp_info["center"]
        grasp_pose = base_env.agent.build_grasp_pose(
            approaching, closing, center
        )

        offset = sapien.Pose(
            [
                -max(
                    0.05,
                    base_env.peg_half_sizes[0, 0].item() / 2 + 0.01,
                ),
                0,
                0,
            ]
        )
        grasp_pose = grasp_pose * offset

        # Stage 1: approach the peg.
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        result = planner.move_to_pose_with_screw(reach_pose)
        if result == -1:
            return {"success": False, "failure_reason": "approach planning failed"}

        # Stage 2: move to the grasp pose.
        result = planner.move_to_pose_with_screw(grasp_pose)
        if result == -1:
            return {"success": False, "failure_reason": "grasp planning failed"}

        planner.close_gripper()

        if stop_after_grasp:
            # Nothing above attaches or kinematically pins the peg to the
            # gripper -- PegInsertionSideEnv's peg is a real dynamic rigid
            # body, so "grasped" means actual finger contact/friction, not
            # scripting. Verify that physically rather than assuming
            # close_gripper() alone means the peg is held:
            #   1. is_grasping() right after closing (real contact force).
            #   2. hold qpos for a few steps with the gripper still
            #      commanded closed, then check is_grasping() again and that
            #      the peg hasn't drifted away from the TCP -- a transient
            #      contact right after closing could still slip afterward.
            is_grasping = bool(base_env.agent.is_grasping(base_env.peg).item())

            arm_dof = len(planner.planner.joint_vel_limits)
            hold_qpos = base_env.agent.robot.get_qpos()[0, :arm_dof].cpu().numpy()
            rel_before = (base_env.peg.pose.p - base_env.agent.tcp.pose.p).cpu().numpy().copy()
            for _ in range(10):
                env.step(np.hstack([hold_qpos, planner.CLOSED]))
            rel_after = (base_env.peg.pose.p - base_env.agent.tcp.pose.p).cpu().numpy()
            grasp_drift_m = float(np.linalg.norm(rel_after - rel_before))
            is_grasping_after_hold = bool(base_env.agent.is_grasping(base_env.peg).item())

            peg_pose = base_env.peg.pose
            tcp_pose = base_env.agent.tcp.pose
            peg_p = peg_pose.p[0].cpu().numpy().copy()
            peg_q = peg_pose.q[0].cpu().numpy().copy()
            tcp_p = tcp_pose.p[0].cpu().numpy().copy()
            tcp_q = tcp_pose.q[0].cpu().numpy().copy()
            poses_finite = bool(
                np.isfinite(peg_p).all() and np.isfinite(peg_q).all()
                and np.isfinite(tcp_p).all() and np.isfinite(tcp_q).all()
            )
            # Drift threshold is generous (5mm) -- its purpose is catching a
            # peg that is visibly slipping/falling, not enforcing insertion-
            # grade precision.
            physically_grasped = (
                is_grasping and is_grasping_after_hold
                and grasp_drift_m < 0.005 and poses_finite
            )

            return {
                "success": False,
                "grasp_reached": True,
                "is_grasping": is_grasping,
                "is_grasping_after_hold": is_grasping_after_hold,
                "grasp_drift_m": grasp_drift_m,
                "poses_finite": poses_finite,
                "physically_grasped": physically_grasped,
                "peg_pose": {"p": peg_p, "q": peg_q},
                "tcp_pose": {"p": tcp_p, "q": tcp_q},
                "failure_reason": None if physically_grasped else "grasp did not hold physically",
            }

        # Stage 3: move the grasped peg toward the built-in goal.

        # Stage 3: move the grasped peg toward the built-in goal.
        insert_pose = base_env.goal_pose * peg_init_pose.inv() * grasp_pose
        offset = sapien.Pose(
            [-0.01 - base_env.peg_half_sizes[0, 0].item(), 0, 0]
        )
        pre_insert_pose = insert_pose * offset

        result = planner.move_to_pose_with_screw(pre_insert_pose)
        if result == -1:
            return {"success": False, "failure_reason": "pre-insertion planning failed"}

        # Stage 4: refine the insertion pose.
        for _ in range(3):
            delta_pose = (
                base_env.goal_pose * offset * base_env.peg.pose.inv()
            )
            pre_insert_pose = delta_pose * pre_insert_pose

            result = planner.move_to_pose_with_screw(pre_insert_pose)
            if result == -1:
                return {"success": False, "failure_reason": "alignment planning failed"}

        # Stage 5: insert.
        result = planner.move_to_pose_with_screw(
            insert_pose * sapien.Pose([0.05, 0, 0])
        )
        if result == -1:
            return {"success": False, "failure_reason": "insertion planning failed"}

        # Read the env's own evaluation straight from the last follow_path
        # step -- do NOT step again with a raw zero action: the gripper
        # controller is normalize_action=True over [-0.01, 0.04], so an
        # action of 0 commands the midpoint (~0.015, half-open), not closed
        # (-1 -> -0.01). An extra env.step(zeros) here was loosening the
        # grip right as success gets measured.
        _, _, terminated, truncated, info = result

        return {
            "success": bool(info.get("success", False)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "failure_reason": None,
        }

    finally:
        planner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    env = gym.make(
        ENV_ID,
        obs_mode="none",
        control_mode="pd_joint_pos",
        reward_mode="dense",
        render_mode="rgb_array" if args.headless else "human",
        max_episode_steps=300,
    )

    try:
        successes = 0

        for episode in range(args.num_episodes):
            seed = args.seed + episode
            result = solve_episode(
                env,
                seed=seed,
                debug=args.debug,
                vis=not args.headless,
            )
            successes += int(result["success"])

            print(
                f"Episode {episode}, seed={seed}: "
                f"success={result['success']}, "
                f"reason={result['failure_reason']}"
            )

        print(f"Success rate: {successes}/{args.num_episodes}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
