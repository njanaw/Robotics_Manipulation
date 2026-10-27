"""Collision-aware vertical peg insertion for ManiSkill 3.

The fixture is a square frame made from four kinematic boxes. The opening is
along world Z; its opening plane is the upper surface and +Z points upward.
Physical geometry drives collision and success. A separately sampled estimated
opening pose is exposed to policies for pose-error experiments.
"""
from typing import Any, Sequence, Union

import gymnasium as gym
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PegInsertion-sutnaw", max_episode_steps=100)
class PegInsertEnv(BaseEnv):
    """Z-axis square-peg insertion.

    Error bounds are per-axis maximum absolute values in metres/radians.
    `T_estimated = T_error_world * T_physical`; the sampled error is fixed
    until the next reset.
    """
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("sparse", "dense", "normalized_dense", "none")
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda", peg_size: Sequence[float] = (0.02, 0.02, 0.08),
                 opening_clearance=0.003, fixture_outer_size=0.10, fixture_height=0.02,
                 fixture_xy: Sequence[float] = (0.15, 0.0), min_insertion_depth=0.012,
                 lateral_tolerance=0.001, orientation_tolerance=0.12,
                 perception_translation_error: Sequence[float] = (0., 0., 0.),
                 perception_rotation_error: Sequence[float] = (0., 0., 0.),
                 include_physical_opening_in_obs=False,
                 grasp_translation: Sequence[float] = (0., 0., 0.),
                 peg_orientation_mode="yaw", full_orientation_tilt_limit=0.35,
                 kinematic_grasp=True, **kwargs):
        self.peg_size = tuple(map(float, peg_size))
        self.peg_half_size = tuple(x / 2 for x in self.peg_size)
        self.opening_clearance = float(opening_clearance)
        self.opening_size = (self.peg_size[0] + 2 * self.opening_clearance,
                             self.peg_size[1] + 2 * self.opening_clearance)
        self.fixture_outer_size, self.fixture_height = float(fixture_outer_size), float(fixture_height)
        self.fixture_xy = tuple(map(float, fixture_xy))
        self.min_insertion_depth, self.lateral_tolerance = float(min_insertion_depth), float(lateral_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.perception_translation_error = tuple(map(float, perception_translation_error))
        self.perception_rotation_error = tuple(map(float, perception_rotation_error))
        self.include_physical_opening_in_obs = bool(include_physical_opening_in_obs)
        self.grasp_translation = tuple(map(float, grasp_translation))
        self.peg_orientation_mode = str(peg_orientation_mode)
        self.full_orientation_tilt_limit = float(full_orientation_tilt_limit)
        self.kinematic_grasp = bool(kinematic_grasp)
        if min(self.peg_size) <= 0 or self.fixture_height <= 0:
            raise ValueError("Peg dimensions and fixture height must be positive")
        if self.fixture_outer_size <= max(self.opening_size):
            raise ValueError("fixture_outer_size must exceed opening dimensions")
        if self.peg_orientation_mode not in ("yaw", "full", "fixed"):
            raise ValueError("peg_orientation_mode must be 'yaw', 'full', or 'fixed'")
        if self.full_orientation_tilt_limit < 0 or self.full_orientation_tilt_limit > torch.pi / 2:
            raise ValueError("full_orientation_tilt_limit must be in [0, pi/2]")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()
        self.peg_tip_offset = Pose.create_from_pq(
            p=torch.tensor([[0., 0., -self.peg_half_size[2]]], device=self.device))
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=self.peg_half_size)
        builder.add_box_visual(half_size=self.peg_half_size, material=sapien.render.RenderMaterial(base_color=[1., .45, .05, 1.]))
        builder.initial_pose = sapien.Pose(p=[0, 0, self.peg_half_size[2]])
        self.peg = builder.build(name="vertical_peg")

        # Four frame pieces leave a genuine square through-opening: no collider
        # is created under its aperture.
        outer, ox, oy, hz = self.fixture_outer_size / 2, self.opening_size[0], self.opening_size[1], self.fixture_height / 2
        pieces = [
            ([0, (oy / 2 + outer) / 2, hz], [outer, (outer - oy / 2) / 2, hz]),
            ([0, -(oy / 2 + outer) / 2, hz], [outer, (outer - oy / 2) / 2, hz]),
            ([(ox / 2 + outer) / 2, 0, hz], [(outer - ox / 2) / 2, oy / 2, hz]),
            ([-(ox / 2 + outer) / 2, 0, hz], [(outer - ox / 2) / 2, oy / 2, hz]),
        ]
        builder = self.scene.create_actor_builder()
        material = sapien.render.RenderMaterial(base_color=[.75, .1, .1, 1.])
        for p, half in pieces:
            local = sapien.Pose(p=p)
            builder.add_box_collision(local, half_size=half)
            builder.add_box_visual(local, half_size=half, material=material)
        builder.initial_pose = sapien.Pose(p=[*self.fixture_xy, 0])
        self.fixture = builder.build_kinematic(name="vertical_insertion_fixture")
        # Explicit frame at the top surface, rather than fixture actor centre.
        self.opening_offset = Pose.create_from_pq(p=torch.tensor([[0., 0., self.fixture_height]], device=self.device))
        self.estimated_opening_pose = Pose.create_from_pq(p=torch.zeros((self.num_envs, 3), device=self.device))
        self.perception_error = torch.zeros((self.num_envs, 6), device=self.device)
        # Per-episode TCP-to-peg transform. It is sampled at reset and used by
        # the optional scripted carry mode only.
        self.grasp_offset = Pose.create_from_pq(p=torch.zeros((self.num_envs, 3), device=self.device))
        self.sampled_peg_orientation = torch.zeros((self.num_envs, 4), device=self.device)
        self.sampled_peg_orientation[:, 0] = 1

    @property
    def physical_opening_pose(self):
        return self.fixture.pose * self.opening_offset

    @property
    def peg_tip_pose(self):
        return self.peg.pose * self.peg_tip_offset

    @staticmethod
    def _euler_xyz_to_quat(euler):
        half = euler * .5
        cx, cy, cz = torch.cos(half).unbind(-1)
        sx, sy, sz = torch.sin(half).unbind(-1)
        return torch.stack((cx*cy*cz + sx*sy*sz, sx*cy*cz - cx*sy*sz,
                            cx*sy*cz + sx*cy*sz, cx*cy*sz - sx*sy*cz), -1)

    def _sample_perception_error(self, env_idx):
        tb = torch.tensor(self.perception_translation_error, device=self.device)
        rb = torch.tensor(self.perception_rotation_error, device=self.device)
        rng = self._batched_episode_rng[env_idx]
        # BatchedRNG itself supplies the leading environment dimension.
        translation = common.to_tensor(rng.uniform(-1, 1, size=(3,)), device=self.device) * tb
        rotation = common.to_tensor(rng.uniform(-1, 1, size=(3,)), device=self.device) * rb
        error = Pose.create_from_pq(translation, self._euler_xyz_to_quat(rotation))
        estimated = error * self.physical_opening_pose[env_idx]
        self.estimated_opening_pose.raw_pose[env_idx] = estimated.raw_pose
        self.perception_error[env_idx] = torch.cat((translation, rotation), -1)

    def _sample_peg_orientation(self, env_idx):
        """Sample a world-frame peg orientation, returned as WXYZ quaternions."""
        b = len(env_idx)
        if self.peg_orientation_mode == "fixed":
            euler = torch.zeros((b, 3), device=self.device)
        else:
            rng = self._batched_episode_rng[env_idx]
            yaw = common.to_tensor(rng.uniform(-torch.pi, torch.pi, size=()), device=self.device)
            euler = torch.zeros((b, 3), device=self.device)
            euler[:, 2] = yaw
            if self.peg_orientation_mode == "full":
                tilt = common.to_tensor(
                    rng.uniform(-self.full_orientation_tilt_limit, self.full_orientation_tilt_limit, size=(2,)),
                    device=self.device,
                )
                euler[:, :2] = tilt
        return self._euler_xyz_to_quat(euler)

    def _set_peg_from_tcp(self, env_idx=None):
        """Apply the stored TCP-relative grasp transform at a defined sim hook."""
        tcp = self.agent.tcp.pose
        pose = tcp * self.grasp_offset
        if not torch.isfinite(pose.raw_pose).all():
            raise RuntimeError("Non-finite TCP-derived peg pose")
        # Actor is dynamic for ordinary collision behaviour. In scripted grasp
        # mode this pose is deliberately imposed before each physics substep.
        self.peg.set_pose(pose if env_idx is None else pose[env_idx])
        if env_idx is None:
            self.peg.set_linear_velocity(torch.zeros((self.num_envs, 3), device=self.device))
            self.peg.set_angular_velocity(torch.zeros((self.num_envs, 3), device=self.device))

    def _before_simulation_step(self):
        if self.kinematic_grasp:
            self._set_peg_from_tcp()

    def _after_control_step(self):
        # Apply once more at the documented end-of-control hook so returned
        # observations exactly represent the scripted grasp after the final
        # physics substep.
        if self.kinematic_grasp:
            self._set_peg_from_tcp()

    def _initialize_episode(self, env_idx, options):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)
        fixture_p = torch.tensor([*self.fixture_xy, 0.], device=self.device).repeat(b, 1)
        self.fixture.set_pose(Pose.create_from_pq(fixture_p))
        # TableSceneBuilder has reset the Panda/Fetch to its installed rest
        # keyframe. Compute from the resulting measured TCP, never from an
        # assumed Cartesian position.
        tcp = self.agent.tcp.pose
        if not torch.isfinite(tcp.raw_pose).all():
            raise RuntimeError("Non-finite robot TCP pose at reset")
        orientation = self._sample_peg_orientation(env_idx)
        position_offset = Pose.create_from_pq(
            p=torch.tensor(self.grasp_translation, device=self.device).repeat(b, 1))
        peg_position = (tcp[env_idx] * position_offset).p
        desired_peg = Pose.create_from_pq(peg_position, orientation)
        # Keep a genuine transform relative to TCP, including the sampled
        # world-yaw orientation. Subsequent attachment uses tcp * this offset.
        offset = tcp[env_idx].inv() * desired_peg
        self.grasp_offset.raw_pose[env_idx] = offset.raw_pose
        self.sampled_peg_orientation[env_idx] = orientation
        self.peg.set_pose(desired_peg)
        self._sample_perception_error(env_idx)

    def _insertion_metrics(self):
        tip = (self.physical_opening_pose.inv() * self.peg_tip_pose).p
        lateral = torch.linalg.norm(tip[:, :2], dim=-1)
        depth = -tip[:, 2]
        q = (self.physical_opening_pose.inv() * self.peg.pose).q
        z_dot = 1 - 2 * (q[:, 1]**2 + q[:, 2]**2)
        orientation = torch.acos(torch.clamp(z_dot, -1., 1.))
        w, x, y, z = q.unbind(-1)
        r00, r01 = 1 - 2*(y*y + z*z), 2*(x*y-z*w)
        r10, r11 = 2*(x*y+z*w), 1 - 2*(x*x + z*z)
        # Cross-sectional XY support dimensions account for yaw/orientation.
        fx = self.peg_half_size[0]*r00.abs() + self.peg_half_size[1]*r01.abs()
        fy = self.peg_half_size[0]*r10.abs() + self.peg_half_size[1]*r11.abs()
        fits = ((fx <= self.opening_size[0]/2 - tip[:, 0].abs()) &
                (fy <= self.opening_size[1]/2 - tip[:, 1].abs()))
        return dict(tip_in_opening=tip, lateral_error=lateral, insertion_depth=depth,
                    orientation_error=orientation, fits_opening=fits)

    def evaluate(self):
        m = self._insertion_metrics()
        success = ((m["insertion_depth"] >= self.min_insertion_depth) &
                   (m["lateral_error"] <= self.lateral_tolerance) &
                   (m["orientation_error"] <= self.orientation_tolerance) & m["fits_opening"])
        return dict(success=success, **m)

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            # Policy gets the estimated pose. Ground truth remains available via
            # physical_opening_pose/evaluate, and can be explicitly enabled for
            # debugging without accidentally leaking it in normal experiments.
            obs.update(peg_pose=self.peg.pose.raw_pose,
                       peg_size=torch.tensor(self.peg_size, device=self.device).repeat(self.num_envs, 1),
                       estimated_opening_pose=self.estimated_opening_pose.raw_pose,
                       grasp_offset=self.grasp_offset.raw_pose,
                       sampled_peg_orientation=self.sampled_peg_orientation,
                       kinematic_grasp=torch.full((self.num_envs, 1), self.kinematic_grasp,
                                                  dtype=torch.bool, device=self.device))
            if self.include_physical_opening_in_obs:
                obs["physical_opening_pose"] = self.physical_opening_pose.raw_pose
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        m = self._insertion_metrics()
        reach = 1 - torch.tanh(5 * torch.linalg.norm(self.agent.tcp.pose.p - self.peg.pose.p, dim=-1))
        align = 1.5 * (1 - torch.tanh(35 * m["lateral_error"]))
        orient = 1 - torch.tanh(4 * m["orientation_error"])
        insertion = 3 * torch.tanh(35 * torch.clamp(m["insertion_depth"], min=0)) * m["fits_opening"].float()
        reward = reach + align + orient + insertion
        return torch.where(info["success"], torch.full_like(reward, 7.), reward)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 7.

def main():
    num_episodes = 100

    env = gym.make(
        "PegInsertion-sutnaw",
        obs_mode="state",
        reward_mode="dense",
        render_mode="human",
        render_backend="gpu",
    )

    try:
        for episode in range(num_episodes):
            obs, info = env.reset(seed=episode)
            print(f"Episode {episode + 1}/{num_episodes} started")

            terminated = False
            truncated = False
            step = 0

            while not (terminated or truncated):
                action = env.action_space.sample() * 0
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                step += 1

            print(
                f"Episode {episode + 1} finished "
                f"after {step} steps. "
                f"Success: {info.get('success', 'not available')}"
            )

    except KeyboardInterrupt:
        print("Visualization stopped.")

    finally:
        env.close()


if __name__ == "__main__":
    main()
