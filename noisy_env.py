import torch

from mani_skill.envs.tasks import PegInsertionSideEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PegInsertionSideNoisy-v1", max_episode_steps=100)
class PegInsertionSideNoisyEnv(PegInsertionSideEnv):
    """PegInsertionSide-v1 whose observed hole pose is wrong by a per-episode offset

    exists so every method (base and residual alike) sees the same wrong hole estimate. the offset
    magnitude is drawn from U[hole_offset_range] metres and its direction uniformly in the plane
    normal to the insertion axis, once per episode. only the observation changes: the physical box,
    reward and success all use the true pose. (r, r) gives the fixed sweep offset, (0, 0.004) the
    noise-trained residual
    """

    def __init__(self, *args, hole_offset_range=(0.0, 0.0), **kwargs):
        self.hole_offset_range = hole_offset_range
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.hole_offsets = torch.zeros((self.num_envs, 3), device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        b = len(env_idx)
        low, high = self.hole_offset_range
        magnitude = torch.rand(b, device=self.device) * (high - low) + low
        angle = torch.rand(b, device=self.device) * 2 * torch.pi
        # hole frame x is the insertion axis (see has_peg_inserted), so the offset lives in y-z
        self.hole_offsets[env_idx, 0] = 0
        self.hole_offsets[env_idx, 1] = magnitude * torch.cos(angle)
        self.hole_offsets[env_idx, 2] = magnitude * torch.sin(angle)

    @property
    def observed_hole_pose(self):
        return self.box_hole_pose * Pose.create_from_pq(p=self.hole_offsets)

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if self.obs_mode_struct.use_state:
            # overwriting the existing key keeps the flattened obs layout identical to the base task
            obs["box_hole_pose"] = self.observed_hole_pose.raw_pose
        return obs
