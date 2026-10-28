import gymnasium as gym
import pytest
import torch

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import noisy_env  # noqa: F401  registers PegInsertionSideNoisy-v1
from residual import HoldBase, ResidualWrapper


def make_env(hole_offset_range):
    env = gym.make("PegInsertionSideNoisy-v1", obs_mode="state", control_mode="pd_ee_delta_pose",
                   sim_backend="physx_cpu", hole_offset_range=hole_offset_range)
    return ManiSkillVectorEnv(env, 1, ignore_terminations=True)


def hole_frame_offset(env):
    base_env = env.unwrapped
    return (base_env.box_hole_pose.inv() * base_env.observed_hole_pose).p[0]


@pytest.mark.parametrize("r", [0.0, 0.002, 0.004, 0.006])
def test_sweep_offset_is_exact_in_hole_plane_and_fixed_per_episode(r):
    env = make_env((r, r))
    for seed in range(3):
        env.reset(seed=seed)
        first = hole_frame_offset(env)
        assert abs(first[0].item()) < 1e-6
        assert abs(torch.linalg.norm(first[1:]).item() - r) < 1e-6
        for _ in range(5):
            env.step(env.action_space.sample() * 0)
            assert torch.allclose(hole_frame_offset(env), first, atol=1e-6)
    env.close()


def test_training_offset_stays_within_range():
    env = make_env((0.0, 0.004))
    for seed in range(20):
        env.reset(seed=seed)
        assert torch.linalg.norm(hole_frame_offset(env)).item() <= 0.004 + 1e-6
    env.close()


def test_observed_hole_pose_reaches_policy_obs():
    env = make_env((0.004, 0.004))
    obs, _ = env.reset(seed=0)
    base_env = env.unwrapped
    observed = base_env.observed_hole_pose.raw_pose[0]
    true = base_env.box_hole_pose.raw_pose[0]
    assert any(torch.allclose(obs[0, i:i + 7], observed, atol=1e-6) for i in range(obs.shape[1] - 6))
    assert not any(torch.allclose(obs[0, i:i + 7], true, atol=1e-6) for i in range(obs.shape[1] - 6))
    env.close()


def test_zero_residual_reproduces_base():
    base_only = make_env((0.002, 0.002))
    residual_env = make_env((0.002, 0.002))
    act_dim = base_only.single_action_space.shape[0]
    base = HoldBase(act_dim)
    wrapped = ResidualWrapper(residual_env, HoldBase(act_dim), 0.1)
    obs, _ = base_only.reset(seed=0)
    wrapped_obs, _ = wrapped.reset(seed=0)
    for _ in range(20):
        assert torch.allclose(wrapped_obs[:, :obs.shape[1]], obs, atol=1e-5)
        assert torch.equal(wrapped_obs[:, obs.shape[1]:], base.act(obs))
        obs, *_ = base_only.step(base.act(obs))
        wrapped_obs, *_ = wrapped.step(torch.zeros((1, act_dim)))
    base_only.close()
    wrapped.close()


class RecordingBase(HoldBase):
    def __init__(self, act_dim):
        super().__init__(act_dim)
        self.events = []

    def reset(self, mask):
        self.events.append(("reset", bool(mask.any())))

    def act(self, obs):
        self.events.append(("act", obs.clone()))
        return super().act(obs)


def test_base_resets_before_acting_on_next_episode():
    env = make_env((0.0, 0.0))
    act_dim = env.single_action_space.shape[0]
    base = RecordingBase(act_dim)
    wrapped = ResidualWrapper(env, base, 0.1)
    wrapped.reset(seed=0)
    for _ in range(100):
        base.events.clear()
        obs, _, _, truncations, _ = wrapped.step(torch.zeros((1, act_dim)))
        if truncations.any():
            break
    assert truncations.any()
    # auto-reset already happened, so obs is the next episode's first obs
    assert env.unwrapped.elapsed_steps[0].item() == 0
    (reset_kind, reset_any), (act_kind, act_obs) = base.events
    assert (reset_kind, reset_any, act_kind) == ("reset", True, "act")
    assert torch.equal(act_obs, obs[:, :act_obs.shape[1]])
    wrapped.close()
