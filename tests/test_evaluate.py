import numpy as np
import pytest
import torch

from mani_skill.utils import common

import evaluate  # first: sets up macOS Vulkan before any env is built
import start_states
from classical import ClassicalBase, ClassicalController

STARTS = "data/interim_test.pt"


@pytest.fixture(scope="module")
def starts():
    return start_states.load(STARTS)


@pytest.fixture(scope="module")
def env():
    env = evaluate.make_env(num_envs=1, sim_backend="physx_cpu")
    yield env
    env.close()


def test_reset_to_restores_saved_scene(env, starts):
    for k in [0, 7]:
        start_states.reset_to(env, starts, [k])
        saved_peg = starts["env_states"]["actors"]["peg"][k, :7]
        assert torch.allclose(env.unwrapped.peg.pose.raw_pose[0], saved_peg, atol=1e-5)
        # contact forces only exist after a physics step, so check the grasp after one closed-gripper hold
        hold = torch.zeros((1, 7))
        hold[:, -1] = -1
        env.step(hold)
        assert env.unwrapped.agent.is_grasping(env.unwrapped.peg)[0]


@pytest.mark.parametrize("m", [0.0, 0.004, 0.006])
def test_sweep_offset_is_exact_and_survives_steps(env, starts, m):
    offsets = evaluate.sweep_offsets(len(starts["seeds"]), m, seed=[0, int(m * 1e6)])
    start_states.reset_to(env, starts, [3])
    evaluate.set_hole_offsets(env, offsets[3:4])
    u = env.unwrapped
    for _ in range(3):
        d = (u.box_hole_pose.inv() * u.observed_hole_pose).p[0]
        assert abs(d[0].item()) < 1e-6
        assert abs(torch.linalg.norm(d[1:]).item() - m) < 1e-6
        obs, *_ = env.step(torch.zeros((1, 7)))
        assert torch.allclose(obs["extra"]["box_hole_pose"][0], u.observed_hole_pose.raw_pose[0])


def test_sweep_directions_depend_only_on_seed():
    a = evaluate.sweep_offsets(10, 0.004, seed=[0, 4000])
    b = evaluate.sweep_offsets(10, 0.004, seed=[0, 4000])
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 0.004)


def test_flat_obs_base_matches_dict_controller(env, starts):
    obs, _ = start_states.reset_to(env, starts, [1])
    flat = common.flatten_state_dict(obs, use_torch=True)
    a = ClassicalController(1).act(obs)
    b = ClassicalBase(1).act(flat)
    assert torch.allclose(a, b)


def test_classical_inserts_with_true_hole_pose():
    results = evaluate.evaluate("classical", STARTS, [0.0], num_envs=1, sim_backend="physx_cpu",
                                max_steps=100, seed=0, limit=3)
    assert results["sweep"][0]["success_once"] == 1.0
