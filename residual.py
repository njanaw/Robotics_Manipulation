import gymnasium as gym
import numpy as np
import torch


class HoldBase:
    """stand-in base until the classical controller lands: hold the pose, keep the gripper closed

    every base follows this interface: reset(mask) clears memory of the masked envs (e.g. a diffusion
    policy's action chunk), act(obs) returns this control step's action for all envs
    """

    def __init__(self, act_dim):
        self.act_dim = act_dim

    def reset(self, mask):
        pass

    def act(self, obs):
        action = torch.zeros((obs.shape[0], self.act_dim), device=obs.device)
        # pd_ee_delta_pose gripper: -1 closes
        action[:, -1] = -1
        return action


class ResidualWrapper(gym.vector.VectorWrapper):
    """ResiP residual: executes base action + scale * residual at every control step

    the agent observes [env obs, base action for this step]; the frozen base sees the same env obs,
    so a hole offset injected by the env reaches both. wraps a ManiSkillVectorEnv, whose auto-reset
    returns the next episode's first obs for done envs, so the base is reset before it acts on them
    """

    def __init__(self, env, base, scale):
        super().__init__(env)
        self.base, self.scale = base, scale
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim + act_dim,), np.float32)
        self.action_low = torch.as_tensor(env.single_action_space.low, device=env.device)
        self.action_high = torch.as_tensor(env.single_action_space.high, device=env.device)
        self.base_action = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if options is not None and "env_idx" in options:
            # partial reset: only the listed envs start a new episode
            mask = torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
            mask[options["env_idx"]] = True
        else:
            mask = torch.ones(obs.shape[0], dtype=torch.bool, device=obs.device)
        self.base.reset(mask)
        self.base_action = self.base.act(obs)
        return torch.cat([obs, self.base_action], -1), info

    def step(self, actions):
        action = torch.clamp(self.base_action + self.scale * actions, self.action_low, self.action_high)
        obs, rew, terminations, truncations, infos = self.env.step(action)
        # unconditional: checking done.any() would force a gpu sync every step
        self.base.reset(terminations | truncations)
        self.base_action = self.base.act(obs)
        return torch.cat([obs, self.base_action], -1), rew, terminations, truncations, infos
