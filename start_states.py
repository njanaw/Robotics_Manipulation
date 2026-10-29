"""Saved start states (peg already grasped) shared by every method.

File format (torch.save of a dict):
    seeds          LongTensor (N,)   episode seed; fixes peg size and hole offset on reconfigure
    env_states     nested dict of tensors with leading dim N, from env.get_state_dict()
    peg_half_sizes FloatTensor (N, 3) used to check that geometry was rebuilt correctly
    meta           dict (how the states were made)
"""

import torch


def stack_states(state_dicts):
    first = state_dicts[0]
    if isinstance(first, dict):
        return {k: stack_states([s[k] for s in state_dicts]) for k in first}
    return torch.cat([torch.as_tensor(s) for s in state_dicts], dim=0)


def index_states(states, idx):
    if isinstance(states, dict):
        return {k: index_states(v, idx) for k, v in states.items()}
    return states[idx]


def load(path):
    return torch.load(path, weights_only=False)


def save(path, seeds, env_states, peg_half_sizes, meta):
    torch.save(
        dict(seeds=torch.as_tensor(seeds), env_states=env_states,
             peg_half_sizes=torch.as_tensor(peg_half_sizes), meta=meta),
        path,
    )


def reset_to(env, starts, idx):
    """Reset env (num_envs == len(idx)) to start states starts[idx]. Returns obs, info."""
    idx = torch.as_tensor(idx)
    u = env.unwrapped
    states = {k: v for k, v in index_states(starts["env_states"], idx).items()}
    states = _to_device(states, u.device)
    obs, info = env.reset(
        seed=starts["seeds"][idx].tolist(),
        options=dict(reconfigure=True, reset_to_env_states=dict(env_states=states)),
    )
    got = u.peg_half_sizes.cpu()
    want = starts["peg_half_sizes"][idx]
    assert torch.allclose(got, want, atol=1e-6), "geometry mismatch: seeds do not reproduce the saved scene"
    return obs, info


def _to_device(states, device):
    if isinstance(states, dict):
        return {k: _to_device(v, device) for k, v in states.items()}
    return states.to(device)
