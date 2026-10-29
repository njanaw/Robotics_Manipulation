"""Controller (1): move the grasped peg in front of the observed hole, push in along the
hole axis under the stiffness controller, and spiral search in the hole plane if the peg
stalls on the rim [Chhatpar and Branicky 2001].

Everything is computed from the observation, so hole pose noise reaches this controller
exactly as it reaches the learned ones. Frames: the hole frame's x axis is the insertion
direction, the box entry face is at x = -L (L = peg half length = box half depth), and
the task counts success once the peg head reaches x >= -0.015.

Action is pd_ee_delta_pose, normalized: position delta / 0.1 m, rotation delta / 0.1 rad
(XYZ Euler, world frame, sign flipped by ManiSkill's scaling), gripper -1 = closed.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

ALIGN, INSERT, SPIRAL = 0, 1, 2
POS_SCALE, ROT_SCALE = 0.1, 0.1
QUARTER_ROLLS = [R.from_euler("x", k * np.pi / 2) for k in range(4)]


def servo_action(tcp_p, tcp_R, des_p, des_R, gripper=-1.0):
    """One proportional step toward a desired TCP pose."""
    dp = np.clip((des_p - tcp_p) / POS_SCALE, -1, 1)
    rot = -(des_R * tcp_R.inv()).as_euler("XYZ") / ROT_SCALE
    n = np.linalg.norm(rot)
    if n > 1:
        rot /= n
    return np.concatenate([dp, rot, [gripper]]).astype(np.float32)


def split_pose(raw):
    raw = np.asarray(raw, dtype=np.float64)
    return raw[:3], R.from_quat(raw[3:], scalar_first=True)


class ClassicalController:
    def __init__(self, num_envs, standoff=0.02, lead=0.015, lead_inside=0.04, x_final=0.01,
                 align_tol=0.002, align_rot_tol=0.03, align_max_steps=30, ki=0.3, bias_max=0.01,
                 stall_steps=4, stall_progress=0.001,
                 spiral_pitch=0.004, spiral_step=0.002, spiral_rmax=0.012, spiral_press=0.005, drop_depth=0.001):
        self.n = num_envs
        self.standoff, self.lead, self.lead_inside, self.x_final = standoff, lead, lead_inside, x_final
        self.align_tol, self.align_rot_tol, self.align_max_steps = align_tol, align_rot_tol, align_max_steps
        self.ki, self.bias_max = ki, bias_max
        self.stall_steps, self.stall_progress = stall_steps, stall_progress
        self.pitch, self.ds, self.rmax, self.press = spiral_pitch, spiral_step, spiral_rmax, spiral_press
        self.drop_depth = drop_depth
        self.reset()

    def reset(self, mask=None):
        """Clear per-episode memory; mask (bool, num_envs) limits it to the envs starting over."""
        if mask is None:
            self.phase = np.full(self.n, ALIGN)
            self.yz_cmd = np.zeros((self.n, 2))
            self.center = np.zeros((self.n, 2))
            self.theta = np.zeros(self.n)
            self.rim_x = np.zeros(self.n)  # head depth when the spiral started (peg resting on the rim)
            self.bias = np.zeros((self.n, 3))  # integral term in the hole frame; cancels gravity sag
            self.sag_bias = np.zeros((self.n, 3))  # bias when the spiral started, restored after the drop
            self.align_steps = np.zeros(self.n, dtype=int)
            self.history = [[] for _ in range(self.n)]
            self.spiral_count = np.zeros(self.n, dtype=int)  # spiral searches started, for logging
            return
        for i in np.flatnonzero(np.asarray(mask.cpu() if torch.is_tensor(mask) else mask)):
            self.phase[i] = ALIGN
            self.yz_cmd[i] = self.center[i] = 0.0
            self.theta[i] = self.rim_x[i] = 0.0
            self.bias[i] = self.sag_bias[i] = 0.0
            self.align_steps[i] = 0
            self.history[i] = []

    def act(self, obs):
        ex = {k: v.cpu().numpy() for k, v in obs["extra"].items()}
        actions = np.stack([self._act_one(i, ex) for i in range(self.n)])
        return torch.from_numpy(actions)

    def _act_one(self, i, ex):
        tcp_p, tcp_R = split_pose(ex["tcp_pose"][i])
        peg_p, peg_R = split_pose(ex["peg_pose"][i])
        hole_p, hole_R = split_pose(ex["box_hole_pose"][i])
        L = float(ex["peg_half_size"][i, 0])

        head = hole_R.inv().apply(peg_p + peg_R.apply([L, 0, 0]) - hole_p)  # head in hole frame
        # square peg: any quarter roll about the hole axis fits, take the closest
        peg_R_des = min((hole_R * q for q in QUARTER_ROLLS),
                        key=lambda Rd: (Rd * peg_R.inv()).magnitude())

        entry = -L
        hist = self.history[i]
        hist.append(head[0])
        del hist[:-self.stall_steps]

        if self.phase[i] == ALIGN:
            target = np.array([entry - self.standoff, *self.yz_cmd[i]])
            self._integrate(i, target - head)
            self.align_steps[i] += 1
            rot_err = (peg_R_des * peg_R.inv()).magnitude()
            aligned = np.linalg.norm(head - target) < self.align_tol and rot_err < self.align_rot_tol
            if aligned or self.align_steps[i] >= self.align_max_steps:
                self.phase[i] = INSERT
                self.bias[i, 0] = 0.0  # keep only the lateral sag correction while pushing
                hist.clear()
        if self.phase[i] == INSERT:
            inside = head[0] > entry + 0.002
            stalled = (len(hist) == self.stall_steps
                       and max(hist) - min(hist) < self.stall_progress
                       and entry - 0.01 < head[0] < entry + 0.003)
            if stalled:
                self.phase[i] = SPIRAL
                self.center[i] = self.yz_cmd[i]
                self.theta[i] = 0.0
                self.rim_x[i] = head[0]
                self.sag_bias[i] = self.bias[i]
                self.spiral_count[i] += 1
            else:
                if inside:  # the walls set the lateral position; follow them instead of pushing sideways
                    target = np.array([min(head[0] + self.lead_inside, self.x_final), *head[1:]])
                else:
                    target = np.array([min(head[0] + self.lead, self.x_final), *self.yz_cmd[i]])
                    self._integrate(i, np.r_[0.0, target[1:] - head[1:]])
        if self.phase[i] == SPIRAL:
            if head[0] > entry + 0.002:  # dropped into the hole: re-center there and push on
                self.phase[i] = INSERT
                self.yz_cmd[i] = head[1:]
                self.bias[i] = self.sag_bias[i]  # drop the friction wind-up from tracking the spiral
                hist.clear()
                target = np.array([min(head[0] + self.lead_inside, self.x_final), *head[1:]])
            else:
                r = self.pitch * self.theta[i] / (2 * np.pi)
                dropping = head[0] >= self.rim_x[i] + self.drop_depth
                if not dropping:  # hold still while the peg starts to drop
                    self.theta[i] += self.ds / max(r, 0.0005)
                r = self.pitch * self.theta[i] / (2 * np.pi)
                if r > self.rmax:
                    self.theta[i] = 0.0
                    r = 0.0
                yz = self.center[i] + r * np.array([np.cos(self.theta[i]), np.sin(self.theta[i])])
                target = np.array([entry + self.press, *yz])
                # rim friction drags the head off the commanded path; track it with the (exact) peg pose,
                # but not while dropping, where wind-up tilts the peg and wedges it
                if not dropping:
                    self._integrate(i, np.r_[0.0, yz - head[1:]])

        # desired peg pose puts its head at `target` (hole frame); carry the grasp offset to the TCP
        peg_p_des = hole_p + hole_R.apply(target + self.bias[i]) - peg_R_des.apply([L, 0, 0])
        tcp_in_peg_p = peg_R.inv().apply(tcp_p - peg_p)
        tcp_in_peg_R = peg_R.inv() * tcp_R
        return servo_action(tcp_p, tcp_R, peg_p_des + peg_R_des.apply(tcp_in_peg_p), peg_R_des * tcp_in_peg_R)

    def _integrate(self, i, err):
        self.bias[i] = np.clip(self.bias[i] + self.ki * err, -self.bias_max, self.bias_max)


# flat obs_mode="state" layout of PegInsertionSide (noisy_env keeps it): qpos 9, qvel 9, then extra
FLAT_EXTRA_SLICES = dict(tcp_pose=slice(18, 25), peg_pose=slice(25, 32), peg_half_size=slice(32, 35),
                         box_hole_pose=slice(35, 42), box_hole_radius=slice(42, 43))


class ClassicalBase:
    """the classical controller behind residual.py's base interface: reset(mask), act(flat obs)"""

    def __init__(self, num_envs, **kwargs):
        self.controller = ClassicalController(num_envs, **kwargs)

    def reset(self, mask):
        self.controller.reset(mask)

    def act(self, obs):
        extra = {k: obs[:, s] for k, s in FLAT_EXTRA_SLICES.items()}
        return self.controller.act({"extra": extra}).to(obs.device)
