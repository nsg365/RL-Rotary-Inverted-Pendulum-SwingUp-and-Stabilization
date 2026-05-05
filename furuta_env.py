"""Furuta simulation env — logic copied from furuta-master
(furuta/rl/envs/furuta_base.py + furuta_sim.py).

Convention:
    THETA = pendulum, THETA = 0 UPRIGHT, THETA = +-pi HANGING
    ALPHA = arm

Physics state vector order: [arm, pendulum, arm_dot, pendulum_dot]
Action: scalar in [-1, 1], scaled to +-max_voltage in QubeDynamics.
Observation (6D): [cos(arm), sin(arm), cos(pend), sin(pend), arm_vel, pend_vel]

Reward cos_alpha (name kept for config compat with repo) is reshaped so the
maximum is at pendulum UPRIGHT (theta = 0) instead of theta = pi.
"""

from typing import List, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from config import CONSTRAINTS
from furuta_utils import ALPHA, ALPHA_DOT, THETA, THETA_DOT, Timing, VelocityFilter
from qube_dynamics import QubeDynamics


# --- reward (repo math, flipped so upright is theta = 0) ---

def pendulum_reward(state):
    # Sharp exponential reward pointing to theta=0 (UPRIGHT). 
    # At theta=0 -> reward=1. At theta=pi/2 (90deg) -> reward is almost 0.
    return np.exp(-4.0 * (state[THETA] ** 2))

def arm_reward(state):
    # soft penalty for large arm angle (same shape as repo's theta_reward)
    arm_rew = (np.cos(state[ALPHA] + np.pi) + 1) / 2
    return 1 - arm_rew**2

# def cos_alpha_reward(state):
#     return pendulum_reward(state) * arm_reward(state)

def cos_alpha_reward(state):
    # Old shape: (1 + np.cos(state[THETA])) / 2
    # New shape: Only give significant reward when it is VERY close to upright.
    
    pend_angle = state[THETA]
    # Sharp exponential reward: drops off to near 0 very quickly the further from upright it is
    pend_rew = np.exp(-3.0 * (pend_angle ** 2)) 
    
    return pend_rew * arm_reward(state)


def exp_pendulum_reward(state, exp=2):
    th = np.mod((state[THETA] + np.pi), 2 * np.pi) - np.pi
    r = 1.0 - np.abs(th) / np.pi          # 1 at upright, 0 at hanging
    r = (np.exp(r * exp) - np.exp(0)) / np.exp(exp)
    return r


REWARDS = {
    "cos_alpha": cos_alpha_reward,
    "exp_alpha_2": lambda x: exp_pendulum_reward(x, exp=2) * arm_reward(x),
    "exp_alpha_3": lambda x: exp_pendulum_reward(x, exp=3) * arm_reward(x),
    "exp_alpha_4": lambda x: exp_pendulum_reward(x, exp=4) * arm_reward(x),
    "exp_alpha_6": lambda x: exp_pendulum_reward(x, exp=6) * arm_reward(x),
}


class FurutaBase(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        control_freq: int = int(round(1.0 / CONSTRAINTS['dt'])),
        reward: str = "cos_alpha",
        angle_limits=[None, None],
        speed_limits=[60, 400],
        soft_arm_bound: bool = False,
    ):
        self.timing = Timing(control_freq)
        self._state = None
        self.reward = reward
        self._reward_func = REWARDS[self.reward]
        self.soft_arm_bound = soft_arm_bound

        act_max = np.array([1.0], dtype=np.float32)

        def _to_inf(arr):
            a = np.array([np.inf if v is None else v for v in arr], dtype=np.float32)
            return a

        angle_limits = _to_inf(angle_limits)
        speed_limits = _to_inf(speed_limits)

        # state_max ordered [arm, pendulum, arm_dot, pend_dot] to match ALPHA/THETA indices
        self.state_max = np.concatenate([angle_limits, speed_limits])
        self._arm_limit = float(self.state_max[ALPHA])

        obs_max = np.array([1.0, 1.0, 1.0, 1.0, 30, 30], dtype=np.float32)
        if not np.isinf(self.state_max[THETA]):
            obs_max = np.concatenate([np.array([self.state_max[THETA]]), obs_max])
        if not np.isinf(self.state_max[ALPHA]):
            obs_max = np.concatenate([np.array([self.state_max[ALPHA]]), obs_max])

        self.state_space = Box(low=-self.state_max, high=self.state_max, dtype=np.float32)
        self.observation_space = Box(low=-obs_max, high=obs_max, dtype=np.float32)
        self.action_space = Box(low=-act_max, high=act_max, dtype=np.float32)

    def step(self, action):
        self._update_state(action[0])

        # Soft arm bound: clip arm angle + kill arm velocity, don't terminate.
        # Reward penalty applied below.
        arm_violated = False
        if self.soft_arm_bound and not np.isinf(self._arm_limit):
            if abs(self._state[ALPHA]) > self._arm_limit:
                arm_violated = True
                self._state[ALPHA] = np.clip(self._state[ALPHA],
                                             -self._arm_limit, self._arm_limit)
                self._state[ALPHA_DOT] = 0.0

        rwd = self._reward_func(self._state)
        if arm_violated:
            rwd -= 1.0  # penalty for hitting the wall, but episode continues

        obs = self.get_obs()

        if self.soft_arm_bound:
            # Only speed limits and pendulum wrap can terminate now
            speed_ok = (abs(self._state[ALPHA_DOT]) <= self.state_max[ALPHA_DOT] and
                        abs(self._state[THETA_DOT]) <= self.state_max[THETA_DOT])
            terminated = not speed_ok
        else:
            terminated = not self.state_space.contains(self._state)
        truncated = False
        return obs, float(rwd), terminated, truncated, {}

    def get_obs(self):
        obs = np.float32([
            np.cos(self._state[ALPHA]),
            np.sin(self._state[ALPHA]),
            np.cos(self._state[THETA]),
            np.sin(self._state[THETA]),
            self._state[ALPHA_DOT],
            self._state[THETA_DOT],
        ])
        if not np.isinf(self.state_max[THETA]):
            obs = np.concatenate([np.array([self._state[THETA]]), obs])
        if not np.isinf(self.state_max[ALPHA]):
            obs = np.concatenate([np.array([self._state[ALPHA]]), obs])
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

    def _update_state(self, a):
        raise NotImplementedError


class FurutaSim(FurutaBase):
    def __init__(
        self,
        dyn: QubeDynamics = None,
        control_freq: int = int(round(1.0 / CONSTRAINTS['dt'])),
        reward: str = "cos_alpha",
        angle_limits=[None, None],
        speed_limits=[60, 400],
        encoders_CPRs: Optional[List[float]] = None,
        velocity_filter: Optional[int] = None,
        integration_dt: float = 1 / 500,
        max_voltage: float = CONSTRAINTS['max_voltage'],
        soft_arm_bound: bool = False,
        action_noise_std: float = 0.0,
    ):
        super().__init__(control_freq, reward, angle_limits, speed_limits, soft_arm_bound)
        self.max_voltage = max_voltage
        self.dyn = dyn if dyn is not None else QubeDynamics(V=max_voltage)
        self.integration_dt = integration_dt
        self.action_noise_std = action_noise_std
        self.encoders_CPRs = encoders_CPRs
        self.velocity_filter = velocity_filter
        self._init_vel_filt()

    def _init_vel_filt(self):
        if self.velocity_filter:
            self.vel_filt = VelocityFilter(self.velocity_filter, dt=self.timing.dt)
        else:
            self.vel_filt = None

    def _init_state(self):
        # Start near hanging (theta=pi), matching furuta-master convention.
        # Forces the agent to learn swing-up — no free starts near upright.
        self._simulation_state = 0.01 * np.float32(np.random.randn(4))
        self._simulation_state[THETA] = np.pi + 0.01 * np.float32(np.random.randn())
        self._theta_start = float(self._simulation_state[THETA])   # for wrap detection
        self._state = self._simulation_state.copy()
        self._state[THETA] = ((self._state[THETA] + np.pi) % (2 * np.pi)) - np.pi

    def _update_state(self, a):
        if self.action_noise_std > 0:
            a = a + np.random.normal(0, self.action_noise_std)
            a = np.clip(a, -1.0, 1.0)
        integration_steps = int(self.timing.dt / self.integration_dt)
        for _ in range(integration_steps):
            arm_acc, pend_acc = self.dyn(self._simulation_state, a)
            self._simulation_state[ALPHA_DOT] += self.integration_dt * arm_acc
            self._simulation_state[THETA_DOT] += self.integration_dt * pend_acc
            self._simulation_state[ALPHA] += self.integration_dt * self._simulation_state[ALPHA_DOT]
            self._simulation_state[THETA] += self.integration_dt * self._simulation_state[THETA_DOT]

        if self.encoders_CPRs:
            ti = 2 * np.pi / self.encoders_CPRs[THETA]
            self._state[THETA] = np.round(self._simulation_state[THETA] / ti) * ti
            ai = 2 * np.pi / self.encoders_CPRs[ALPHA]
            self._state[ALPHA] = np.round(self._simulation_state[ALPHA] / ai) * ai
        else:
            self._state[THETA] = self._simulation_state[THETA]
            self._state[ALPHA] = self._simulation_state[ALPHA]

        # Wrap pendulum angle to (-pi, pi] so small oscillations across +-pi
        # don't trip the state_space bound. Full rotations are caught separately
        # in step() via cumulative unwrapped displacement.
        self._state[THETA] = ((self._state[THETA] + np.pi) % (2 * np.pi)) - np.pi

        if self.vel_filt:
            self._state[2:4] = self.vel_filt(self._state[0:2])
        else:
            self._state[THETA_DOT] = self._simulation_state[THETA_DOT]
            self._state[ALPHA_DOT] = self._simulation_state[ALPHA_DOT]

    def step(self, action):
        obs, rwd, terminated, truncated, info = super().step(action)
        # If soft arm bound clipped _state, also clip the sim state so they stay in sync.
        if self.soft_arm_bound and not np.isinf(self._arm_limit):
            if abs(self._simulation_state[ALPHA]) > self._arm_limit:
                self._simulation_state[ALPHA] = np.clip(
                    self._simulation_state[ALPHA], -self._arm_limit, self._arm_limit)
                self._simulation_state[ALPHA_DOT] = 0.0
        # Wire-safety: terminate if the pendulum has rotated more than one full
        # revolution from its starting position (true wrap-around).
        if abs(float(self._simulation_state[THETA]) - self._theta_start) >= 2 * np.pi:
            terminated = True
        return obs, rwd, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        self.dyn.randomize()
        self._init_state()
        obs = self.get_obs()
        self._init_vel_filt()
        return obs, {}
