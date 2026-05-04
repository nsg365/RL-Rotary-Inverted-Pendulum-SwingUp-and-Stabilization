"""Parse classical_data_real.csv and pre-fill an SB3 replay buffer.

CSV columns (in order): arm_rad, arm_vel, pend_rad, pend_vel, voltage_u

Mapping into the repo's state / obs / action convention:
    physics state vector = [arm_angle, pend_angle, arm_vel, pend_vel]
    obs (6D, matches FurutaBase.get_obs with no angle limits prepended):
        [cos(arm), sin(arm), cos(pend), sin(pend), arm_vel, pend_vel]
    action (1D in [-1, 1]) = voltage_u / max_voltage

Reward is recomputed with the same REWARDS['cos_alpha'] used by the env, so
bootstrapped transitions are perfectly consistent with the training signal.

Usage:
    python bootstrap_from_csv.py --csv classical_data_real.csv --out buffer.pkl

Then in train.py set replay_buffer_path to that pickle.
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

from config import CONSTRAINTS
from furuta_env import FurutaSim, REWARDS
from furuta_utils import ALPHA, THETA, ALPHA_DOT, THETA_DOT


def csv_to_buffer(csv_path: str, out_path: str, buffer_size: int = 1_000_000):
    df = pd.read_csv(csv_path)
    needed = ["arm_rad", "arm_vel", "pend_rad", "pend_vel", "voltage_u"]
    for c in needed:
        assert c in df.columns, f"missing column {c}"

    # Build a throwaway env just to grab the observation/action spaces.
    # Must match train.py settings or the buffer's obs dim won't align.
    env = FurutaSim(reward="exp_alpha_2", angle_limits=[np.pi / 2, None], speed_limits=[60, 400])
    obs_space = env.observation_space
    act_space = env.action_space

    buf = ReplayBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=act_space,
        handle_timeout_termination=False,
    )

    max_v = CONSTRAINTS['max_voltage']
    reward_fn = REWARDS["cos_alpha"]

    def state_from_row(r):
        s = np.zeros(4, dtype=np.float32)
        s[ALPHA] = r["arm_rad"]
        # CSV convention matches env convention: pendulum = 0 at upright.
        s[THETA] = r["pend_rad"]
        s[ALPHA_DOT] = r["arm_vel"]
        s[THETA_DOT] = r["pend_vel"]
        return s

    def obs_from_state(s):
        o = np.array([
            np.cos(s[ALPHA]), np.sin(s[ALPHA]),
            np.cos(s[THETA]), np.sin(s[THETA]),
            s[ALPHA_DOT], s[THETA_DOT],
        ], dtype=np.float32)
        # prepend raw angles if state_max is finite (matches FurutaBase.get_obs)
        if not np.isinf(env.state_max[THETA]):
            o = np.concatenate([np.array([s[THETA]], dtype=np.float32), o])
        if not np.isinf(env.state_max[ALPHA]):
            o = np.concatenate([np.array([s[ALPHA]], dtype=np.float32), o])
        return o

    n = len(df)
    added = 0
    for i in range(n - 1):
        s_t = state_from_row(df.iloc[i])
        s_tp1 = state_from_row(df.iloc[i + 1])

        o_t = obs_from_state(s_t)
        o_tp1 = obs_from_state(s_tp1)

        a_t = np.array([np.clip(df.iloc[i]["voltage_u"] / max_v, -1.0, 1.0)], dtype=np.float32)
        r_t = float(reward_fn(s_t))
        done = (i == n - 2)

        buf.add(
            obs=o_t,
            next_obs=o_tp1,
            action=a_t,
            reward=np.array([r_t], dtype=np.float32),
            done=np.array([done], dtype=np.float32),
            infos=[{}],
        )
        added += 1

    with open(out_path, "wb") as f:
        pickle.dump(buf, f)

    print(f"Pre-filled {added} transitions → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="classical_data_real.csv")
    p.add_argument("--out", default="buffer.pkl")
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    args = p.parse_args()
    csv_to_buffer(args.csv, args.out, args.buffer_size)
