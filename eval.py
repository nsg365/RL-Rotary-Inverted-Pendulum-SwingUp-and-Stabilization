"""Rollout a trained SAC policy."""

import argparse

import numpy as np
from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit

from config import CONSTRAINTS
from furuta_env import FurutaSim
from furuta_utils import ALPHA, THETA


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/sac_gsde/model.zip")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max_voltage", type=float, default=CONSTRAINTS['max_voltage'])
    p.add_argument("--arm_limit_deg", type=float, default=90.0)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--soft_arm", action="store_true",
                   help="Soft arm bound (must match training setting).")
    args = p.parse_args()

    arm_limit_rad = float(np.deg2rad(args.arm_limit_deg))
    env = TimeLimit(
        FurutaSim(reward="cos_alpha", angle_limits=[arm_limit_rad, None],
                  speed_limits=[60, 400], max_voltage=args.max_voltage,
                  soft_arm_bound=args.soft_arm),
        max_episode_steps=args.max_steps,
    )
    model = SAC.load(args.model)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_r = 0.0
        steps = 0
        max_pend = 0.0
        max_arm = 0.0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            total_r += r
            steps += 1
            max_pend = max(max_pend, abs(env.unwrapped._state[THETA]))
            max_arm = max(max_arm, abs(env.unwrapped._state[ALPHA]))
            if term or trunc:
                break
        print(f"ep {ep}: steps={steps} reward={total_r:.2f} "
              f"max|pend|={max_pend:.2f} rad ({np.degrees(max_pend):.0f} deg) "
              f"max|arm|={max_arm:.2f} rad ({np.degrees(max_arm):.0f} deg)")


if __name__ == "__main__":
    main()
