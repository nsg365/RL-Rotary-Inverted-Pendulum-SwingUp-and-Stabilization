"""Pure inference on hardware: load sim-trained SAC model, run deterministic
actions, print state every N steps. No learning, no exploration noise.

Use this to verify the sim policy works on real hardware before training.

Usage:
    python eval_real.py --port /dev/cu.usbmodem1401 \
        --model runs/sac_gsde/best/best_model.zip
"""

import argparse
import time

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC

from config import CONSTRAINTS
from furuta_real import FurutaReal
from furuta_utils import ALPHA, THETA


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--model", default="runs/sac_gsde/best/best_model.zip")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--action_smoothing", type=float, default=0.3)
    p.add_argument("--print_every", type=int, default=20)
    args = p.parse_args()

    env = TimeLimit(
        FurutaReal(port=args.port, action_smoothing=args.action_smoothing),
        max_episode_steps=CONSTRAINTS['max_steps'],
    )
    model = SAC.load(args.model)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        step = 0
        t0 = time.time()
        max_pend_up = 0.0  # track best (smallest |theta|) proximity to upright
        max_pend_up = np.pi
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            step += 1

            s = env.unwrapped._state
            th = float(s[THETA])
            al = float(s[ALPHA])
            v = float(a[0]) * CONSTRAINTS['max_voltage']
            if abs(th) < max_pend_up:
                max_pend_up = abs(th)

            if step % args.print_every == 0:
                hz = step / (time.time() - t0)
                print(f"ep{ep} step{step:4d} "
                      f"θ={np.degrees(th):+7.1f}°  α={np.degrees(al):+7.1f}°  "
                      f"V={v:+6.2f}  best|θ|={np.degrees(max_pend_up):6.1f}°  "
                      f"rate={hz:5.1f} Hz")

            if term or trunc:
                print(f"-- ep {ep} ended at step {step}, best |θ|={np.degrees(max_pend_up):.1f}°")
                break

    env.close()


if __name__ == "__main__":
    main()
