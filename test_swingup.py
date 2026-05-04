"""Sanity check: can the pendulum swing up in our simulation?
Uses aggressive energy pumping — full voltage, alternating direction."""

import numpy as np
from gymnasium.wrappers import TimeLimit
from furuta_env import FurutaSim
from furuta_utils import THETA, THETA_DOT, ALPHA, ALPHA_DOT

env = FurutaSim(
    reward="cos_alpha",
    angle_limits=[np.deg2rad(150), None],
    speed_limits=[60, 400],
    soft_arm_bound=True,
)
env = TimeLimit(env, max_episode_steps=2000)

obs, _ = env.reset()

best_pend_angle = np.pi

for step in range(2000):
    s = env.unwrapped._state
    theta = s[THETA]
    theta_dot = s[THETA_DOT]

    # Near upright: PD balance
    if abs(theta) < 0.3:
        action = np.clip(-3.0 * theta - 0.5 * theta_dot, -1.0, 1.0)
    else:
        # Energy pumping: FULL voltage in the direction that adds energy
        # Classic swing-up: u = sign(theta_dot * cos(theta))
        pump_sign = np.sign(theta_dot * np.cos(theta))
        if abs(theta_dot) < 0.5:
            # Pendulum barely moving — just oscillate the arm to kick it
            pump_sign = 1.0 if (step // 50) % 2 == 0 else -1.0
        action = -pump_sign * 1.0  # FULL voltage

    obs, r, term, trunc, _ = env.step(np.array([float(action)]))
    best_pend_angle = min(best_pend_angle, abs(env.unwrapped._state[THETA]))

    if step % 100 == 0:
        print(f"step {step:4d}: theta={np.degrees(s[THETA]):7.1f}°  "
              f"theta_dot={s[THETA_DOT]:7.1f}  "
              f"arm={np.degrees(s[ALPHA]):7.1f}°  "
              f"arm_dot={s[ALPHA_DOT]:7.1f}  "
              f"action={action:+.1f}")

    if term or trunc:
        print(f"Episode ended at step {step} ({'terminated' if term else 'truncated'})")
        break

print(f"\nClosest to upright: {np.degrees(best_pend_angle):.1f}° from vertical")
if best_pend_angle < np.deg2rad(30):
    print("SUCCESS — swing-up is physically possible!")
elif best_pend_angle < np.deg2rad(90):
    print("PARTIAL — pendulum got past horizontal but didn't reach upright")
else:
    print("FAILED — pendulum never got past horizontal. Physics/params may be wrong.")
