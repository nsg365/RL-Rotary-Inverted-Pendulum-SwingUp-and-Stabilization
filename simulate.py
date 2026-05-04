"""Live 2D visualization of the trained SAC policy running against FurutaSim.

Two panels:
    left  : top-down view of the rotary arm (shows +-90 deg wire bound)
    right : side view of the pendulum (theta=0 upright, +-pi hanging)

Usage:
    python simulate.py                                # use best model
    python simulate.py --model runs/sac_gsde/model.zip
    python simulate.py --save rollout.mp4             # write video, no window
    python simulate.py --no-model                     # passive (0 V) physics only
"""

import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import TimeLimit
from matplotlib.animation import FuncAnimation

from config import CONSTRAINTS, HARDWARE
from furuta_env import FurutaSim
from furuta_utils import ALPHA, THETA


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/sac_gsde/best/best_model.zip")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--save", default=None, help="save to mp4 (optional)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--no-model", action="store_true", help="run passive physics with 0 V")
    p.add_argument("--max_voltage", type=float, default=CONSTRAINTS['max_voltage'])
    p.add_argument("--arm_limit_deg", type=float, default=360.0)
    p.add_argument("--soft_arm", action="store_true")
    p.add_argument("--max_steps", type=int, default=500)
    args = p.parse_args()

    arm_limit_rad = float(np.deg2rad(args.arm_limit_deg))
    env = TimeLimit(
        FurutaSim(reward="cos_alpha",
                  angle_limits=[arm_limit_rad, None],
                  speed_limits=[60, 400],
                  max_voltage=args.max_voltage,
                  soft_arm_bound=args.soft_arm),
        max_episode_steps=args.max_steps,
    )

    model = None
    if not args.no_model:
        from stable_baselines3 import SAC
        model = SAC.load(args.model)

    Lr = HARDWARE['Lr']
    Lp = HARDWARE['Lp']

    fig, (ax_arm, ax_pend) = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.suptitle("Furuta Pendulum — Simulated Rollout", fontsize=13)

    # --- Arm (top-down) ---
    ax_arm.set_aspect('equal')
    ax_arm.set_xlim(-Lr * 1.4, Lr * 1.4)
    ax_arm.set_ylim(-Lr * 1.4, Lr * 1.4)
    ax_arm.set_title(f"Arm (top-down)   |  Lr = {Lr*100:.1f} cm")
    ax_arm.grid(alpha=0.2)
    # wire bound arc (+-90 deg)
    arc = mpatches.Arc((0, 0), 2*Lr*1.2, 2*Lr*1.2,
                       angle=0, theta1=-90, theta2=90,
                       linestyle='--', edgecolor='red', alpha=0.5)
    ax_arm.add_patch(arc)
    ax_arm.text(Lr*1.25, 0, ' +90°', color='red', alpha=0.6, va='center')
    ax_arm.text(-Lr*1.25, 0, '-90° ', color='red', alpha=0.6, va='center', ha='right')
    arm_line, = ax_arm.plot([], [], 'b-', lw=4)
    arm_tip,  = ax_arm.plot([], [], 'bo', ms=10)
    ax_arm.plot([0], [0], 'ks', ms=8)

    # --- Pendulum (side view) ---
    ax_pend.set_aspect('equal')
    ax_pend.set_xlim(-Lp * 1.4, Lp * 1.4)
    ax_pend.set_ylim(-Lp * 1.4, Lp * 1.4)
    ax_pend.set_title(f"Pendulum (side view)   |  Lp = {Lp*100:.1f} cm")
    ax_pend.grid(alpha=0.2)
    ax_pend.axhline(0, color='gray', lw=0.3)
    ax_pend.axvline(0, color='gray', lw=0.3)
    # Upright marker (target)
    ax_pend.plot([0, 0], [0, Lp], 'g--', lw=0.8, alpha=0.4)
    ax_pend.text(0, Lp*1.08, 'upright', color='green', alpha=0.6, ha='center', fontsize=9)
    pend_line, = ax_pend.plot([], [], 'g-', lw=4)
    pend_tip,  = ax_pend.plot([], [], 'go', ms=10)
    ax_pend.plot([0], [0], 'ks', ms=8)

    info = ax_pend.text(0.02, 0.98, '', transform=ax_pend.transAxes,
                        va='top', fontsize=10, family='monospace',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # --- State ---
    obs, _ = env.reset()
    st = dict(obs=obs, done=False, ep=0, ep_r=0.0, ep_steps=0, volt=0.0, status="")

    steps_per_frame = max(1, int(round(1.0 / args.fps / CONSTRAINTS['dt'])))

    def step_once():
        if st['done']:
            st['status'] = f"ep {st['ep']} done ({st['ep_steps']} steps, R={st['ep_r']:.0f})"
            st['ep'] += 1
            if st['ep'] >= args.episodes:
                return False
            st['obs'], _ = env.reset()
            st['done'] = False
            st['ep_r'] = 0.0
            st['ep_steps'] = 0
        if model is None:
            a = np.zeros(1, dtype=np.float32)
        else:
            a, _ = model.predict(st['obs'], deterministic=True)
        st['obs'], r, term, trunc, _ = env.step(a)
        st['done'] = bool(term or trunc)
        st['ep_r'] += r
        st['ep_steps'] += 1
        st['volt'] = float(a[0]) * args.max_voltage
        return True

    def update(_frame):
        for _ in range(steps_per_frame):
            if not step_once():
                break

        arm = float(env.unwrapped._state[ALPHA])
        pend = float(env.unwrapped._state[THETA])

        arm_line.set_data([0, Lr * np.cos(arm)], [0, Lr * np.sin(arm)])
        arm_tip.set_data([Lr * np.cos(arm)], [Lr * np.sin(arm)])

        # theta = 0 upright => tip above pivot; theta = pi hanging => tip below
        pend_line.set_data([0, Lp * np.sin(pend)], [0, Lp * np.cos(pend)])
        pend_tip.set_data([Lp * np.sin(pend)], [Lp * np.cos(pend)])

        info.set_text(
            f"ep {st['ep']}  step {st['ep_steps']:>4d}\n"
            f"θ (pend) = {np.degrees(pend):+7.1f}°\n"
            f"α (arm)  = {np.degrees(arm):+7.1f}°\n"
            f"V        = {st['volt']:+6.2f}\n"
            f"reward   = {st['ep_r']:.1f}\n"
            f"{st['status']}"
        )
        return arm_line, arm_tip, pend_line, pend_tip, info

    total_frames = args.episodes * CONSTRAINTS['max_steps'] // steps_per_frame + 10
    ani = FuncAnimation(fig, update, frames=total_frames,
                        interval=1000 / args.fps, blit=False, repeat=False)

    if args.save:
        ani.save(args.save, fps=args.fps, dpi=120)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
