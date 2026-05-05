"""3D visualization of the trained SAC policy running against FurutaSim.

Left  : 3D view of rotary arm + pendulum, with the pendulum tip's recent trace.
Right : live plots of pendulum angle, arm angle, and motor voltage.

Usage:
    python visualize.py
    python visualize.py --model runs/sac_gsde/best/best_model.zip
    python visualize.py --angle 170                # start pendulum near hanging (deg)
    python visualize.py --no-model                 # passive (0 V) physics
"""

import argparse
import os

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers import TimeLimit
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from config import CONSTRAINTS, HARDWARE
from furuta_env import FurutaSim
from furuta_utils import ALPHA, ALPHA_DOT, THETA, THETA_DOT


matplotlib.rcParams.update({
    'axes.facecolor': '#1a1a2e',
    'figure.facecolor': '#0f0f1a',
    'axes.labelcolor': '#cccccc',
    'xtick.color': '#888888',
    'ytick.color': '#888888',
    'axes.edgecolor': '#333355',
    'grid.color': '#222244',
    'text.color': '#e0e0e0',
})

C_ARM = '#4a9eff'
C_PEND = '#ff4a4a'
C_JOINT = '#ffdd44'
C_TRACE = '#44ffaa'

HISTORY = 200
TRACE_LEN = 80


def get_coords(state):
    """Return (origin, arm_tip, pend_tip) in 3D.
    Convention: theta=0 upright, theta=pi hanging; alpha rotates arm in XY plane.
    """
    arm = state[ALPHA]
    pend = state[THETA]
    Lr = HARDWARE['Lr']
    lp = HARDWARE['lp']

    ax = Lr * np.cos(arm)
    ay = Lr * np.sin(arm)
    az = 0.0

    # Right-sided coordinate system to prevent visual stretching.
    # Pendulum swings along the arm's radius if mounted parallel.
    # If the pendulum is mounted perpendicular to the arm, it swings perpendicularly.
    # Adjust this specific to your custom hardware mount.
    px = ax + lp * np.sin(pend) * np.sin(arm)
    py = ay - lp * np.sin(pend) * np.cos(arm)
    pz = az + lp * np.cos(pend)
    return (0, 0, 0), (ax, ay, az), (px, py, pz)


def run(model_path, start_angle_deg=None, episode_len=5000, no_model=False,
        max_voltage=CONSTRAINTS['max_voltage']):
    env = TimeLimit(
        FurutaSim(reward="cos_alpha",
                  angle_limits=[np.pi / 2, None],
                  speed_limits=[60, 400],
                  max_voltage=max_voltage),
        max_episode_steps=CONSTRAINTS['max_steps'],
    )

    model = None
    if not no_model:
        from sbx import CrossQ
        # Try CrossQ first, fallback if it's actually PyTorch SAC
        try:
            model = CrossQ.load(model_path)
            print(f"Loaded CrossQ model from {model_path}.")
        except Exception:
            from stable_baselines3 import SAC
            model = SAC.load(model_path)
            print(f"Loaded Native SAC model from {model_path}.")

    obs, _ = env.reset()
    if start_angle_deg is not None:
        rad = float(np.radians(start_angle_deg))
        raw = env.unwrapped
        raw._simulation_state = np.zeros(4, dtype=np.float32)
        raw._simulation_state[THETA] = rad
        raw._theta_start = rad
        raw._state = raw._simulation_state.copy()
        raw._state[THETA] = ((raw._state[THETA] + np.pi) % (2 * np.pi)) - np.pi
        obs = raw.get_obs()

    Lr = HARDWARE['Lr']
    lp = HARDWARE['lp']
    limit = Lr + lp + 0.05

    th_hist = [0.0] * HISTORY
    al_hist = [0.0] * HISTORY
    v_hist = [0.0] * HISTORY
    trace_x, trace_y, trace_z = [], [], []

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Furuta Pendulum — SAC Policy Rollout", fontsize=14, color='#e0e0e0')
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.5, 1, 1])

    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax3d.view_init(elev=25, azim=-45)
    ax3d.set_xlim(-limit, limit)
    ax3d.set_ylim(-limit, limit)
    ax3d.set_zlim(-limit, limit)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')

    # base square
    ax3d.plot([0], [0], [0], 's', color='#888', ms=8)

    line_arm, = ax3d.plot([], [], [], '-', color=C_ARM, lw=5)
    line_pen, = ax3d.plot([], [], [], '-', color=C_PEND, lw=4)
    pt_joint, = ax3d.plot([], [], [], 'o', color=C_JOINT, ms=8)
    pt_tip,   = ax3d.plot([], [], [], 'o', color=C_PEND, ms=6)
    line_trace, = ax3d.plot([], [], [], '-', color=C_TRACE, lw=1, alpha=0.5)
    status_text = ax3d.text2D(0.02, 0.97, "", transform=ax3d.transAxes,
                              family='monospace', fontsize=10,
                              verticalalignment='top',
                              bbox=dict(facecolor='#0f0f1a', edgecolor='#333355', alpha=0.8))

    def setup_2d(ax, title, y_label, y_lim, color):
        ax.set_title(title, color='#e0e0e0')
        ax.set_ylabel(y_label)
        ax.set_ylim(y_lim)
        ax.set_xlim(0, HISTORY)
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color='#555', lw=0.5)
        line, = ax.plot([], [], lw=1.5, color=color)
        return line

    ax_th = fig.add_subplot(gs[0, 1])
    ln_th = setup_2d(ax_th, "Pendulum Angle θ", "rad", (-np.pi - 0.2, np.pi + 0.2), C_PEND)
    ax_th.axhline(0, color=C_TRACE, lw=0.7, ls='--', alpha=0.5)  # upright target

    ax_al = fig.add_subplot(gs[1, 1])
    ln_al = setup_2d(ax_al, "Arm Angle α", "rad", (-np.pi / 2 - 0.2, np.pi / 2 + 0.2), C_ARM)
    ax_al.axhline(np.pi / 2, color='#ff4a4a', lw=0.5, ls='--', alpha=0.4)
    ax_al.axhline(-np.pi / 2, color='#ff4a4a', lw=0.5, ls='--', alpha=0.4)

    ax_v = fig.add_subplot(gs[:, 2])
    ln_v = setup_2d(ax_v, "Motor Voltage", "V",
                    (-max_voltage - 1, max_voltage + 1), C_JOINT)

    steps_per_frame = max(1, int(round(0.02 / CONSTRAINTS['dt'])))

    state = {'obs': obs, 'done': False, 'ep': 0, 'ep_r': 0.0, 'ep_steps': 0, 'volt': 0.0}

    def update(_frame):
        for _ in range(steps_per_frame):
            if state['done']:
                state['obs'], _ = env.reset()
                state['done'] = False
                state['ep'] += 1
                state['ep_r'] = 0.0
                state['ep_steps'] = 0
                trace_x.clear(); trace_y.clear(); trace_z.clear()
            if model is None:
                a = np.zeros(1, dtype=np.float32)
            else:
                a, _ = model.predict(state['obs'], deterministic=True)
            state['obs'], r, term, trunc, _ = env.step(a)
            state['done'] = bool(term or trunc)
            state['ep_r'] += r
            state['ep_steps'] += 1
            state['volt'] = float(a[0]) * max_voltage

        s = env.unwrapped._state
        theta = float(s[THETA])
        alpha = float(s[ALPHA])

        origin, arm_tip, pen_tip = get_coords(s)
        line_arm.set_data([origin[0], arm_tip[0]], [origin[1], arm_tip[1]])
        line_arm.set_3d_properties([origin[2], arm_tip[2]])
        line_pen.set_data([arm_tip[0], pen_tip[0]], [arm_tip[1], pen_tip[1]])
        line_pen.set_3d_properties([arm_tip[2], pen_tip[2]])
        pt_joint.set_data([arm_tip[0]], [arm_tip[1]])
        pt_joint.set_3d_properties([arm_tip[2]])
        pt_tip.set_data([pen_tip[0]], [pen_tip[1]])
        pt_tip.set_3d_properties([pen_tip[2]])

        trace_x.append(pen_tip[0])
        trace_y.append(pen_tip[1])
        trace_z.append(pen_tip[2])
        if len(trace_x) > TRACE_LEN:
            trace_x.pop(0); trace_y.pop(0); trace_z.pop(0)
        line_trace.set_data(trace_x, trace_y)
        line_trace.set_3d_properties(trace_z)

        th_hist.append(theta); al_hist.append(alpha); v_hist.append(state['volt'])
        if len(th_hist) > HISTORY:
            th_hist.pop(0); al_hist.pop(0); v_hist.pop(0)
        ln_th.set_data(range(len(th_hist)), th_hist)
        ln_al.set_data(range(len(al_hist)), al_hist)
        ln_v.set_data(range(len(v_hist)), v_hist)

        status_text.set_text(
            f"ep {state['ep']}  step {state['ep_steps']:>4d}\n"
            f"θ (pend) = {np.degrees(theta):+7.1f}°\n"
            f"α (arm)  = {np.degrees(alpha):+7.1f}°\n"
            f"V        = {state['volt']:+6.2f}\n"
            f"reward   = {state['ep_r']:.1f}"
        )
        return line_arm, line_pen, pt_joint, pt_tip, line_trace

    ani = FuncAnimation(fig, update, frames=episode_len, interval=20, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="runs/sac_gsde/best/best_model.zip")
    p.add_argument("--angle", type=float, default=None,
                   help="Start pendulum angle in degrees (0 = upright, 180 = hanging)")
    p.add_argument("--no-model", action="store_true", help="Passive 0 V physics")
    p.add_argument("--frames", type=int, default=5000)
    p.add_argument("--max_voltage", type=float, default=CONSTRAINTS['max_voltage'])
    args = p.parse_args()

    model_file = args.model
    if not args.no_model and not os.path.exists(model_file):
        raise SystemExit(f"Model not found: {model_file}")

    run(model_file, args.angle, args.frames, args.no_model, args.max_voltage)
