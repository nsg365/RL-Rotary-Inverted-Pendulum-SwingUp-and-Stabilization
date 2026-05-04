"""Hardware fine-tuning. Loads a sim-trained SAC+gSDE policy (first run) or
auto-resumes from the latest hardware checkpoint (subsequent runs).

Trains indefinitely — Ctrl+C saves and exits cleanly. Next invocation picks
up from the last checkpoint (model + replay buffer) automatically.

Usage:
    # first run (seed from sim model + sim buffer)
    python train_real.py --port /dev/cu.usbmodem1401 \
        --model runs/sac_gsde/best/best_model.zip \
        --replay_buffer runs/sac_gsde/buffer.pkl

    # any subsequent run (auto-resumes)
    python train_real.py --port /dev/cu.usbmodem1401
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import CONSTRAINTS
from furuta_real import FurutaReal


def latest_checkpoint(ckpt_dir: str):
    """Return (model_path, buffer_path, steps) for the newest sac_real_*_steps.zip, or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    models = glob.glob(os.path.join(ckpt_dir, "sac_real_*_steps.zip"))
    if not models:
        return None

    def step_of(p):
        m = re.search(r"_(\d+)_steps\.zip$", p)
        return int(m.group(1)) if m else -1

    latest = max(models, key=step_of)
    steps = step_of(latest)
    buf = os.path.join(ckpt_dir, f"sac_real_replay_buffer_{steps}_steps.pkl")
    return latest, (buf if os.path.exists(buf) else None), steps


def load_sac_model(model_path: str, env, fixed_ent_coef: float | None = None, device: str = "auto"):
    """Load SAC, repairing fixed-entropy checkpoints saved with stale auto metadata."""
    try:
        return SAC.load(model_path, env=env, device=device)
    except (KeyError, ValueError) as exc:
        if fixed_ent_coef is None:
            raise
        print(
            "Normal SAC load failed; retrying as a fixed-entropy checkpoint "
            f"(ent_coef={fixed_ent_coef}). Original error: {exc}"
        )
        return SAC.load(
            model_path,
            env=env,
            device=device,
            custom_objects={"ent_coef": float(fixed_ent_coef)},
        )


class EpisodeRewardCallback(BaseCallback):
    """Print episode reward after each episode so we can track learning."""
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._ep_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_r = info["episode"]["r"]
                ep_l = info["episode"]["l"]
                self._ep_rewards.append(ep_r)
                avg = np.mean(self._ep_rewards[-10:])
                print(f"  ▸ ep {len(self._ep_rewards):>4d}  "
                      f"reward={ep_r:7.1f}  len={ep_l:>4d}  "
                      f"avg10={avg:7.1f}")
        return True


def fix_entropy_coefficient(model, ent_coef: float):
    """Switch SAC to a constant entropy coefficient and save that mode correctly."""
    value = float(ent_coef)
    model.ent_coef = value
    model.ent_coef_optimizer = None
    model.ent_coef_tensor = torch.tensor(value, device=model.device)
    model.log_ent_coef = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="Arduino serial port (e.g. /dev/cu.usbmodem1401)")
    p.add_argument("--model", default=None,
                   help="Sim-trained model.zip to seed from (ignored if a hardware checkpoint exists)")
    p.add_argument("--replay_buffer", default=None,
                   help="Sim replay buffer pickle to seed from (ignored if a hardware checkpoint exists)")
    p.add_argument("--total_timesteps", type=int, default=10_000_000,
                   help="Upper bound. Training is meant to be Ctrl+C-stopped; set high.")
    p.add_argument("--save_dir", default="runs/sac_real")
    p.add_argument("--save_freq", type=int, default=2_000,
                   help="Checkpoint every N env steps")
    p.add_argument("--action_smoothing", type=float, default=0.3,
                   help="EMA factor for voltage command (0 = none, 1 = frozen)")
    p.add_argument("--max_voltage", type=float, default=CONSTRAINTS['max_voltage'],
                   help="Voltage ceiling (must match what the model was trained with)")
    p.add_argument("--arm_limit_deg", type=float, default=135.0,
                   help="Arm soft bound (degrees). Wider than sim (90) lets the policy explore.")
    p.add_argument("--gradient_steps", type=int, default=1,
                   help="SAC gradient steps per env step (1 is safer on hardware; -1 = one-per-step)")
    p.add_argument("--reset_exploration", action="store_true",
                   help="When seeding from sim, reset gSDE std + entropy so the policy re-explores.")
    p.add_argument("--soft_arm", action="store_true",
                   help="Soft arm bound: clip + penalize instead of terminating episode.")
    p.add_argument("--ent_coef", type=float, default=None,
                   help="Fixed entropy coefficient (e.g. 0.05). Overrides auto-tuning.")
    p.add_argument("--device", type=str, default="auto",
                   help="PyTorch device: 'auto', 'cpu', 'cuda', or 'mps' (Apple Silicon GPU).")
    args = p.parse_args()

    ckpt_dir = os.path.join(args.save_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    arm_limit_rad = float(np.deg2rad(args.arm_limit_deg))
    env = FurutaReal(
        port=args.port,
        action_smoothing=args.action_smoothing,
        max_voltage=args.max_voltage,
        angle_limits=[arm_limit_rad, None],
        soft_arm_bound=args.soft_arm,
    )
    env = TimeLimit(env, max_episode_steps=CONSTRAINTS['max_steps'])
    env = Monitor(env)  # enables episode reward/length tracking
    vec_env = DummyVecEnv([lambda: env])

    resume = latest_checkpoint(ckpt_dir)
    if resume is not None:
        model_path, buf_path, steps = resume
        print(f"Resuming from hardware checkpoint: {model_path} ({steps} steps)")
        model = load_sac_model(model_path, env=vec_env, fixed_ent_coef=args.ent_coef, device=args.device)
        if buf_path:
            model.load_replay_buffer(buf_path)
            print(f"Loaded replay buffer: {buf_path}")
        reset_num_timesteps = False
        seeding_from_sim = False
    else:
        assert args.model is not None, (
            "No hardware checkpoint found and --model not given. "
            "First run needs --model pointing to the sim-trained model."
        )
        print(f"Seeding from sim model: {args.model}")
        model = load_sac_model(args.model, env=vec_env, device=args.device)
        if args.replay_buffer and os.path.exists(args.replay_buffer):
            model.load_replay_buffer(args.replay_buffer)
            print(f"Loaded sim replay buffer: {args.replay_buffer}")
        reset_num_timesteps = True
        seeding_from_sim = True

    model.gradient_steps = int(args.gradient_steps)
    print(f"gradient_steps set to {model.gradient_steps}")

    # Re-enable exploration if seeding from sim.
    if seeding_from_sim and args.reset_exploration:
        print("Resetting gSDE std <- -1.0 (std ~ 0.37)")
        with torch.no_grad():
            if hasattr(model.actor, "log_std"):
                model.actor.log_std.data.fill_(-1.0)

    # Fixed entropy coefficient: disable auto-tuning, use a constant.
    if args.ent_coef is not None:
        print(f"Fixing ent_coef = {args.ent_coef} (auto-tuning disabled)")
        fix_entropy_coefficient(model, args.ent_coef)

    ckpt_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=ckpt_dir,
        name_prefix="sac_real",
        save_replay_buffer=True,
    )
    reward_cb = EpisodeRewardCallback()

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[ckpt_cb, reward_cb],
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving final checkpoint.")
    finally:
        env.close()

    # Save a final snapshot on top of whatever the periodic callback produced.
    final_steps = int(model.num_timesteps)
    model.save(os.path.join(ckpt_dir, f"sac_real_{final_steps}_steps.zip"))
    model.save_replay_buffer(os.path.join(ckpt_dir, f"sac_real_replay_buffer_{final_steps}_steps.pkl"))
    model.save(os.path.join(args.save_dir, "model.zip"))
    print(f"Saved final checkpoint at {final_steps} steps.")


if __name__ == "__main__":
    main()
