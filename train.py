"""SAC + gSDE training, hyperparameters copied from
furuta-master/scripts/configs/algo/sac.yaml.

Usage:
    # (optional) pre-fill replay buffer from classical controller CSV
    python bootstrap_from_csv.py --csv classical_data_real.csv --out buffer.pkl

    # train from scratch
    python train.py --total_timesteps 300000 --replay_buffer buffer.pkl

    # retrain with a softer voltage ceiling (policy must pump across more swings)
    python train.py --max_voltage 9 --save_dir runs/sac_gsde_9v

    # fine-tune an existing model under the new ceiling
    python train.py -
    -max_voltage 9 --init_from runs/sac_gsde/best/best_model.zip \
        --save_dir runs/sac_gsde_9v --total_timesteps 100000
"""

import argparse
import os

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gymnasium.wrappers import TimeLimit

from config import CONSTRAINTS, HARDWARE
from furuta_env import FurutaSim
from qube_dynamics import QubeDynamics


def make_env(max_voltage: float, arm_limit_rad: float, soft_arm_bound: bool = False,
             domain_rand: bool = False, max_steps: int = 500):
    dyn_kwargs = dict(V=max_voltage)
    if domain_rand:
        dyn_kwargs.update(
            Mp_std=HARDWARE['Mp'] * 0.3,
            Lp_std=HARDWARE['Lp'] * 0.15,
            lp_std=HARDWARE['lp'] * 0.15,
            Jp_std=HARDWARE['Jp'] * 0.3,
            Mr_std=HARDWARE['Mr'] * 0.3,
            Lr_std=HARDWARE['Lr'] * 0.15,
            Jr_std=HARDWARE['Jr'] * 0.3,
            Rm_std=HARDWARE['Rm'] * 0.3,
            kt_std=HARDWARE['kt'] * 0.3,
            km_std=HARDWARE['km'] * 0.3,
            Dr_std=HARDWARE['Dr'] * 2.0,
            Dp_std=HARDWARE['Dp'] * 2.0,
        )
    dyn = QubeDynamics(**dyn_kwargs)
    env = FurutaSim(
        dyn=dyn,
        reward="cos_alpha",
        angle_limits=[arm_limit_rad, None],
        speed_limits=[60, 400],
        max_voltage=max_voltage,
        soft_arm_bound=soft_arm_bound,
        action_noise_std=0.05 if domain_rand else 0.0,
    )
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total_timesteps", type=int, default=300_000)
    p.add_argument("--replay_buffer", type=str, default=None,
                   help="Path to pre-filled replay buffer pickle (optional)")
    p.add_argument("--save_dir", type=str, default="runs/sac_gsde")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_voltage", type=float, default=CONSTRAINTS['max_voltage'],
                   help="Ceiling for motor voltage (action=1 maps to this). Lower = gentler.")
    p.add_argument("--init_from", type=str, default=None,
                   help="Optional SAC .zip to warm-start policy weights from (fine-tune).")
    p.add_argument("--arm_limit_deg", type=float, default=90.0,
                   help="Arm soft bound in degrees (hardware wire allowance).")
    p.add_argument("--soft_arm", action="store_true",
                   help="Soft arm bound: clip + penalize instead of terminating.")
    p.add_argument("--domain_rand", action="store_true",
                   help="Randomize physics params each episode for sim-to-real transfer.")
    p.add_argument("--ent_coef", type=float, default=None,
                   help="Fixed entropy coefficient (default: SAC auto-tunes it).")
    p.add_argument("--n_envs", type=int, default=8,
                   help="Number of parallel environments (furuta-master uses 12).")
    p.add_argument("--max_steps", type=int, default=500,
                   help="Max steps per episode (furuta-master uses 400 at 50Hz; 500 at 100Hz ≈ same wall time).")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    arm_limit_rad = float(np.deg2rad(args.arm_limit_deg))

    # Check env with a throwaway instance
    check_env(make_env(args.max_voltage, arm_limit_rad, args.soft_arm, args.domain_rand, args.max_steps))

    # Parallel envs (furuta-master uses 12 SubprocVecEnv — massive speedup)
    vec_env = SubprocVecEnv([
        lambda: make_env(args.max_voltage, arm_limit_rad, args.soft_arm, args.domain_rand, args.max_steps)
        for _ in range(args.n_envs)
    ])

    if args.init_from and os.path.exists(args.init_from):
        print(f"Fine-tuning from {args.init_from} (max_voltage={args.max_voltage})")
        model = SAC.load(args.init_from, env=vec_env, custom_objects={
            "observation_space": vec_env.observation_space,
            "action_space": vec_env.action_space,
        })
    else:
        # Hyperparameters — straight from furuta-master/scripts/configs/algo/sac.yaml
        sac_kwargs = dict(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            tau=0.005,
            gamma=0.99,
            batch_size=256,
            target_update_interval=1,
            learning_starts=500,
            use_sde=True,
            use_sde_at_warmup=True,
            sde_sample_freq=64,
            train_freq=1,
            gradient_steps=-1,
            stats_window_size=10,
            tensorboard_log=args.save_dir,
            seed=args.seed,
            verbose=1,
        )
        if args.ent_coef is not None:
            sac_kwargs["ent_coef"] = args.ent_coef
        model = SAC(**sac_kwargs)

    if args.replay_buffer and os.path.exists(args.replay_buffer):
        print(f"Loading replay buffer from {args.replay_buffer}")
        model.load_replay_buffer(args.replay_buffer)

    eval_env = DummyVecEnv([lambda: make_env(args.max_voltage, arm_limit_rad, args.soft_arm,
                                              args.domain_rand, args.max_steps)])
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_dir, "best"),
        log_path=os.path.join(args.save_dir, "eval"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_cb,
            progress_bar=True,
            reset_num_timesteps=(args.init_from is None),
        )
    except KeyboardInterrupt:
        print("Interrupted — saving current model.")

    model.save(os.path.join(args.save_dir, "model.zip"))
    model.save_replay_buffer(os.path.join(args.save_dir, "buffer.pkl"))
    print("Done.")


if __name__ == "__main__":
    main()
