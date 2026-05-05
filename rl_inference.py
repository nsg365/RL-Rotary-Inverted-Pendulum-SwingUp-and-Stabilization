import argparse
import numpy as np
from stable_baselines3 import SAC
from furuta_real import FurutaReal
from config import CONSTRAINTS

def main():
    p = argparse.ArgumentParser(description="Deploy a trained SAC model to the Furuta pendulum.")
    p.add_argument("--port", required=True, help="Arduino serial port")
    p.add_argument("--model", required=True, help="Path to the trained model.zip")
    p.add_argument("--action_smoothing", type=float, default=0.3)
    p.add_argument("--max_voltage", type=float, default=CONSTRAINTS['max_voltage'])
    args = p.parse_args()

    print(f"Connecting to hardware on {args.port}...")
    env = FurutaReal(
        port=args.port,
        action_smoothing=args.action_smoothing,
        max_voltage=args.max_voltage,
        soft_arm_bound=True, 
    )

    print(f"Loading trained model from {args.model}...")
    model = SAC.load(args.model)

    print("\nStarting deployment loop. Press Ctrl+C to stop.\n")
    obs, _ = env.reset()

    BALANCE_THRESH = np.deg2rad(15.0)

    try:
        while True:
            # --- Coordinate Transformation for RL ---
            # Negate pendulum angle (sin(-x) = -sin(x)) and pendulum velocity
            mod_obs = obs.copy()
            # mod_obs[3] = -mod_obs[3] # sin(pend)
            # mod_obs[5] = -mod_obs[5] # pend_vel

            # Get action from the model using the modified observation
            action, _states = model.predict(mod_obs, deterministic=True)
            
            # Send action to Arduino (Arduino decides whether to use it or override it)
            obs, reward, terminated, truncated, info = env.step(action)
            
            theta_pend = env._state[1]
            theta_arm = env._state[0]
            
            if abs(theta_pend) <= BALANCE_THRESH:
                print(f"Balancing (LQR) | Pend: {np.rad2deg(theta_pend):5.1f} deg | Arm: {np.rad2deg(theta_arm):5.0f} deg", end="\r")
            else:
                print(f"Swinging (RL)   | Pend: {np.rad2deg(theta_pend):5.1f} deg | Arm: {np.rad2deg(theta_arm):5.0f} deg", end="\r")
            
            if terminated or truncated:
                print("\nEpisode terminated. Resetting hardware...")
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\nDeployment interrupted by user.")
    finally:
        print("Shutting down hardware safely...")
        env.close()

if __name__ == "__main__":
    main()