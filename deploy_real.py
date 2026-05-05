"""
Hardware Deployment Script.
Loads a trained SAC policy and runs it on the physical Furuta pendulum.
"""

import argparse
from stable_baselines3 import SAC
from furuta_real import FurutaReal
from config import CONSTRAINTS
import numpy as np

def main():
    p = argparse.ArgumentParser(description="Deploy a trained SAC model to the Furuta pendulum.")
    p.add_argument("--port", required=True, 
                   help="Arduino serial port (e.g., /dev/cu.usbmodem1401)")
    p.add_argument("--model", required=True, 
                   help="Path to the trained model.zip (e.g., runs/sac_real/ckpts/sac_real_50000_steps.zip)")
    p.add_argument("--action_smoothing", type=float, default=0.3,
                   help="EMA factor for voltage command (must match training)")
    p.add_argument("--max_voltage", type=float, default=CONSTRAINTS['max_voltage'],
                   help="Voltage ceiling (must match training)")
    args = p.parse_args()

    print(f"🔌 Connecting to hardware on {args.port}...")
    # Initialize the hardware environment
    env = FurutaReal(
        port=args.port,
        action_smoothing=args.action_smoothing,
        max_voltage=args.max_voltage,
        # It's generally a good idea to keep soft bounds on during deployment
        # to prevent the arm from smashing into physical limits if the policy slips up.
        soft_arm_bound=True, 
    )

    #print(f"🧠 Loading trained model from {args.model}...")
    # Load the model. We don't need a dummy vector env for simple prediction.
    model = SAC.load(args.model)

    print("\n🚀 Starting deployment loop. Press Ctrl+C to stop.\n")
    
    # The reset() function in FurutaReal will automatically zero the motor, 
    # recenter the arm, and wait for the pendulum to hang perfectly still.
    obs, _ = env.reset()

    BALANCE_THRESH = np.deg2rad(10.0)

    try:
        while True:
            # Extract the raw angles from the environment's internal state
            # state = [theta_arm, theta_pend, omega_arm, omega_pend]
            theta_pend = env._state[1] 
            print(f"State vector:",env._state[0], env._state[1], env._state[2], env._state[3], flush=True)

            # Check if pendulum is within 10 degrees of upright (0 radians)
            if abs(theta_pend) <= BALANCE_THRESH:
                # ==========================================
                # REGIME 1: FULL STATE FEEDBACK (Balancing)
                # ==========================================
                
                # Replace these values with your actual calculated K matrix!
                # Order matters: [k_arm, k_pend, k_arm_dot, k_pend_dot]
                # K = np.array([0.5 * -4.4681, 1.0 * -39.5037, 0.5 * -1.6210, 0.75 * -5.0402]) 
                # k_prev = 0.5 * -0.4387

                # K = np.array([0.5 * -4.4681, 1.55 * -39.5037, 0.5 * -1.6210, 0.6 * -5.0402]) 
                # k_prev = 0.75 * -0.4387

                # K = np.array([0.4 * -4.4681, 1.38 * -39.5037, 0.5 * -1.6210, 0.5 * -5.0402]) 
                # k_prev = 0.75 * -0.4387

                K = np.array([0.2 * -4.4681, 1.38 * -39.5037, 0.5 * -1.6210, 0.5 * -5.0402]) 
                k_prev = 0.75 * -0.4387

                # K = np.array([0.5 * -4.4681, 1.7 * -39.5037, 0.5 * -1.6210, 0.5 * -5.0402]) 
                # k_prev = 0
                
                # The control law: u = -Kx
                # np.dot multiplies the arrays element-wise and sums them up
                voltage_command = -np.dot(K, env._state) - k_prev * env._prev_volts
                
                # ==========================================================
                # CRITICAL SCALING STEP:
                # If your K matrix was calculated to output raw VOLTS (e.g. -12V to 12V),
                # you MUST scale it down to the [-1.0, 1.0] range the environment expects.
                # ==========================================================
                action_scalar = voltage_command / env.max_voltage
                
                # Clip it for safety just in case the math blows up
                action_scalar = np.clip(action_scalar, -1.0, 1.0)
                action = np.array([action_scalar], dtype=np.float32)
                
                #print(f"⚖️ Balancing (LQR): {np.rad2deg(theta_pend):.1f}° | Arm: {np.rad2deg(env._state[0]):.0f}°", end="\r")

            else:
                # ==========================================
                # REGIME 2: RL POLICY (Swing-up)
                # ... (keep the same as before) ...
                action, _states = model.predict(obs, deterministic=True)
                
                # Optional: Print to terminal
                #print(f"🚀 Swinging:  {np.rad2deg(theta_pend):.1f}°", end="\r")

            # Execute the action on hardware
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("\n⚠️ Episode terminated (limit reached). Resetting hardware...")
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted by user.")
    finally:
        print("⚡ Shutting down hardware safely (0 Volts)...")
        env.close()

if __name__ == "__main__":
    main()