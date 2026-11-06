# RL Swing-Up and Stabilization of a Furuta Pendulum

A low-cost rotary inverted (Furuta) pendulum that swings itself up from hanging and balances upright.
A **Soft Actor-Critic (SAC)** policy, trained in simulation and fine-tuned on the real rig, does the swing-up.
A **linear state-feedback (LQR) controller running on the Arduino** does the balancing.

- **Result:** 95 of 100 cold-start swing-ups succeeded, and every successful one held balance for the full 60 s window.
- **Video:** https://youtu.be/YtkpQR3PJXU
- **Paper:** *Reinforcement Learning-Aided Swing Up and Stabilization of a Furuta Pendulum* — N. Sagar G, E. Ben, A. Pandav, A. Kulkarni, G. K. Kamath M, IIT Dharwad.

---

## How it works

Two controllers run at the same time, and the **Arduino** decides which one drives the motor, based on how far the pendulum is from upright:

| Pendulum distance from upright | Motor voltage comes from |
|---|---|
| more than 20° | RL policy (computed on the laptop, sent over USB) |
| 10° – 20° | blend: 0.4 × RL + 0.6 × LQR |
| less than 10° | LQR only (computed on the Arduino) |

```
 Laptop (Python)                              Arduino Mega (prop_on_ard.ino)
 ┌──────────────────────────┐   state @100Hz  ┌─────────────────────────────────┐
 │ rl_inference.py          │ <────────────── │ read encoders, filter velocity  │
 │  SAC policy -> RL voltage│ ──────────────> │ compute LQR, pick/blend voltage │──> motor
 └──────────────────────────┘   RL voltage    └─────────────────────────────────┘
```

Because the balancing loop never leaves the Arduino, USB delays and laptop hiccups can't knock the pendulum over once it's up.

---

## Repository layout

| File | What it does |
|---|---|
| [`prop_on_ard/prop_on_ard.ino`](prop_on_ard/prop_on_ard.ino) | **Arduino firmware.** Encoders, velocity filter, LQR, RL/LQR arbitration, dead-zone compensation, 100 ms watchdog. |
| [`rl_inference.py`](rl_inference.py) | **Run this for the demo.** Loads the trained policy and streams its voltage to the Arduino. |
| [`furuta_real.py`](furuta_real.py) | Hardware environment: serial protocol, angle conventions, automatic reset routine. |
| [`furuta_env.py`](furuta_env.py) | Simulation environment, reward, observation format (shared base class with the hardware env). |
| [`qube_dynamics.py`](qube_dynamics.py) | Equations of motion used by the simulator, with optional parameter randomization. |
| [`config.py`](config.py) | Physical constants of the rig, 12 V limit, 100 Hz control rate. |
| [`furuta_utils.py`](furuta_utils.py) | State index names and small helpers. |
| [`train.py`](train.py) | Train SAC in simulation. |
| [`train_real.py`](train_real.py) | Fine-tune SAC on the real rig (auto-resumes from the latest checkpoint). |
| [`eval.py`](eval.py), [`visualize.py`](visualize.py) | Run a policy in simulation (text summary / 3D animation with live plots). |
| [`probe_hardware.py`](probe_hardware.py) | Prints live encoder readings with the motor off, for checking the wiring. |
| `runs/sac_real/ckpts/sac_real_407500_steps.zip` | **The trained policy used in the demo.** |

---

## Hardware

**Parts**
- Arduino Mega 2560
- DC motor driving the arm, through a PWM + DIR H-bridge, on a 12 V supply
- Two 2000-count-per-revolution quadrature encoders: one on the arm, one on the pendulum pivot
- USB cable to the laptop

**Wiring** (pin numbers are set at the top of `prop_on_ard.ino`)

| Signal | Mega pin |
|---|---|
| Arm encoder A / B | 20 / 21 |
| Pendulum encoder A / B | 19 / 18 |
| Motor driver PWM | 5 |
| Motor driver DIR | 4 |
| Ground | shared between Arduino, encoders and the 12 V supply |

**Direction signs.** `MOTOR_SIGN`, `ARM_SIGN` and `PEND_SIGN` at the top of the sketch fix direction mismatches in software, so you never need to rewire.
The current rig uses `MOTOR_SIGN = -1.0`.
If a positive voltage moves the arm toward a *negative* arm angle, flip `MOTOR_SIGN` and reflash.

---

## Setup

1. Install the [Arduino IDE](https://www.arduino.cc/en/software) and select the board **Arduino Mega or Mega 2560**.
2. Install Python 3 and the dependencies:
   ```
   pip install -r requirements.txt
   ```

---

## Running the demo

1. **Flash** `prop_on_ard/prop_on_ard.ino` to the Mega.
2. **Quit the Arduino IDE completely** (Cmd+Q on a Mac). If it stays open it can grab the serial port, and the Python script will just wait forever.
3. **Let the pendulum hang perfectly still and centre the arm.** The encoders zero when the board starts, and starting the script restarts the board, so this position becomes "zero".
4. **Find the port:**
   ```
   ls /dev/cu.usbmodem*
   ```
5. **Run:**
   ```
   python3 rl_inference.py --port /dev/cu.usbmodemXXXX \
       --model runs/sac_real/ckpts/sac_real_407500_steps.zip
   ```
6. **Stop** with Ctrl+C. The script sends 0 V before exiting, and the Arduino also cuts the motor by itself if it hears nothing for 100 ms.

**What you'll see:** the arm recentres, the pendulum settles, and the policy starts pumping it up.
The terminal shows `Swinging (RL)` or `Balancing (LQR)` with live angles.
The arm may swing past 150° during swing-up; that's allowed and doesn't interrupt anything.
If the pendulum falls after having been balanced, the script prints `Pendulum fell. Resetting hardware...`, recentres the arm and tries again.

> Keep an eye on the cables during the first few runs. Nothing stops the arm from winding them up if the swing-up keeps going one way.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Stuck at `Starting deployment loop`, nothing moves | The Arduino isn't sending data. Quit the Arduino IDE, unplug and replug the USB, and run again. |
| Arm drives off to one side and stays there | Motor direction is reversed. Flip `MOTOR_SIGN` in `prop_on_ard.ino` and reflash. |
| Angles look wrong or have the wrong sign | Run `python3 probe_hardware.py --port /dev/cu.usbmodemXXXX` (motor off), move the arm and pendulum by hand, and check the readings. This shows the raw firmware angle: hanging should read about 0, upright about ±180°. |
| Arm slowly drifts or sways while balancing | Expected to a degree (motor dead zone). Booting with the pendulum completely still helps. The arm gains are `k1`, `k2` in the sketch. |

---

## Tuning knobs

All in `prop_on_ard.ino` (reflash after changing):

| Setting | Current value | Meaning |
|---|---|---|
| `k1`, `k2`, `k3`, `k4` | LQR gains × 0.5, 0.5, 1.0, 0.75 | Gains on arm angle, arm velocity, pendulum angle, pendulum velocity |
| `BLEND_INNER`, `BLEND_OUTER` | 10°, 20° | Edges of the RL/LQR blend band |
| `alpha` | 0.4 | RL share inside the blend band |
| `DEADZONE`, `THRESH` | 0.5 V | Motor static-friction compensation |

---

## Training (optional)

The demo only needs the saved policy. These steps reproduce it.

**1. Simulation** — SAC with gSDE, parameter randomization, arm angle soft-bounded at 150°:
```
python3 train.py --arm_limit_deg 150 --soft_arm --domain_rand
```
Models are saved to `runs/sac_gsde/`.

**2. Hardware fine-tuning** — flash `prop_on_ard.ino`, then:
```
# first run: start from the simulation model
python3 train_real.py --port /dev/cu.usbmodemXXXX \
    --model runs/sac_gsde/best/best_model.zip --arm_limit_deg 150 --soft_arm --ent_coef 0.05

# later runs: resumes from the newest checkpoint in runs/sac_real/ckpts/
python3 train_real.py --port /dev/cu.usbmodemXXXX --arm_limit_deg 150 --soft_arm --ent_coef 0.05
```
- Training runs until Ctrl+C and saves a checkpoint (model + replay buffer) every 2000 steps.
- `--arm_limit_deg 150` must match the model, or it won't load.
- `--ent_coef 0.05` fixes the exploration level; automatic tuning broke checkpoint reloading.
- Within 20° of upright the Arduino's LQR takes over, so the policy only learns the swing-up part.

**Running a policy in simulation:**
```
python3 visualize.py --model runs/sac_real/ckpts/sac_real_407500_steps.zip
```

---

## Conventions

| Quantity | Convention |
|---|---|
| Pendulum angle θ | 0 = upright, ±π = hanging |
| Arm angle α | 0 = position at startup |
| Policy input (7 values) | `[α, cos α, sin α, cos θ, sin θ, α̇, θ̇]`, with α clipped to ±150° |
| Policy output | one number in [−1, 1], scaled to ±12 V |
| Control rate | 100 Hz |

**Serial protocol** (500000 baud): the Arduino sends a `0xABCD` sync word followed by four float32 values (pendulum angle, arm angle, pendulum velocity, arm velocity); the laptop sends one float32 voltage.

---

## Saved models

| Path | Description |
|---|---|
| `runs/sac_real/ckpts/sac_real_407500_steps.zip` | Final hardware-fine-tuned policy (used in the demo and paper) |
| `runs/sac_real/ckpts/sac_real_replay_buffer_407500_steps.pkl` | Matching replay buffer, for resuming `train_real.py` |
| `runs/sac_real/model.zip` | Last model saved by `train_real.py` |
| `runs/sac_real/ckpts/sac_real_12027_steps.zip` | Early hardware checkpoint |
