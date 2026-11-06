# RL Swing-Up and Stabilization of a Furuta Pendulum

A low-cost rotary inverted (Furuta) pendulum that swings itself up from hanging and balances upright.
A **Soft Actor-Critic (SAC)** policy, trained in simulation and fine-tuned on the real rig, does the swing-up.
A **proportional controller running on the Arduino** does the balancing.

- **Result:** 95 of 100 cold-start swing-ups succeeded, and every successful one held balance for the full 60 s window.
- **Video:** https://youtu.be/YtkpQR3PJXU
- **Paper:** *Reinforcement Learning-Aided Swing Up and Stabilization of a Furuta Pendulum* — N. Sagar G, E. Ben, A. Pandav, A. Kulkarni, G. K. Kamath M, IIT Dharwad.

---

## How it works

Two controllers run at the same time, and the **Arduino** decides which one drives the motor, based on how far the pendulum is from upright:

| Pendulum distance from upright | Motor voltage comes from |
|---|---|
| more than 20° | RL policy (computed on the laptop, sent over USB) |
| 10° – 20° | blend: 0.4 × RL + 0.6 × proportional |
| less than 10° | proportional controller only (computed on the Arduino) |

```
 Laptop (Python)                              Arduino Mega (prop_on_ard.ino)
 ┌──────────────────────────┐   state @100Hz  ┌─────────────────────────────────┐
 │ rl_inference.py          │ <────────────── │ read encoders, filter velocity  │
 │  SAC policy -> RL voltage│ ──────────────> │ proportional ctrl, blend voltage│──> motor
 └──────────────────────────┘   RL voltage    └─────────────────────────────────┘
```

Because the balancing loop never leaves the Arduino, USB delays and laptop hiccups can't knock the pendulum over once it's up.

---

## Repository layout

```
RL-Rotary-Inverted-Pendulum-SwingUp-and-Stabilization/
├── deploy/        run the real rig
├── training/      train the policy (simulation and hardware)
├── sim/           try a policy in simulation
├── furuta/        shared code used by everything above (never run directly)
├── firmware/      Arduino sketch
└── models/        saved policies
```

| Folder | File | What it does |
|---|---|---|
| `deploy/` | [`rl_inference.py`](deploy/rl_inference.py) | **Run this for the demo.** Loads the trained policy and streams its voltage to the Arduino. |
| | [`probe_hardware.py`](deploy/probe_hardware.py) | Prints live encoder readings with the motor off, for checking the wiring. |
| `training/` | [`train.py`](training/train.py) | Train SAC in simulation. |
| | [`train_real.py`](training/train_real.py) | Fine-tune SAC on the real rig (auto-resumes from the latest checkpoint). |
| `sim/` | [`eval.py`](sim/eval.py) | Run a policy in simulation and print a summary. |
| | [`visualize.py`](sim/visualize.py) | 3D animation of a policy in simulation, with live plots. |
| `furuta/` | [`real_env.py`](furuta/real_env.py) | Hardware environment: serial protocol, angle conventions, automatic reset routine. |
| | [`sim_env.py`](furuta/sim_env.py) | Simulation environment, reward, observation format (shared base class with the hardware env). |
| | [`dynamics.py`](furuta/dynamics.py) | Equations of motion used by the simulator, with optional parameter randomization. |
| | [`config.py`](furuta/config.py) | Physical constants of the rig, 12 V limit, 100 Hz control rate. |
| | [`utils.py`](furuta/utils.py) | State index names and small helpers. |
| `firmware/` | [`prop_on_ard/prop_on_ard.ino`](firmware/prop_on_ard/prop_on_ard.ino) | **Arduino firmware.** Encoders, velocity filter, proportional controller, RL/proportional arbitration, dead-zone compensation, 100 ms watchdog. |
| `models/` | `sac_real/ckpts/sac_real_407500_steps.zip` | **The trained policy used in the demo.** |

**Run every command from the top-level project folder**, using the `python3 -m folder.script` form shown below (for example `python3 -m deploy.rl_inference`).
Running a script by its path, like `python3 deploy/rl_inference.py`, fails with `ModuleNotFoundError: No module named 'furuta'`.

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

## Setup (once per computer)

1. **Get the code** and move into the project folder:
   ```
   git clone https://github.com/nsg365/RL-Rotary-Inverted-Pendulum-SwingUp-and-Stabilization.git
   cd RL-Rotary-Inverted-Pendulum-SwingUp-and-Stabilization
   ```
2. **Install the Python dependencies** (Python 3.10 or newer):
   ```
   pip install -r requirements.txt
   ```
3. **Install the [Arduino IDE](https://www.arduino.cc/en/software).**

---

## Running the demo

**Step 1 — Flash the firmware** (only needed once, or after changing the sketch)
1. Plug the Arduino Mega into the laptop over USB.
2. In the Arduino IDE, open `firmware/prop_on_ard/prop_on_ard.ino`.
3. Select **Tools → Board → Arduino Mega or Mega 2560** and **Tools → Port →** the Mega's port.
4. Click **Upload** and wait for "Done uploading".
5. **Quit the Arduino IDE completely** (Cmd+Q on a Mac). If it stays open it can grab the serial port, and the Python script will just wait forever.

**Step 2 — Prepare the rig**
1. Turn on the 12 V motor supply.
2. Make sure the cables aren't wound up and the arm can swing freely.
3. **Let the pendulum hang perfectly still and centre the arm.** The encoders zero when the board starts, and starting the script restarts the board, so this position becomes "zero".

**Step 3 — Run the policy**
1. Find the Arduino's port:
   ```
   ls /dev/cu.usbmodem*        # macOS
   ls /dev/ttyACM*             # Linux   (Windows: COM3, COM4, ... in Device Manager)
   ```
2. From the project's top-level folder, run:
   ```
   python3 -m deploy.rl_inference --port /dev/cu.usbmodemXXXX \
       --model models/sac_real/ckpts/sac_real_407500_steps.zip
   ```
   Replace `/dev/cu.usbmodemXXXX` with your port.
3. **Stop** with Ctrl+C. The script sends 0 V before exiting, and the Arduino also cuts the motor by itself if it hears nothing for 100 ms.

**What you'll see:** the arm recentres, the pendulum settles, and the policy starts pumping it up.
The terminal shows `Swinging (RL)` or `Balancing (P)` with live angles.
The arm may swing past 150° during swing-up; that's allowed and doesn't interrupt anything.
If the pendulum falls after having been balanced, the script prints `Pendulum fell. Resetting hardware...`, recentres the arm and tries again.

> Keep an eye on the cables during the first few runs. Nothing stops the arm from winding them up if the swing-up keeps going one way.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Stuck at `Starting deployment loop`, nothing moves | The Arduino isn't sending data. Quit the Arduino IDE, unplug and replug the USB, and run again. |
| Arm drives off to one side and stays there | Motor direction is reversed. Flip `MOTOR_SIGN` in `prop_on_ard.ino` and reflash. |
| Angles look wrong or have the wrong sign | Run `python3 -m deploy.probe_hardware --port /dev/cu.usbmodemXXXX` (motor off), move the arm and pendulum by hand, and check the readings. This shows the raw firmware angle: hanging should read about 0, upright about ±180°. |
| Arm slowly drifts or sways while balancing | Expected to a degree (motor dead zone). Booting with the pendulum completely still helps. The arm gains are `k1`, `k2` in the sketch. |

---

## Tuning knobs

All in `prop_on_ard.ino` (reflash after changing):

| Setting | Current value | Meaning |
|---|---|---|
| `k1`, `k2`, `k3`, `k4` | base gains × 0.5, 0.5, 1.0, 0.75 | Gains on arm angle, arm velocity, pendulum angle, pendulum velocity |
| `BLEND_INNER`, `BLEND_OUTER` | 10°, 20° | Edges of the RL/proportional blend band |
| `alpha` | 0.4 | RL share inside the blend band |
| `DEADZONE`, `THRESH` | 0.5 V | Motor static-friction compensation |

---

## Training (optional)

The demo only needs the saved policy. These steps reproduce it.

**1. Simulation** — SAC with gSDE, parameter randomization, arm angle soft-bounded at 150°:
```
python3 -m training.train --arm_limit_deg 150 --soft_arm --domain_rand
```
Models are saved to `models/sac_gsde/`.

**2. Hardware fine-tuning** — flash `prop_on_ard.ino`, then:
```
# first run: start from the simulation model
python3 -m training.train_real --port /dev/cu.usbmodemXXXX \
    --model models/sac_gsde/best/best_model.zip --arm_limit_deg 150 --soft_arm --ent_coef 0.05

# later runs: resumes from the newest checkpoint in models/sac_real/ckpts/
python3 -m training.train_real --port /dev/cu.usbmodemXXXX --arm_limit_deg 150 --soft_arm --ent_coef 0.05
```
- Training runs until Ctrl+C and saves a checkpoint (model + replay buffer) every 2000 steps.
- `--arm_limit_deg 150` must match the model, or it won't load.
- `--ent_coef 0.05` fixes the exploration level; automatic tuning broke checkpoint reloading.
- Within 20° of upright the Arduino's proportional controller takes over, so the policy only learns the swing-up part.

**Running a policy in simulation:**
```
python3 -m sim.visualize     # uses the demo policy by default
python3 -m sim.eval
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
| `models/sac_real/ckpts/sac_real_407500_steps.zip` | Final hardware-fine-tuned policy (used in the demo and paper) |
| `models/sac_real/ckpts/sac_real_replay_buffer_407500_steps.pkl` | Matching replay buffer, for resuming `train_real.py` |
| `models/sac_real/model.zip` | Last model saved by `train_real.py` |
| `models/sac_real/ckpts/sac_real_12027_steps.zip` | Early hardware checkpoint |
