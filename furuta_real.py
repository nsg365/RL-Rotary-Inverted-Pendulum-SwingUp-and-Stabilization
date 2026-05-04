"""Hardware env — equivalent of furuta-master/furuta/rl/envs/furuta_real.py,
adapted to the user's Arduino firmware protocol.

Serial protocol (matches arduino_firmware.ino):
    RX (from Arduino at 100 Hz):
        2 bytes sync word 0xABCD (little-endian -> bytes CD AB on wire)
        16 bytes payload: float32 thetaPend, thetaArm, omegaPend, omegaArm
    TX (voltage command, any time):
        4 bytes: float32 voltage in volts

Convention:
    thetaPend : pendulum angle, 0 = UPRIGHT, +-pi = HANGING (Arduino firmware)
    thetaArm  : arm angle (free accumulating)

The RL action is in [-1, 1]; it is scaled to +-max_voltage before being sent.
"""

import struct
import time
from typing import Optional

import numpy as np
import serial

from config import CONSTRAINTS
from furuta_env import FurutaBase
from furuta_utils import ALPHA, ALPHA_DOT, THETA, THETA_DOT


SYNC_WORD = b"\xCD\xAB"
PAYLOAD_BYTES = 16


class ArduinoRobot:
    def __init__(self, port: str, baud: int = 500000, bootloader_wait: float = 2.0):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 0  # non-blocking

        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        time.sleep(0.1)
        self.ser.dtr = True
        self.ser.rts = True
        time.sleep(bootloader_wait)
        self.ser.reset_input_buffer()

        self._rx = b""

    def read_state(self) -> tuple:
        """Block until one fresh state packet is received. Returns
        (thetaPend, thetaArm, omegaPend, omegaArm)."""
        while True:
            chunk = self.ser.read(1024)
            if chunk:
                self._rx += chunk
                idx = self._rx.rfind(SYNC_WORD)
                if idx != -1 and (len(self._rx) - idx) >= 2 + PAYLOAD_BYTES:
                    start = idx + 2
                    data = self._rx[start : start + PAYLOAD_BYTES]
                    # drop everything up to and including this packet
                    self._rx = self._rx[start + PAYLOAD_BYTES :]
                    try:
                        return struct.unpack("<ffff", data)
                    except struct.error:
                        continue

    def send_voltage(self, volts: float):
        self.ser.write(struct.pack("<f", float(volts)))
        self.ser.flush()

    def close(self):
        try:
            self.send_voltage(0.0)
        finally:
            self.ser.close()


class FurutaReal(FurutaBase):
    """Hardware Furuta env. Drop-in replacement for FurutaSim — same obs/action
    space, same reward — plugs into the same SB3 SAC training loop.

    Episode reset procedure (autonomous, no human):
      1. Send 0 V, wait for pendulum to settle near HANGING (theta ~ +-pi).
      2. Start episode.
    Episodes are terminated only by the SB3 TimeLimit wrapper.
    """

    SETTLE_COS_THRESH = np.cos(np.deg2rad(10))  # |theta - pi| < 10 deg
    SETTLE_HOLD_SEC = 0.2
    MAX_RESET_SEC = 4.0

    # Arm auto-recenter (runs before the hanging-settle wait)
    ARM_CENTER_DEG = 15.0     # acceptable arm zone at reset start
    ARM_RECENTER_VOLTS = 3.5  # gentle corrective voltage
    ARM_RECENTER_SEC = 3.0    # cap time spent recentering

    def __init__(
        self,
        port: str,
        control_freq: int = int(round(1.0 / CONSTRAINTS['dt'])),
        reward: str = "cos_alpha",
        angle_limits=[np.pi / 2, None],   # arm soft bound; pendulum wrap-detected
        speed_limits=[60, 400],
        max_voltage: float = CONSTRAINTS['max_voltage'],
        action_smoothing: float = 0.3,    # 0 = no smoothing, 1 = frozen. v = (1-a)*v_prev + a*v_new
        soft_arm_bound: bool = False,
    ):
        super().__init__(control_freq, reward, angle_limits, speed_limits, soft_arm_bound)
        self.max_voltage = max_voltage
        self.action_smoothing = float(action_smoothing)
        self.robot = ArduinoRobot(port)
        self._state = None
        self._prev_volts = 0.0

    @staticmethod
    def _wrap(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    @staticmethod
    def _to_sim(th_arduino):
        """Arduino firmware zeroes the encoder at boot, and we boot with the
        pendulum hanging -> Arduino's theta=0 is the HANGING pose.
        The sim expects theta=0 UPRIGHT, theta=+-pi HANGING. Shift by pi."""
        return FurutaReal._wrap(th_arduino + np.pi)

    def _read_and_update(self, action_norm: float):
        """Send one voltage command, read one fresh state packet, update state."""
        raw_volts = float(np.clip(action_norm, -1.0, 1.0)) * self.max_voltage
        a = self.action_smoothing
        volts = (1.0 - a) * self._prev_volts + a * raw_volts
        self._prev_volts = volts
        self.robot.send_voltage(volts)
        th_pend, th_arm, om_pend, om_arm = self.robot.read_state()
        th_pend = self._to_sim(th_pend)
        self._state = np.array([th_arm, th_pend, om_arm, om_pend], dtype=np.float32)

    def _update_state(self, a):
        self._read_and_update(a)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        self._prev_volts = 0.0

        # --- Step 1: walk the arm back toward centre (|alpha| < ARM_CENTER_DEG).
        # Nudge with a small voltage whose sign opposes the current arm offset.
        center_rad = np.deg2rad(self.ARM_CENTER_DEG)
        t_rc = time.time()
        while time.time() - t_rc < self.ARM_RECENTER_SEC:
            th_pend, th_arm, om_pend, om_arm = self.robot.read_state()
            if abs(th_arm) < center_rad:
                self.robot.send_voltage(0.0)
                break
            # push arm toward 0: if th_arm > 0 we need negative voltage (and vice-versa)
            v = -np.sign(th_arm) * self.ARM_RECENTER_VOLTS
            self.robot.send_voltage(float(v))
        self.robot.send_voltage(0.0)

        # --- Step 2: 0 V and wait for pendulum to settle hanging (theta ~ +-pi).
        t0 = time.time()
        held = 0.0
        last = time.time()
        while time.time() - t0 < self.MAX_RESET_SEC:
            th_pend, th_arm, om_pend, om_arm = self.robot.read_state()
            th_pend = self._to_sim(th_pend)
            # cos(theta) ~ -1 when hanging  (theta = +-pi in sim convention)
            now = time.time()
            dt = now - last
            last = now
            if np.cos(th_pend) < -self.SETTLE_COS_THRESH and abs(om_pend) < 0.5:
                held += dt
                if held >= self.SETTLE_HOLD_SEC:
                    break
            else:
                held = 0.0
            self.robot.send_voltage(0.0)

        # One fresh read to populate self._state
        th_pend, th_arm, om_pend, om_arm = self.robot.read_state()
        th_pend = self._to_sim(th_pend)
        self._state = np.array([th_arm, th_pend, om_arm, om_pend], dtype=np.float32)
        return self.get_obs(), {}

    def close(self):
        self.robot.close()
