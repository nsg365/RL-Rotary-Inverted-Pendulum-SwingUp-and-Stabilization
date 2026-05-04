"""QubeDynamics class — interface preserved from furuta-master
(action in [-1, 1], params/randomize API, returns (arm_acc, pend_acc)),
but the equations of motion are the user's original parallel-axis form
from the pre-adaptation furuta_env.py.

Convention (matches swing-up/furuta_env.py):
    state = [arm_angle, pend_angle, arm_vel, pend_vel]   (ALPHA, THETA, ... indexing)
    pend_angle = 0 at UPRIGHT, +-pi at HANGING (unstable equilibrium at 0)

Equations (user's original, renamed to match this state layout):

    tau  = kt * (V - km * arm_vel) / Rm
    M11  = Mp*lp^2*sin^2(pend) + Jr + Mp*Lr^2
    M12  = Mp*Lr*lp*cos(pend)
    M22  = Jp + Mp*lp^2
    C1   = 2*Mp*lp^2*cos(pend)*sin(pend)*arm_vel*pend_vel - Mp*Lr*lp*sin(pend)*pend_vel^2
    C2   = -Mp*lp^2*cos(pend)*sin(pend)*arm_vel^2
    G2   = -Mp*g*lp*sin(pend)       # zero at upright (pend=0) -> unstable equilibrium
    tau1 = tau - C1 - Dr*arm_vel
    tau2 = -C2 - G2 - Dp*pend_vel
    arm_acc  = (M22*tau1 - M12*tau2) / det
    pend_acc = (M11*tau2 - M12*tau1) / det
"""

import numpy as np

from config import HARDWARE, CONSTRAINTS


class QubeDynamics:
    def __init__(
        self,
        g=HARDWARE['g'],
        Rm=HARDWARE['Rm'], Rm_std=0.0,
        V=CONSTRAINTS['max_voltage'],
        kt=HARDWARE['kt'], kt_std=0.0,
        km=HARDWARE['km'], km_std=0.0,
        Mp=HARDWARE['Mp'], Mp_std=0.0,
        Lp=HARDWARE['Lp'], Lp_std=0.0,
        lp=HARDWARE['lp'], lp_std=0.0,
        Jp=HARDWARE['Jp'], Jp_std=0.0,
        Mr=HARDWARE['Mr'], Mr_std=0.0,
        Lr=HARDWARE['Lr'], Lr_std=0.0,
        Jr=HARDWARE['Jr'], Jr_std=0.0,
        Dr=HARDWARE['Dr'], Dr_std=0.0,
        Dp=HARDWARE['Dp'], Dp_std=0.0,
        stall_torque=0.16,
    ):
        self.g = g
        self.V = V
        self.stall_torque = stall_torque

        self.Rm_mean = Rm; self.Rm_std = Rm_std
        self.kt_mean = kt; self.kt_std = kt_std
        self.km_mean = km; self.km_std = km_std
        self.Mp_mean = Mp; self.Mp_std = Mp_std
        self.Lp_mean = Lp; self.Lp_std = Lp_std
        self.lp_mean = lp; self.lp_std = lp_std
        self.Jp_mean = Jp; self.Jp_std = Jp_std
        self.Mr_mean = Mr; self.Mr_std = Mr_std
        self.Lr_mean = Lr; self.Lr_std = Lr_std
        self.Jr_mean = Jr; self.Jr_std = Jr_std
        self.Dr_mean = Dr; self.Dr_std = Dr_std
        self.Dp_mean = Dp; self.Dp_std = Dp_std

        self.randomize()

    def randomize(self):
        def _sample_pos(mean, std):
            return max(mean * 0.3, np.random.normal(mean, std))

        self.Rm = _sample_pos(self.Rm_mean, self.Rm_std)
        self.kt = _sample_pos(self.kt_mean, self.kt_std)
        self.km = _sample_pos(self.km_mean, self.km_std)
        self.Mp = _sample_pos(self.Mp_mean, self.Mp_std)
        self.Lp = _sample_pos(self.Lp_mean, self.Lp_std)
        self.lp = _sample_pos(self.lp_mean, self.lp_std)
        self.Jp = _sample_pos(self.Jp_mean, self.Jp_std)
        self.Mr = _sample_pos(self.Mr_mean, self.Mr_std)
        self.Lr = _sample_pos(self.Lr_mean, self.Lr_std)
        self.Jr = _sample_pos(self.Jr_mean, self.Jr_std)
        self.Dr = max(0.0, np.random.normal(self.Dr_mean, self.Dr_std))
        self.Dp = max(0.0, np.random.normal(self.Dp_mean, self.Dp_std))
        self._init_const()

    def _init_const(self):
        # Precomputed coefficients mirroring the user's original furuta_env.py.
        self._Mp_lp2    = self.Mp * self.lp ** 2
        self._Jr_Mp_Lr2 = self.Jr + self.Mp * self.Lr ** 2
        self._Mp_Lr_lp  = self.Mp * self.Lr * self.lp
        self._Jp_Mp_lp2 = self.Jp + self.Mp * self.lp ** 2
        self._Mp_g_lp   = self.Mp * self.g * self.lp
        self._2_Mp_lp2  = 2.0 * self.Mp * self.lp ** 2

    @property
    def params(self):
        params = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return params

    @params.setter
    def params(self, params):
        self.__dict__.update(params)
        self._init_const()

    def __call__(self, state, action):
        """state = [arm, pend, arm_vel, pend_vel]; action in [-1, 1].
        Returns (arm_acc, pend_acc)."""
        arm, pend, arm_dot, pend_dot = state
        voltage = action * self.V

        # Motor torque (back-EMF model; arm_dot is the rotor velocity)
        tau = (self.kt * (voltage - self.km * arm_dot)) / self.Rm
        tau = np.clip(tau, -self.stall_torque, self.stall_torque)

        sin_p = np.sin(pend)
        cos_p = np.cos(pend)

        # Mass matrix
        M11 = self._Mp_lp2 * (sin_p ** 2) + self._Jr_Mp_Lr2
        M12 = self._Mp_Lr_lp * cos_p
        M22 = self._Jp_Mp_lp2
        det = M11 * M22 - M12 * M12

        # Coriolis / centripetal
        C1 = (self._2_Mp_lp2 * cos_p * sin_p * arm_dot * pend_dot
              - self._Mp_Lr_lp * sin_p * pend_dot ** 2)
        C2 = -self._Mp_lp2 * cos_p * sin_p * arm_dot ** 2

        # Gravity (on pendulum only)
        G2 = -self._Mp_g_lp * sin_p

        # RHS
        tau1 = tau - C1 - self.Dr * arm_dot
        tau2 = -C2 - G2 - self.Dp * pend_dot

        arm_acc  = (M22 * tau1 - M12 * tau2) / det
        pend_acc = (M11 * tau2 - M12 * tau1) / det

        return arm_acc, pend_acc
