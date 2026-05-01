"""Adapted from furuta-master/furuta/utils.py.

Label convention is FLIPPED vs. repo to match user project:
    THETA = pendulum  (repo used ALPHA)
    ALPHA = arm       (repo used THETA)

The internal physics state vector layout is kept identical to the repo
(state = [arm_angle, pendulum_angle, arm_vel, pendulum_vel]) so that
QubeDynamics math stays untouched. We just rename the indices.
"""

import numpy as np
from scipy import signal

ALPHA = 0        # arm
THETA = 1        # pendulum
ALPHA_DOT = 2    # arm velocity
THETA_DOT = 3    # pendulum velocity


class VelocityFilter:
    def __init__(self, x_len, dt, num=(50, 0), den=(1, 50), x_init=None):
        derivative_filter = signal.cont2discrete((num, den), dt)
        self.b = derivative_filter[0].ravel().astype(np.float32)
        self.a = derivative_filter[1].astype(np.float32)
        if x_init is None:
            self.z = np.zeros((max(len(self.a), len(self.b)) - 1, x_len), dtype=np.float32)
        else:
            self.set_initial_state(x_init)

    def set_initial_state(self, x_init):
        assert isinstance(x_init, np.ndarray)
        zi = signal.lfilter_zi(self.b, self.a)
        self.z = np.outer(zi, x_init)

    def __call__(self, x):
        xd, self.z = signal.lfilter(self.b, self.a, x[None, :], 0, self.z)
        return xd.ravel()


class Timing:
    def __init__(self, f):
        self.f = f
        self.dt = 1.0 / f
