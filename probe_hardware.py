"""Passive (0 V) probe of the Arduino state stream. Helps verify the
convention matches sim: theta=0 upright, theta=+-pi hanging.

Run it, then MANUALLY move the pendulum through these poses and watch the prints:
    1. Hold pendulum upright.    Expect theta ~ 0.
    2. Let it hang.              Expect theta ~ +-pi (~ +-3.14).
    3. Lift +90 deg clockwise.   Expect theta ~ +pi/2 or -pi/2 (consistent sign).

Also rotate the ARM left and right while pendulum hangs.
    Expect arm angle to increase one way, decrease the other, smoothly.
"""

import argparse
import time
import numpy as np

from furuta_real import ArduinoRobot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    args = p.parse_args()

    robot = ArduinoRobot(args.port)
    print("Sending 0 V. Move the pendulum/arm by hand. Ctrl+C to stop.")
    t0 = time.time()
    try:
        while True:
            robot.send_voltage(0.0)
            th_pend, th_arm, om_pend, om_arm = robot.read_state()
            th_pend_wrap = ((th_pend + np.pi) % (2 * np.pi)) - np.pi
            print(f"t={time.time()-t0:5.1f}  "
                  f"theta(pend) raw={th_pend:+7.3f}  wrap={th_pend_wrap:+7.3f}  "
                  f"({np.degrees(th_pend_wrap):+6.1f} deg)   "
                  f"arm={th_arm:+7.3f} ({np.degrees(th_arm):+6.1f} deg)   "
                  f"om_pend={om_pend:+6.2f}  om_arm={om_arm:+6.2f}")
    except KeyboardInterrupt:
        robot.close()


if __name__ == "__main__":
    main()
