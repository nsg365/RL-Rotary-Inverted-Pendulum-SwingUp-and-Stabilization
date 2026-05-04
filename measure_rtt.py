"""
Measures the Round Trip Time (RTT) of the Arduino communication.

RTT is defined as the time from sending a voltage command to receiving a
complete, fresh state packet back from the Arduino.

This script sends a command, immediately waits for a reply, and records the
time difference. It repeats this many times to calculate statistics on the
communication latency.
"""

import argparse
import time
import numpy as np
from furuta_real import ArduinoRobot
import statistics

def main():
    p = argparse.ArgumentParser(description="Measure communication RTT with the Furuta pendulum hardware.")
    p.add_argument("--port", required=True, 
                   help="Arduino serial port (e.g., /dev/cu.usbmodem1401)")
    p.add_argument("--num_samples", type=int, default=1000,
                   help="Number of RTT samples to collect.")
    args = p.parse_args()

    print(f"🔌 Connecting to hardware on {args.port}...")
    robot = ArduinoRobot(args.port)
    
    rtt_samples = []
    print(f"🏃 Running RTT measurement for {args.num_samples} samples...")

    try:
        # Initial read to make sure buffer is clear and we're synced
        robot.read_state()

        for i in range(args.num_samples):
            # Time right before sending
            t_start = time.perf_counter()
            
            # Send a command and immediately wait for a state update
            robot.send_voltage(0.0)
            robot.read_state()
            
            # Time right after receiving
            t_end = time.perf_counter()
            
            rtt = (t_end - t_start) * 1000  # Convert to milliseconds
            rtt_samples.append(rtt)
            
            # Print progress without spamming the console
            if (i + 1) % 100 == 0:
                print(f"  ... collected {i + 1}/{args.num_samples} samples", end='\r')

    except KeyboardInterrupt:
        print("\n🛑 Measurement interrupted by user.")
    finally:
        print("\n⚡ Shutting down hardware safely...")
        robot.close()

    if rtt_samples:
        print("\n--- RTT Statistics ---")
        print(f"Samples collected: {len(rtt_samples)}")
        print(f"Min RTT:    {min(rtt_samples):.3f} ms")
        print(f"Average RTT:{statistics.mean(rtt_samples):.3f} ms")
        print(f"Max RTT:    {max(rtt_samples):.3f} ms")
        print(f"Std Dev:    {statistics.stdev(rtt_samples):.3f} ms")
    else:
        print("\nNo RTT samples were collected.")

if __name__ == "__main__":
    main()
