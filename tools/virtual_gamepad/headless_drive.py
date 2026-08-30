#!/usr/bin/env python3
"""Headless virtual-gamepad macro sender for the EngineAI SDK.

Publishes a button combo on LCM exactly like tools/virtual_gamepad does
(20 Hz, combo held for 10 cycles, then 10 cycles of all-zeros), then exits.
Run INSIDE the engineai_robotics_env container (has python3-lcm; host netns).

  python3 sdk_gamepad_drive.py LB,START     # 'start' mode
  python3 sdk_gamepad_drive.py LB,A         # pd_stand
  python3 sdk_gamepad_drive.py RB,A         # rl_teleop
  python3 sdk_gamepad_drive.py LB,RB        # passive (soft e-stop)
"""
import sys
import time

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcm  # noqa: E402
from lcm_msgs.data import GamepadKeys  # noqa: E402

BUTTON_INDEX = {"LB": 0, "RB": 1, "A": 2, "B": 3, "X": 4, "Y": 5, "START": 7}
RATE_HZ = 20
HOLD_CYCLES = 10   # matches MACRO_BUTTON_SENDING_COUNT in the UI
ZERO_CYCLES = 10

def main():
    combo = [b.strip().upper() for b in sys.argv[1].split(",")]
    handle = lcm.LCM("udpm://239.255.76.67:7667")

    def publish(pressed):
        msg = GamepadKeys()
        msg.timestamp = int(time.time() * 1_000_000)
        for name in pressed:
            msg.digital_states[BUTTON_INDEX[name]] = 1
        handle.publish("virtual_gamepad/gamepad_keys", msg.encode())

    for _ in range(HOLD_CYCLES):
        publish(combo)
        time.sleep(1.0 / RATE_HZ)
    for _ in range(ZERO_CYCLES):
        publish([])
        time.sleep(1.0 / RATE_HZ)
    print(f"sent {'+'.join(combo)}")

if __name__ == "__main__":
    main()
