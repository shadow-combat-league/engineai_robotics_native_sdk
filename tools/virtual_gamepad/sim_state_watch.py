#!/usr/bin/env python3
"""Watch the SDK sim's ground-truth base state over LCM (channel sim_state).

Hand-decodes data::SimState (big-endian after the 8-byte hash):
  timestamp d | num_ranges i | 3x vector<double>[n] | base_pos d[3] |
  base_linvel d[3] | base_quat wxyz d[4] | ...
Prints height/tilt/xy once per second. Runs in the container (python3-lcm).
Usage: sim_state_watch.py [seconds]
"""
import struct
import sys
import time

import lcm

def handler(channel, data):
    global last_print, t0
    off = 8  # skip hash
    (_, n) = struct.unpack_from(">di", data, off)
    off += 12 + 3 * n * 8
    vals = struct.unpack_from(">10d", data, off)
    pos = vals[0:3]
    quat = vals[6:10]  # wxyz
    w, x, y, z = quat
    import math
    zz = max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))
    tilt = math.degrees(math.acos(zz))
    now = time.time()
    if now - last_print >= 1.0:
        last_print = now
        print(f"t={now-t0:5.1f}s z={pos[2]:.3f} tilt={tilt:5.1f} xy=({pos[0]:+.2f},{pos[1]:+.2f})", flush=True)

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
last_print = 0.0
t0 = time.time()
lc = lcm.LCM("udpm://239.255.76.67:7667")
lc.subscribe("sim_state", handler)
end = time.time() + seconds
while time.time() < end:
    lc.handle_timeout(200)
