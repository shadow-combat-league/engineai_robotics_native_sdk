#!/usr/bin/env python3
"""Record the sim<->executor LCM traffic with wall timestamps.

Writes CSV rows to the output file:
    S,<wall_us>,<msg_ts>,q0..q24        (sim_state joint_position, serial order)
    C,<wall_us>,<msg_ts>,qd0..qd24      (sim_command joint_position targets)

Cross-correlated against the rl_teleop flight log's wall-us column, this
yields the REAL per-tick obs staleness and command transport lag of the
vendor loop — the loop-trace used for trace-driven delay DR in training.

Usage (in container): python3 loop_tap.py <out.csv> [seconds]
"""
import struct
import sys
import time

import lcm

SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
rows = []  # buffered in memory; per-row disk writes drop messages at 500 Hz


def on_state(_ch, data):
    rows.append((b"S", time.time(), data))


def on_cmd(_ch, data):
    rows.append((b"C", time.time(), data))


lc = lcm.LCM("udpm://239.255.76.67:7667")
lc.subscribe("sim_state", on_state)
lc.subscribe("sim_command", on_cmd)
end = time.time() + SECONDS
while time.time() < end:
    lc.handle_timeout(200)

with open(sys.argv[1], "w") as out:
    for tag, wall, data in rows:
        ts, n = struct.unpack_from(">di", data, 8)
        q = struct.unpack_from(">%dd" % n, data, 20)[:n]
        vals = list(q)
        if tag == b"S":
            # after 3 joint vectors: base pos(3) linvel(3) quat(4) angvel(3),
            # imu_link pos(3) linvel(3) quat(4) angvel(3),
            # imu_sensor quat(4) linacc(3) angvel(3)
            off = 20 + 3 * n * 8
            base = struct.unpack_from(">13d", data, off)
            imu_sensor = struct.unpack_from(">10d", data, off + 26 * 8)
            vals += list(base[6:13])       # base quat wxyz + base angvel
            vals += list(imu_sensor[:4])   # imu_sensor quat wxyz
            vals += list(imu_sensor[7:10])  # imu_sensor angvel
        else:
            # SimCommand: q, qd_des, tau_ff, kp, kd — record the other four
            # vectors too (the blob may remap gains posture-dependently)
            rest = struct.unpack_from(">%dd" % (4 * n), data, 20 + n * 8)
            vals += list(rest)
        out.write("%s,%d,%.6f," % (tag.decode(), int(wall * 1e6), ts)
                  + ",".join("%.6f" % v for v in vals) + "\n")
print("tap done -> %s (%d rows)" % (sys.argv[1], len(rows)))
