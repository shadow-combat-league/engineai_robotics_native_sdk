#!/usr/bin/env python3
"""Measure the vendor sim's realtime factor from sim_state timestamps.

Decodes the leading `timestamp d` of each sim_state message and compares
sim-time progression against wall clock. A factor < 1.0 means the sim runs
slower than realtime, which inflates every wall-clock latency (executor
tick, LCM transport, action LPF) when expressed in sim time — invisible to
any static config audit.

    python3 realtime_probe.py [seconds]
"""
import struct
import sys
import time

import lcm

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
samples = []  # (wall, sim_ts)


def handler(_ch, data):
    ts = struct.unpack_from(">d", data, 0)[0]
    samples.append((time.time(), ts))


lc = lcm.LCM("udpm://239.255.76.67:7667")
lc.subscribe("sim_state", handler)
end = time.time() + seconds
while time.time() < end:
    lc.handle_timeout(200)

if len(samples) < 10:
    sys.exit(f"only {len(samples)} sim_state messages - is the sim running?")

w0, s0 = samples[0]
w1, s1 = samples[-1]
dw = w1 - w0
ds = s1 - s0
# infer units: try s, ms, us, ns - pick the one giving a plausible factor
for unit, div in (("s", 1.0), ("ms", 1e3), ("us", 1e6), ("ns", 1e9)):
    f = (ds / div) / dw
    if 0.01 < f < 10.0:
        print(f"sim timestamp unit: {unit}")
        print(f"wall {dw:.2f}s, sim {ds / div:.2f}s -> REALTIME FACTOR {f:.3f}")
        break
else:
    print(f"raw: wall {dw:.3f}s, sim delta {ds:.6g} (unit unclear)")
print(f"messages: {len(samples)} ({len(samples) / dw:.0f} Hz wall)")
# jitter: sim-time step distribution across consecutive messages
import statistics
steps = [(samples[i + 1][1] - samples[i][1]) for i in range(len(samples) - 1)]
nz = [x for x in steps if x != 0]
if nz:
    print(f"sim-step per msg: median {statistics.median(nz):.6g}, "
          f"min {min(nz):.6g}, max {max(nz):.6g}, zeros {steps.count(0)}")
