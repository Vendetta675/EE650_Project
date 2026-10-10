"""Compare flight while the FTC is armed (pass-through) with the hover just before it. Usage: armed_check.py <run>"""
import sys
import numpy as np
import pandas as pd

d = pd.read_csv(f"{sys.argv[1]}/run.csv")
a = d[d.phase == "armed"]
h = d[d.phase == "hover"].tail(len(a))
for name, g in (("armed (pass-through)", a), ("hover before", h)):
    print(f"{name:22s} {len(g) / 10:4.1f} s: alt {-g.z.max():.2f}-{-g.z.min():.2f} m | horiz max {np.hypot(g.x, g.y).max():.2f} m"
          f" | max tilt {max(g.roll.abs().max(), g.pitch.abs().max()):.1f} deg | max |yaw rate| {g.yawspeed.abs().max():.2f}")
f = d[d.phase.isin(["failure", "hold_after_failure"])]
m = f[["m1", "m2", "m3", "m4"]]
cut = f[(m.max(axis=1) - m.min(axis=1)) > 300]
if len(cut):
    t0 = cut.t.iloc[0]
    print("first row with a split motor pattern at", round(t0 - f.t.iloc[0], 2), "s after the trigger;",
          "rows around it:")
    for _, r in f[(f.t >= t0 - 0.2) & (f.t <= t0 + 0.3)].iterrows():
        print(f"  {r.t - t0:+.2f} m={int(r.m1)},{int(r.m2)},{int(r.m3)},{int(r.m4)} roll={r.roll:5.1f} pitch={r.pitch:5.1f} yr={r.yawspeed:5.2f}")
