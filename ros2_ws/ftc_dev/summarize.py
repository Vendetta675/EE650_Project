"""Summary of an FTC test run: sink rate, time to ground, position error, spin, motor saturation, controller log."""
import sys, glob, os
import numpy as np, pandas as pd
from pymavlink import mavutil
run = sys.argv[1]
d = pd.read_csv(f"{run}/run.csv"); i = d.index[d.phase == "failure"][0]; f = d.loc[i:].copy(); f["t"] -= f.t.iloc[0]
air = f[(f.armed == 1) & (f.z < -0.5)]
gnd = f[f.z > -0.5]
t_gnd = gnd.t.iloc[0] if len(gnd) else None
w = f[(f.t > 1.5) & (f.t < (t_gnd or 1e9))]
herr = np.hypot(w.x - w.tx, w.y - w.ty)
print(f"{os.path.basename(run)}: airborne after failure {air.t.max():.1f} s | touchdown {t_gnd if t_gnd is None else round(t_gnd,1)} s | "
      f"alt at +3 s {-f[f.t>=3].z.iloc[0]:.1f} m | horiz err median {herr.median():.2f} m max {herr.max():.2f} | "
      f"yaw rate median {w.yawspeed.median():.1f} rad/s | m3/m4 at max {(w.m3>=1940).mean():.0%}/{(w.m4>=1940).mean():.0%} | "
      f"|roll|,|pitch| p90 {w.roll.abs().quantile(.9):.0f},{w.pitch.abs().quantile(.9):.0f} deg")
logs = sorted(glob.glob(f"{run}/logs/*.BIN"))
if logs:
    m = mavutil.mavlink_connection(logs[-1]); c = []
    while True:
        r = m.recv_match(type="CCF", blocking=False)
        if r is None: break
        c.append((r.Err, r.DX, r.DY, r.UR, r.UP))
    if c:
        a = np.array(c[50:]) if len(c) > 60 else np.array(c)
        print(f"   CCF: tilt err median {np.median(a[:,0]):.2f} rad | dist est median {np.median(a[:,1]):.3f},{np.median(a[:,2]):.3f} N m | "
              f"uR,uP median {np.median(a[:,3]):.2f},{np.median(a[:,4]):.2f}")
