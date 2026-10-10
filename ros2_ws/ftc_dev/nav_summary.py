"""Navigation after a rotor failure (plans square and mission): cross-track error to the planned 20 m square, waypoints
reached after the cut, landing point. Usage: nav_summary.py <run>   (run.csv + events.csv from the plugin)"""
import os, sys
import numpy as np, pandas as pd
run = sys.argv[1]
SQ = np.array([(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)], float)
d = pd.read_csv(f"{run}/run.csv")
ev = pd.read_csv(f"{run}/events.csv")
t_cut = ev[ev.event == "cut"].wall_time.iloc[0] if (ev.event == "cut").any() else ev[ev.event == "start"].wall_time.iloc[0]

def xtrack(n, e):  # distance to the closest segment of the square
    p, best = np.array([n, e]), 1e9
    for a, b in zip(SQ[:-1], SQ[1:]):
        u = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
        best = min(best, np.linalg.norm(p - (a + u * (b - a))))
    return best

nav = d[d.phase.str.startswith("wp") & (d.armed == 1)]
after = nav[nav.t > t_cut + 3]  # past the cut transient
air_after = d[(d.t > t_cut) & (d.armed == 1) & (d.z < -0.5)]
xt = np.array([xtrack(r.x, r.y) for r in after.itertuples()]) if len(after) else np.array([np.nan])
cut_tr = d[(d.t >= t_cut) & (d.t < t_cut + 3)]
reached = []
for i, (n, e) in enumerate(SQ[1:], 1):
    near = d[(d.t > t_cut) & (np.hypot(d.x - n, d.y - e) < 2.0) & (d.z < -5)]
    if len(near):
        reached.append(f"WP{i} +{near.t.iloc[0] - t_cut:.0f}s")
last = d[d.armed == 1].iloc[-1]
alive = len(air_after) and (air_after.t.max() - t_cut)
print(f"{os.path.basename(run)}: airborne {alive:.1f} s after the cut | waypoints reached after the cut: "
      f"{', '.join(reached) or 'none'}")
print(f"   cross-track error after the cut (+3 s on, navigating): median {np.nanmedian(xt):.2f} m, p95 "
      f"{np.nanpercentile(xt, 95):.2f} m, max {np.nanmax(xt):.2f} m | alt {-after.z.max():.1f}..{-after.z.min():.1f} m"
      if len(after) else "   no navigation after the cut")
print(f"   cut transient (3 s): max tilt {max(cut_tr.roll.abs().max(), cut_tr.pitch.abs().max()):.0f} deg, "
      f"height loss {cut_tr.z.max() - cut_tr.z.iloc[0]:.2f} m | final position ({last.x:.1f}, {last.y:.1f}), "
      f"{np.hypot(last.x - SQ[-1][0], last.y - SQ[-1][1]):.1f} m from the last waypoint; max sink {air_after.vz.max():.1f} m/s")
