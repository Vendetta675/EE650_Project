"""Outcome of a ros_ftc_e2e_test.sh run with the battery model: supervisor states, flight after the rotor cut and the
battery (plugin state). Usage: ftc_battery_summary.py <run> [<run> ...]"""
import re, sys
import numpy as np, pandas as pd
for r in sys.argv[1:]:
    ev = pd.read_csv(f"{r}/events.csv"); w = ev.wall_time.values; d = pd.read_csv(f"{r}/run.csv")
    c = d[d.t >= w[1]]; air = c[c.z < -0.3]; h = c[(c.t > w[1] + 3) & (c.t < w[1] + 20)]
    sup = [l.split("supervisor: ")[1].split(" (")[0] for l in open(f"{r}/supervisor.log") if "supervisor: " in l]
    line = (f"{r}: {' | '.join(sup)} || after cut: airborne {air.t.max() - w[1]:.1f} s, max sink {air.vz.max():.1f} m/s, "
            f"max tilt {max(air.roll.abs().max(), air.pitch.abs().max()):.0f} deg, alt 3-20 s {-h.z.max():.1f}..{-h.z.min():.1f} m")
    try:
        t = open(f"{r}/battery_state.txt").read()
        m = np.array([x for x in (np.array(re.findall(r"data: ([-\d.e+]+)", b), float) for b in t.split("\n\n") if "data:" in b)
                      if len(x) == 12])
        line += (f" || battery: SoC {m[0, 2]:.2f} -> {m[-1, 2]:.2f}, V min {m[:, 0].min():.2f}, I max {m[:, 1].max():.1f} A, "
                 f"max rotor {np.abs(m[:, 8:]).max():.0f} rad/s")
    except (FileNotFoundError, IndexError, ValueError):
        line += " || battery: off"
    print(line)
