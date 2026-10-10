"""Outcome of a low-battery fault-response run (ros_ftc_e2e_test.sh): supervisor timing, altitude at the rotor cut,
time to ground, touchdown sink rate and attitude. SOFT = touchdown sink <= 1.5 m/s and tilt <= 45 deg (the criterion of
section 6) and upright afterwards; TIPPED = resting tilt > 45 deg; HARD = upright but outside the soft limits. Touchdown = first row after the cut at ground level. Usage: landing_summary.py <run> [<run> ...]"""
import sys
import pandas as pd
for r in sys.argv[1:]:
    ev = pd.read_csv(f"{r}/events.csv"); w = ev.wall_time.values; d = pd.read_csv(f"{r}/run.csv")
    arm = [float(l.split("[")[2].split("]")[0]) for l in open(f"{r}/supervisor.log") if "-> DESCEND" in l or "-> ARMED" in l]
    c = d[d.t >= w[0]]; air = c[(c.z < -0.3) & (c.armed == 1)]
    # ground contact: first row after the cut at the resting altitude reading (0.19 m on the ground), whether or not
    # the motors are still armed (a motor cut disarms in the air); fall back to the last armed airborne row
    gnd = c[(c.t >= w[1]) & (-c.z < 0.25)]
    tg = gnd.t.iloc[0] if len(gnd) else air.t.max()
    td = c[(c.t > tg - 0.6) & (c.t <= tg + 0.2)]; after = c[(c.t > tg + 1) & (c.t < tg + 4)]
    tilt = max(td.roll.abs().max(), td.pitch.abs().max()); sink = td.vz.max()
    rest = max(after.roll.abs().median(), after.pitch.abs().median())
    verdict = "TIPPED" if rest > 45 else "SOFT" if sink <= 1.5 and tilt <= 45 else "HARD (upright)"
    alarm = f"{arm[0] - w[0]:.1f}" if arm else "-"
    print(f"{r.split('ros_ftc_e2e_')[-1]:14s} alarm {alarm} s after onset | alt at cut {-d[d.t >= w[1]].z.iloc[0]:.1f} m | "
          f"ground {tg - w[1]:.1f} s after cut | touchdown sink {sink:.1f} m/s, tilt {tilt:.0f} deg, resting {rest:.0f} deg"
          f" => {verdict}")
