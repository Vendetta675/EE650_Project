"""Summary of a battery_sim.sh run: SITL hover numbers (test.log RESULT line) and, from the plugin's own state
(battery_state.txt, gz.msgs.Float_V [V, I, SoC, mAh, I_m x4, omega x4]), the peak thrust-to-weight ratio
sum(kT omega^2) / (m g) during the full-throttle climb. Usage: battery_summary.py <run> <mass_kg> [kT]"""
import re, sys
import numpy as np
run, mass = sys.argv[1], float(sys.argv[2]); kT = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0e-5
res = [l for l in open(f"{run}/test.log") if l.startswith("RESULT")]
txt = open(f"{run}/battery_state.txt").read()
msgs = [np.array(re.findall(r"data: ([-\d.e+]+)", m), float) for m in txt.split("\n\n") if "data:" in m]
msgs = np.array([m for m in msgs if len(m) == 12])
line = res[0].strip()[7:] if res else "no RESULT"
if len(msgs):
    w = np.abs(msgs[:, 8:12]); thrust = kT * (w ** 2).sum(1); twr = thrust / (mass * 9.81)
    print(f"{run}: {line}\n   plugin: SoC {msgs[0, 2]:.2f} -> {msgs[-1, 2]:.2f}, peak TWR {twr.max():.2f} "
          f"(max rotor speed {w.max():.0f} rad/s), peak current {msgs[:, 1].max():.1f} A, min voltage {msgs[:, 0].min():.2f} V")
else:
    print(f"{run}: {line}\n   plugin: no battery state (model disabled)")
