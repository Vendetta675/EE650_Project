"""Outcome of each fault episode with and without the fault-response supervisor.

Inputs: flight CSVs (eval_supervisor.py rows, or recorder rows for the supervisor-off baseline flights) + the event log
with the plugin's ground truth. Only the first fault episode after takeoff of each flight is scored.
Outcome:
  airborne      rel_alt > 1 m when the fault was recovered (or at the end of the log)
  hard_landing  touched down (rel_alt < 0.3 m) with sink rate > 1.5 m/s (d rel_alt / dt over the last 0.5 s before
                touchdown) or |roll| / |pitch| > 45 deg in the 2 s before touchdown
  soft_landing  touched down without meeting the hard-landing criteria
  Criteria history: first declared as sink > 3 m/s or tilt > 60 deg. Checked on the supervisor-off baseline flights only
  (before any supervisor flight), that rule labelled 13 of 14 lost vehicles "controlled": the failing copter sinks
  level at 2-3 m/s. Revised to 1.5 m/s / 45 deg (multirotor gear tolerance; ArduPilot LAND touches down at 0.5 m/s).
  Continuous metrics (sink rate, tilt, peak vib_z within 1 s of touchdown) are reported for every episode.
Also: time from fault start to the first alarm and to the supervisor's DESCEND, altitude at DESCEND, false-alarm
holds per minute of normal flight (supervisor HOLD entries before the fault started).
Usage: python3 analyze_supervisor.py --runs supervisor_eval/flight_*.csv --events fault_events_sup.csv \
           --baseline flights_v2/flight_{06..16}.csv --baseline-events fault_events_v2.csv --severity ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def first_start(ev, t0, t1):
    s = ev[(ev.event == "start") & (ev.wall_time >= t0) & (ev.wall_time <= t1)].sort_values("wall_time")
    return None if s.empty else s.iloc[0]


def outcome(df, ev, sev):
    df = df.sort_values("timestamp").reset_index(drop=True)
    st = first_start(ev, df.timestamp.iloc[0], df.timestamp.iloc[-1])
    if st is None:
        return None
    t0 = st.wall_time
    rec = ev[(ev.event == "recover") & (ev.wall_time > t0)].wall_time.min()
    after = df[df.timestamp >= t0]
    td = after[(after.rel_alt < 0.3)]
    res = {"loe_final": sev, "motor": int(st.motor_index)}
    if td.empty or (np.isfinite(rec) and td.timestamp.iloc[0] > rec and
                    df[df.timestamp <= rec].rel_alt.iloc[-1] > 1.0):
        res["outcome"] = "airborne"
    else:
        t_td = td.timestamp.iloc[0]
        pre = df[(df.timestamp > t_td - 0.5) & (df.timestamp <= t_td)]
        sink = float((pre.rel_alt.iloc[0] - pre.rel_alt.iloc[-1]) / max(pre.timestamp.iloc[-1] - pre.timestamp.iloc[0], 1e-3)) \
            if len(pre) > 1 else float("nan")
        att = df[(df.timestamp > t_td - 2) & (df.timestamp <= t_td)]
        tilt = float(np.nanmax(np.abs(att[["roll", "pitch"]].to_numpy()))) if len(att) else float("nan")
        near = df[(df.timestamp > t_td - 1) & (df.timestamp < t_td + 1)]
        impact = float(near.vib_z.max()) if "vib_z" in near and len(near) else float("nan")
        res.update(touchdown_s=round(t_td - t0, 2), sink_rate=round(sink, 2), max_tilt_deg=round(tilt, 1),
                   impact_vib_z=round(impact, 1),
                   outcome="hard_landing" if (sink > 1.5 or tilt > 45.0) else "soft_landing")
    if "alarm" in df:
        al = after[after.alarm == 1]
        res["first_alarm_s"] = None if al.empty else round(al.timestamp.iloc[0] - t0, 2)
    if "sup_state" in df:
        d = after[after.sup_state == "DESCEND"]
        res["descend_s"] = None if d.empty else round(d.timestamp.iloc[0] - t0, 2)
        res["alt_at_descend"] = None if d.empty else round(float(d.rel_alt.iloc[0]), 2)
        pre = df[(df.timestamp < t0) & (df.armed == 1) & (df.rel_alt > 1.0)]
        holds = int(((pre.sup_state == "HOLD") & (pre.sup_state.shift() == "NORMAL")).sum())
        res.update(false_alarm_holds=holds, normal_minutes=round(len(pre) / 600, 2))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--events")
    ap.add_argument("--severity", help="flight_severity.csv of the runs")
    ap.add_argument("--baseline", nargs="*", default=[])
    ap.add_argument("--baseline-events")
    ap.add_argument("--baseline-severity")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for group, files, evp, sevp in (("supervisor", a.runs, a.events, a.severity),
                                    ("baseline", a.baseline, a.baseline_events, a.baseline_severity)):
        if not files:
            continue
        ev = pd.read_csv(evp)
        sev = dict(pd.read_csv(sevp, header=None, dtype=str).values)
        for f in sorted(files):
            r = outcome(pd.read_csv(f), ev, sev.get(Path(f).stem.split("_")[1]))
            if r:
                rows.append({"group": group, "flight": Path(f).stem, **r})
    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False)
    print(t.to_string(index=False))
    print()
    print(t.groupby(["group", "loe_final", "outcome"]).size().unstack(fill_value=0).to_string())
    if "false_alarm_holds" in t:
        s = t[t.group == "supervisor"]
        print(f"\nfalse-alarm holds: {int(s.false_alarm_holds.sum())} in {s.normal_minutes.sum():.1f} min of normal flight")


if __name__ == "__main__":
    main()
