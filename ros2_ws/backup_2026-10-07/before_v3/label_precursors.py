"""Precursor labelling: join recorded flights with the failure plugin's ground-truth event log.

Each 10 Hz row of flights/flight_XX.csv gets, from fault_events.csv (same container clock):
  phase        0 normal | 1 precursor = the LOE ramp [start, start + ramp_duration) | 2 degraded = after the
               ramp until recover (or cut) | 3 recovery transient = RECOVERY_S after recover | 4 after a cut
  target       1 for phases 1-2 (fault developing / present), 0 for phase 0, -1 (excluded) for phases 3-4.
               The anomaly is flagged from the first moment of degradation, not when thrust is lost, so the
               models learn the early warning signs.
  fault_motor  motor index 0-3 (= m1..m4) of the active fault, -1 otherwise
  effectiveness  k(t) reconstructed from the logged ramp parameters (1.0 outside faults)
  t_since_start  seconds since the fault started (NaN outside faults)
  airborne     armed and relative altitude > 1 m (only airborne rows are used for training/evaluation)
  episode      running id of the fault episode within the flight (rows before episode e belong to e)
Usage: python3 label_precursors.py --flights flights/*.csv --events fault_events.csv --out flight_data_labeled.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RECOVERY_S = 5.0


def episodes_from_events(ev: pd.DataFrame) -> list[dict]:
    out, cur = [], None
    for r in ev.sort_values("wall_time").itertuples():
        if r.event == "start":
            cur = dict(start=r.wall_time, motor=int(r.motor_index), ramp=float(r.ramp_duration),
                       loe=float(r.loe_final), cut=np.nan, end=np.nan)
        elif cur is not None and r.event == "cut":
            cur["cut"] = r.wall_time
        elif cur is not None and r.event == "recover":
            cur["end"] = r.wall_time
            out.append(cur)
            cur = None
    if cur is not None:  # still active at the end of the log
        out.append(cur)
    return out


def label_flight(df: pd.DataFrame, eps: list[dict], flight: str) -> pd.DataFrame:
    t = df["timestamp"].to_numpy()
    phase = np.zeros(len(df), int)
    motor = np.full(len(df), -1)
    k = np.ones(len(df))
    since = np.full(len(df), np.nan)
    episode = np.zeros(len(df), int)
    mine = [e for e in eps if t[0] - 1 <= e["start"] <= t[-1]]
    for i, e in enumerate(mine):
        end = e["end"] if np.isfinite(e["end"]) else t[-1] + 1
        ramp_end = e["start"] + e["ramp"]
        m_fault = (t >= e["start"]) & (t < end)
        phase[m_fault & (t < ramp_end)] = 1
        phase[m_fault & (t >= ramp_end)] = 2
        if np.isfinite(e["cut"]):
            phase[(t >= e["cut"]) & (t < end)] = 4
        phase[(t >= end) & (t < end + RECOVERY_S)] = 3
        motor[m_fault] = e["motor"]
        el = t[m_fault] - e["start"]
        since[m_fault] = el
        k[m_fault] = 1.0 - (1.0 - e["loe"]) * np.clip(el / e["ramp"], 0, 1)
        episode[t >= (mine[i - 1]["end"] if i else -np.inf)] = i
    out = df.copy()
    out["flight"] = flight
    out["phase"] = phase
    out["target"] = np.where(phase == 0, 0, np.where(np.isin(phase, [1, 2]), 1, -1))
    out["fault_motor"] = motor
    out["effectiveness"] = k
    out["t_since_start"] = since
    out["episode"] = episode
    out["airborne"] = ((out["armed"] == 1) & (out["rel_alt"] > 1.0)).astype(int)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flights", nargs="+", required=True)
    ap.add_argument("--events", default="fault_events.csv")
    ap.add_argument("--out", default="flight_data_labeled.csv")
    a = ap.parse_args()
    eps = episodes_from_events(pd.read_csv(a.events))
    parts = []
    for f in sorted(a.flights):
        df = pd.read_csv(f).sort_values("timestamp").reset_index(drop=True)
        parts.append(label_flight(df, eps, Path(f).stem))
    lab = pd.concat(parts, ignore_index=True)
    lab.to_csv(a.out, index=False)
    air = lab[lab.airborne == 1]
    print(f"{len(eps)} fault episodes in the event log; {len(lab)} rows, {len(air)} airborne -> {a.out}")
    print(air.groupby(["flight", "phase"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
