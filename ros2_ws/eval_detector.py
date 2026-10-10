"""Replay labelled flights through the deployed runtime (detector.FaultDetector: INT8 TFLite, scaler, alarm rule)
row by row, as on the vehicle, and report episode metrics overall and per airframe / battery state of charge.
Airborne segments only (as in training); a new segment restarts the detector.
Usage: python3 eval_detector.py --model-dir models_imu_v2 --data flight_data_labeled_v3.csv --flights flight_05 ...
       [--plan flights_v3/plan.csv] [--out scores.csv]"""
import argparse, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/root/ros2_ws/src/motor_failure_predictor")
from motor_failure_predictor.detector import FaultDetector

def replay(model_dir, df, **kw):
    out = []
    for fl, g in df.groupby("flight", sort=True):
        g = g.sort_values("timestamp")
        seg = ((g.timestamp.diff() > 0.5) | (g.airborne.diff().fillna(0) != 0)).cumsum()
        for _, s in g[g.airborne == 1].groupby(seg[g.airborne == 1]):
            det = FaultDetector(model_dir, **kw)
            raw = s[det.raw_fields].ffill().bfill().to_numpy(np.float32)
            for r, (_, row) in zip(raw, s.iterrows()):
                p, a = det.step(r)
                out.append((fl, row.timestamp, row.phase, row.episode, row.t_since_start, np.nan if p is None else p, a))
    return pd.DataFrame(out, columns=["flight", "timestamp", "phase", "episode", "t_since_start", "prob", "alarm"])

def metrics(sc, ramp_s=5.0):
    lat, n_ep = [], 0
    for _, g in sc[sc.phase.isin([1, 2])].groupby(["flight", "episode"]):
        n_ep += 1
        hit = g[g.alarm]
        if len(hit):
            lat.append(float(hit.t_since_start.iloc[0]))
    onsets, minutes = 0, 0.0
    for _, g in sc[sc.phase == 0].groupby("flight"):
        a = g.alarm.to_numpy(int)
        onsets += int(np.sum(np.diff(np.concatenate([[0], a])) == 1)); minutes += len(g) / 600
    lat = np.array(lat)
    return {"episodes": n_ep, "detected": len(lat), "in_precursor": int((lat < ramp_s).sum()),
            "latency_median_s": round(float(np.median(lat)), 2) if len(lat) else None,
            "fa_per_min": round(onsets / max(minutes, 1e-9), 2), "fa_onsets": onsets, "normal_min": round(minutes, 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--flights", nargs="+", required=True); ap.add_argument("--plan", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.data); df = df[df.flight.isin(a.flights)]
    sc = replay(a.model_dir, df)
    if a.out:
        sc.to_csv(a.out, index=False)
    rows = [("all", metrics(sc))]
    if a.plan:
        plan = pd.read_csv(a.plan, dtype={"flight": str}); plan["flight"] = "flight_" + plan.flight
        sc = sc.merge(plan[["flight", "airframe", "soc"]], on="flight", how="left")
        for af in ("stock", "light"):
            rows.append((af, metrics(sc[sc.airframe == af])))
        rows.append(("SoC < 0.7", metrics(sc[sc.soc < 0.7]))); rows.append(("SoC >= 0.7", metrics(sc[sc.soc >= 0.7])))
    print(pd.DataFrame([{"subset": k, **v} for k, v in rows]).to_string(index=False))

if __name__ == "__main__":
    main()
