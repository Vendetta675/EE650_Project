"""Where do the GRU-5 classifier's false alarms on the test flights happen, and how does detection depend on severity?"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/root/ros2_ws")
import train_fault_models as T
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
D = os.environ.get("MODEL_DIR", "/root/ros2_ws/models_gru")
pre = np.load(f"{D}/preprocessing.npz"); T.set_feature_set(str(pre["feature_set"]) if "feature_set" in pre.files else "v1"); met = json.load(open(f"{D}/metrics.json"))
df = pd.read_csv(os.environ.get("DATA", "/root/ros2_ws/flight_data_labeled.csv"))
test = met["split"]["test"]
sev = dict(pd.read_csv(os.environ.get("SEVERITY", "/root/ros2_ws/flights/flight_severity.csv"), header=None, dtype=str).values)
for name, seq in (("cls_gru5", 5), ("cls_lstm20", 20)):
    m = tf.keras.models.load_model(f"{D}/{name}.h5", compile=False)
    X, info = T.flight_windows(df[df.flight.isin(test)], pre["median"], pre["scale"], seq, pre["feature_index"])
    p = m.predict(X, verbose=0)[:, 0]
    thr = met["results"][name]["threshold"]
    keep = info.target.to_numpy() >= 0
    it = info[keep].reset_index(drop=True); pk = p[keep]
    al = T.alarms_with_persistence(pk, thr, it)
    onset = np.zeros(len(it), bool)
    for _, idx in it.groupby("flight").indices.items():
        a = al[idx].astype(int); onset[idx] = np.diff(np.concatenate([[0], a])) == 1
    fa = it[onset & (it.phase == 0)].copy()
    # time to the next fault start in the same flight
    starts = df[(df.phase == 1)].groupby(["flight", "episode"]).timestamp.min().reset_index()
    def to_next(r):
        s = starts[(starts.flight == r.flight) & (starts.timestamp > r.timestamp)].timestamp
        return float(s.min() - r.timestamp) if len(s) else np.nan
    fa["s_before_next_fault"] = fa.apply(to_next, axis=1)
    print(f"== {name}: {len(fa)} false-alarm onsets on test normal rows")
    print(fa.groupby("flight").size().to_string())
    print("seconds before next fault start (NaN = after last fault):", np.round(fa.s_before_next_fault.to_numpy(), 1).tolist())
    lat = []
    for (f, e), g in it[it.phase.isin([1, 2])].groupby(["flight", "episode"]):
        h = g[al[g.index]]
        lat.append((f, sev[f.split("_")[1]], round(float(h.t_since_start.iloc[0]), 2) if len(h) else None))
    print("latency per episode (flight, loe_final, s):", lat)
