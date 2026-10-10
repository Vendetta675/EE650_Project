"""Per-episode detection latency on the test flights: GRU-5 with v1 features (motors + vibration) vs v2 features
(+ IMU / attitude rates), trained on the same flights, each at its validation-chosen operating point
(operating_point.json: threshold, persistence). Keras models (the INT8 v2 model matches Keras at its operating point)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/root/ros2_ws")
import train_fault_models as T
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
df = pd.read_csv("/root/ros2_ws/flight_data_labeled_v2.csv")
sev = dict(pd.read_csv("/root/ros2_ws/flights_v2/flight_severity.csv", header=None, dtype=str).values)
lat = {}
for fs in ("v1", "v2"):
    D = f"/root/ros2_ws/models_imu_{fs}"
    pre = np.load(f"{D}/preprocessing.npz"); met = json.load(open(f"{D}/metrics.json")); op = json.load(open(f"{D}/operating_point.json"))
    T.set_feature_set(fs); T.PERSIST = op["persistence"]
    m = tf.keras.models.load_model(f"{D}/cls_gru5.h5", compile=False)
    X, info = T.flight_windows(df[df.flight.isin(met["split"]["test"])], pre["median"], pre["scale"], 5, pre["feature_index"])
    keep = info.target.to_numpy() >= 0
    it = info[keep].reset_index(drop=True)
    thr = op.get("test_keras_same_rule", {}).get("threshold", op["threshold"])
    al = T.alarms_with_persistence(m.predict(X, verbose=0)[:, 0][keep], thr, it)
    for (f, e), g in it[it.phase.isin([1, 2])].groupby(["flight", "episode"]):
        h = g[al[g.index]]
        lat.setdefault((f, e), {"loe": sev[f.split("_")[1]]})[fs] = round(float(h.t_since_start.iloc[0]), 2) if len(h) else None
t = pd.DataFrame([{"flight": f, "episode": e, **v} for (f, e), v in lat.items()])
t["delta_s"] = t.v2 - t.v1
print(t.to_string(index=False))
both = t.dropna(subset=["v1", "v2"])
print(f"\npaired episodes detected by both: {len(both)}; median v2 - v1 = {both.delta_s.median():+.2f} s; "
      f"v2 earlier in {(both.delta_s < 0).sum()}, later in {(both.delta_s > 0).sum()}")
print("detected only by v1:", int((t.v1.notna() & t.v2.isna()).sum()), "| only by v2:", int((t.v2.notna() & t.v1.isna()).sum()))
t.to_csv("/root/ros2_ws/models_imu_v2/latency_v1_vs_v2.csv", index=False)
