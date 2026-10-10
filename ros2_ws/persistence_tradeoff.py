"""Alarm-persistence trade-off for the GRU-5 and LSTM-20 classifiers: for each persistence k, the threshold is picked
on the VALIDATION flights (<= 0.5 false alarms / min), then detection and false alarms are measured on TEST.
Operating point (written to operating_point.json for GRU-5) is chosen on VALIDATION metrics only: most detections
within the precursor, then lowest median latency."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/root/ros2_ws")
import train_fault_models as T
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
D = os.environ.get("MODEL_DIR", "/root/ros2_ws/models_gru")
pre = np.load(f"{D}/preprocessing.npz"); T.set_feature_set(str(pre["feature_set"]) if "feature_set" in pre.files else "v1"); met = json.load(open(f"{D}/metrics.json"))
df = pd.read_csv(os.environ.get("DATA", "/root/ros2_ws/flight_data_labeled.csv"))
rows = []
for name, seq in (("cls_gru5", 5), ("cls_lstm20", 20)):
    m = tf.keras.models.load_model(f"{D}/{name}.h5", compile=False)
    sc = {}
    for s in ("val", "test"):
        X, info = T.flight_windows(df[df.flight.isin(met["split"][s])], pre["median"], pre["scale"], seq, pre["feature_index"])
        keep = info.target.to_numpy() >= 0
        sc[s] = (m.predict(X, verbose=0)[:, 0][keep], info[keep].reset_index(drop=True))
    for k in (2, 3, 5, 8):
        T.PERSIST = k
        thr = T.pick_threshold(*sc["val"])
        v = T.episode_metrics(T.alarms_with_persistence(sc["val"][0], thr, sc["val"][1]), sc["val"][1])
        r = T.episode_metrics(T.alarms_with_persistence(sc["test"][0], thr, sc["test"][1]), sc["test"][1])
        rows.append({"model": name, "persistence": k, "persist_s": k / 10, "threshold": round(thr, 3),
                     "val_detected": v["detected"], "val_in_precursor": v["detected_in_precursor"],
                     "val_latency_s": None if v["latency_median_s"] is None else round(v["latency_median_s"], 2),
                     "val_fa_per_min": round(v["false_alarms_per_min"], 2),
                     "detected": f"{r['detected']}/{r['episodes']}", "in_precursor": r["detected_in_precursor"],
                     "latency_median_s": None if r["latency_median_s"] is None else round(r["latency_median_s"], 2),
                     "fa_per_min_test": round(r["false_alarms_per_min"], 2)})
t = pd.DataFrame(rows); print(t.to_string(index=False)); t.to_csv(f"{D}/persistence_tradeoff.csv", index=False)
g = t[t.model == "cls_gru5"].copy()
g["lat"] = g.val_latency_s.fillna(1e9)
best = g.sort_values(["val_in_precursor", "lat"], ascending=[False, True]).iloc[0]
op = {"model": "cls_gru5_int8.tflite", "source_h5": "cls_gru5.h5", "threshold": float(best.threshold),
      "persistence": int(best.persistence),
      "selection": "persistence chosen on VALIDATION flights only (most detections within the precursor, then lowest "
                   "median latency); threshold per persistence picked on validation for <= 0.5 false alarms / min"}
json.dump(op, open(f"{D}/operating_point.json", "w"), indent=1)
print("operating point:", op)
