"""Alarm-rule selection on VALIDATION flights only, for a trained classifier (Keras .h5, same windows as training).
Rules: k-of-n votes over the last n windows (persistence k consecutive = k-of-k). For each rule the threshold is the
smallest one giving <= MAX_FA false alarms / min on validation; the rule with the most validation detections inside
the 5 s precursor, then all detections, then the lowest median latency is written to operating_point.json.
Test metrics are printed for every rule for the record, never used for the choice.
Usage: MODEL_DIR=models_v3 DATA=flight_data_labeled_v3.csv [MODEL=cls_gru5] [MAX_FA=0.5] python3 select_operating_point.py"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/root/ros2_ws")
import train_fault_models as T
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
D = os.environ["MODEL_DIR"]; NAME = os.environ.get("MODEL", "cls_gru5"); MAX_FA = float(os.environ.get("MAX_FA", 0.5))
SEQ = int("".join(c for c in NAME if c.isdigit()))
pre = np.load(f"{D}/preprocessing.npz"); T.set_feature_set(str(pre["feature_set"])); met = json.load(open(f"{D}/metrics.json"))
df = pd.read_csv(os.environ["DATA"])
m = tf.keras.models.load_model(f"{D}/{NAME}.h5", compile=False)
sc = {}
for s in ("val", "test"):
    X, info = T.flight_windows(df[df.flight.isin(met["split"][s])], pre["median"], pre["scale"], SEQ, pre["feature_index"])
    keep = info.target.to_numpy() >= 0
    sc[s] = (m.predict(X, verbose=0)[:, 0][keep], info[keep].reset_index(drop=True))

def vote_alarms(score, thr, info, k, n):
    out = np.zeros(len(score), bool)
    for _, idx in info.groupby("flight").indices.items():
        a = (score[idx] > thr).astype(int)
        out[idx] = np.convolve(a, np.ones(n, int), mode="full")[:len(a)] >= k
    return out

def pick(score, info, k, n):
    for thr in np.unique(np.quantile(score, np.linspace(0.5, 0.9999, 400))):
        if T.episode_metrics(vote_alarms(score, thr, info, k, n), info)["false_alarms_per_min"] <= MAX_FA:
            return float(thr)
    return float(score.max())

rules = [(k, k) for k in (2, 3, 5, 8)] + [(3, 5), (4, 6), (5, 8), (6, 10), (8, 12), (10, 15)]
rows = []
for k, n in rules:
    thr = pick(*sc["val"], k, n)
    v = T.episode_metrics(vote_alarms(*sc["val"][:1], thr, sc["val"][1], k, n), sc["val"][1])
    r = T.episode_metrics(vote_alarms(*sc["test"][:1], thr, sc["test"][1], k, n), sc["test"][1])
    rows.append({"rule": f"{k}-of-{n}", "k": k, "n": n, "threshold": round(thr, 4),
                 "val_detected": f"{v['detected']}/{v['episodes']}", "val_det": v["detected"],
                 "val_in_precursor": v["detected_in_precursor"], "val_latency_s": v["latency_median_s"],
                 "val_fa_min": round(v["false_alarms_per_min"], 2),
                 "test_detected": f"{r['detected']}/{r['episodes']}", "test_in_precursor": r["detected_in_precursor"],
                 "test_latency_s": None if r["latency_median_s"] is None else round(r["latency_median_s"], 2),
                 "test_fa_min": round(r["false_alarms_per_min"], 2)})
t = pd.DataFrame(rows)
print(t.drop(columns=["k", "n", "val_det"]).to_string(index=False))
t["lat"] = t.val_latency_s.fillna(1e9)
best = t.sort_values(["val_in_precursor", "val_det", "lat"], ascending=[False, False, True]).iloc[0]
t.drop(columns=["lat"]).to_csv(f"{D}/alarm_rules_{NAME}.csv", index=False)
op = {"model": f"{NAME}_int8.tflite", "source_h5": f"{NAME}.h5", "threshold": float(best.threshold),
      "persistence": int(best.k), "vote": [int(best.k), int(best.n)],
      "selection": f"alarm rule (k-of-n votes) and threshold chosen on VALIDATION flights only: threshold per rule for "
                   f"<= {MAX_FA} false alarms / min, then most detections within the precursor, most detections, lowest "
                   f"median latency"}
json.dump(op, open(f"{D}/operating_point.json", "w"), indent=1)
print("operating point:", op)
