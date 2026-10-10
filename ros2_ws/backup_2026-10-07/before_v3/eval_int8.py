"""Episode-level evaluation of the deployed INT8 TFLite GRU-5 classifier. The threshold is re-picked on INT8 scores of
the validation flights (<= 0.5 false alarms / min, persistence from operating_point.json), then test detection /
false alarms are compared with the Keras model at its own threshold. Updates operating_point.json with the INT8 threshold."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/root/ros2_ws")
import train_fault_models as T
from export_tflite import Lite
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
D = os.environ.get("MODEL_DIR", "/root/ros2_ws/models_gru")
pre = np.load(f"{D}/preprocessing.npz"); T.set_feature_set(str(pre["feature_set"]) if "feature_set" in pre.files else "v1"); met = json.load(open(f"{D}/metrics.json"))
op = json.load(open(f"{D}/operating_point.json"))
df = pd.read_csv(os.environ.get("DATA", "/root/ros2_ws/flight_data_labeled.csv"))
T.PERSIST = op["persistence"]
keras = tf.keras.models.load_model(f"{D}/cls_gru5.h5", compile=False)
lite = Lite(open(f"{D}/cls_gru5_int8.tflite", "rb").read())
sc = {}
for s in ("val", "test"):
    X, info = T.flight_windows(df[df.flight.isin(met["split"][s])], pre["median"], pre["scale"], 5, pre["feature_index"])
    keep = info.target.to_numpy() >= 0
    it = info[keep].reset_index(drop=True)
    sc[s] = {"keras": keras.predict(X, verbose=0)[:, 0][keep],
             "int8": np.array([lite(x[None])[0, 0] for x in X])[keep], "info": it}
out = {}
for kind in ("keras", "int8"):
    thr = T.pick_threshold(sc["val"][kind], sc["val"]["info"])
    r = T.episode_metrics(T.alarms_with_persistence(sc["test"][kind], thr, sc["test"]["info"]), sc["test"]["info"])
    out[kind] = {"threshold": thr, **r}
    print(kind, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[kind].items()}))
op.update(threshold=float(out["int8"]["threshold"]), threshold_source="picked on INT8 validation scores (eval_int8.py)",
          test_int8=out["int8"], test_keras_same_rule=out["keras"])
json.dump(op, open(f"{D}/operating_point.json", "w"), indent=1)
