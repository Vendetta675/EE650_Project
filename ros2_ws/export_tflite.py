"""Export a trained Keras .h5 detector to TensorFlow Lite with full-integer INT8 quantization, check parity and
benchmark latency.

The recurrent model is rebuilt with unroll=True and a fixed batch size of 1 (5 steps -> plain FULLY_CONNECTED / MUL / ADD / LOGISTIC / TANH
ops that TFLite quantizes to int8) and the trained weights are copied over. Calibration: representative windows
from the training flights (repr_windows_seq<L>.npy written by train_fault_models.py). Input and output tensors are
int8; the caller quantizes with the input (scale, zero_point) and dequantizes the output (see lstm_node.py).

Reports: file sizes (.h5, fp32 .tflite, int8 .tflite), max / mean |p_int8 - p_keras| and ROC-AUC of both on the
test windows, and median single-window latency (1 thread, 2,000 runs) for Keras eager call, TFLite fp32 and
TFLite int8. Latency is measured on this machine's CPU (x86); ARM / Jetson numbers must be measured on device.
Usage: python3 export_tflite.py --model models_gru/cls_gru5.h5 --kind classifier --seq 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_fault_models import build_autoencoder, build_classifier  # noqa: E402


def convert(model, repr_x, int8):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([x[None].astype(np.float32)] for x in repr_x[:500])
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


class Lite:
    def __init__(self, blob, threads=1):
        self.it = tf.lite.Interpreter(model_content=blob, num_threads=threads)
        self.it.allocate_tensors()
        self.i = self.it.get_input_details()[0]
        self.o = self.it.get_output_details()[0]

    def __call__(self, x):
        if self.i["dtype"] == np.int8:
            s, z = self.i["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        self.it.set_tensor(self.i["index"], x.astype(self.i["dtype"]))
        self.it.invoke()
        y = self.it.get_tensor(self.o["index"])
        if self.o["dtype"] == np.int8:
            s, z = self.o["quantization"]
            y = (y.astype(np.float32) - z) * s
        return y


def bench(fn, x, n=2000):
    for _ in range(50):
        fn(x)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(x)
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1e6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["classifier", "autoencoder"], required=True)
    ap.add_argument("--rnn", default="gru")
    ap.add_argument("--seq", type=int, default=5)
    a = ap.parse_args()
    tf.config.set_visible_devices([], "GPU")
    from sklearn.metrics import roc_auc_score
    src = Path(a.model)
    d = src.parent
    trained = tf.keras.models.load_model(src, compile=False)
    nf = trained.input_shape[-1]
    build = build_classifier if a.kind == "classifier" else build_autoencoder
    unrolled = build(a.rnn, a.seq, nf, unroll=True, batch_size=1)  # fixed batch 1: no dynamic-shape ops
    unrolled.set_weights(trained.get_weights())
    repr_x = np.load(d / f"repr_windows_seq{a.seq}.npy")
    test = np.load(d / f"test_windows_seq{a.seq}.npz")
    X, y = test["X"], test["target"]
    keep = y >= 0

    blobs = {"fp32": convert(unrolled, repr_x, False), "int8": convert(unrolled, repr_x, True)}
    paths = {}
    for k, b in blobs.items():
        paths[k] = d / f"{src.stem}_{k}.tflite"
        paths[k].write_bytes(b)

    def score_keras(Z):
        p = trained.predict(Z, verbose=0, batch_size=512)
        return p[:, 0] if a.kind == "classifier" else np.mean(np.square(Z - p), axis=(1, 2))

    def score_lite(lite, Z):
        out = np.stack([lite(z[None])[0] for z in Z])
        return out[:, 0] if a.kind == "classifier" else np.mean(np.square(Z - out), axis=(1, 2))

    sk = score_keras(X)
    rep = {"model": str(src), "kind": a.kind, "params": int(trained.count_params()),
           "size_bytes": {"h5": src.stat().st_size, **{k: p.stat().st_size for k, p in paths.items()}},
           "test_roc_auc": {"keras": float(roc_auc_score(y[keep], sk[keep]))}}
    x1 = X[:1].astype(np.float32)
    rep["latency_us_1thread"] = {"keras_call": bench(lambda z: trained(z, training=False), x1, 500)}
    for k in ("fp32", "int8"):
        lite = Lite(blobs[k])
        s = score_lite(lite, X)
        rep["test_roc_auc"][k] = float(roc_auc_score(y[keep], s[keep]))
        rep.setdefault("abs_diff_vs_keras", {})[k] = {"max": float(np.max(np.abs(s - sk))),
                                                      "mean": float(np.mean(np.abs(s - sk)))}
        rep["latency_us_1thread"][f"tflite_{k}"] = bench(lite, x1)
        if k == "int8":
            rep["int8_io_quantization"] = {"input": [float(v) for v in lite.i["quantization"]],
                                           "output": [float(v) for v in lite.o["quantization"]]}
            rep["int8_op_types"] = sorted({op["op_name"] for op in lite.it._get_ops_details()})
    (d / f"{src.stem}_tflite_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
