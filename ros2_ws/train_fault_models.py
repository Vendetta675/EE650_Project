"""Train early motor-fault detectors on precursor-labelled flight data and compare LSTM vs GRU, 20 vs 5 steps.

Data: flight_data_labeled.csv (label_precursors.py). Airborne rows only. Split by flight (no flight is shared
between train / val / test). Features: fault_features.py (relative PWM, vibration, current, PWM spread),
robust-scaled with statistics of NORMAL training rows only.

Models (all: one recurrent layer of UNITS, Adam 1e-3, early stopping on validation loss, seed 0 or $SEED):
  classifier  RNN -> Dropout(0.2) -> Dense(16, relu) -> Dense(1, sigmoid); target = fault developing/present at
              the last step of the window (precursor labelling: 1 from the first instant of degradation).
              Class-weighted binary cross-entropy.
  autoencoder RNN encoder -> RepeatVector -> RNN decoder -> TimeDistributed(Dense); trained on normal windows only;
              alarm threshold = 99th percentile of the reconstruction MSE on normal validation windows.
Variants: {LSTM, GRU} x {SEQ 20 (2 s), SEQ 5 (0.5 s)} for the classifier; LSTM-20 (previous design) and GRU-5
for the autoencoder.

Metrics on the test flights:
  window-level   ROC-AUC, PR-AUC, precision / recall at the operating threshold
  episode-level  detection latency = first alarm (persistence: 2 consecutive windows) after the fault starts,
                 detected-during-precursor rate (alarm before the 5 s ramp ends), false alarms per minute of
                 normal flight (alarm onsets in phase-0 rows). Operating threshold for the classifier: the
                 smallest probability threshold giving <= 0.5 false alarms / min on validation flights.
Outputs (out_dir): <name>.h5 for every model, metrics.json, comparison table on stdout, preprocessing.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fault_features import (FEATURES, FEATURES_V2, FEATURES_V3, RAW_FIELDS, RAW_FIELDS_V2, RAW_FIELDS_V3,  # noqa: E402
                            SAMPLE_HZ, apply_scaler, compute_features, compute_features_v2, compute_features_v3,
                            fit_scaler, windows)

FEATURE_SETS = {"v1": (RAW_FIELDS, compute_features, FEATURES), "v2": (RAW_FIELDS_V2, compute_features_v2, FEATURES_V2),
                "v3": (RAW_FIELDS_V3, compute_features_v3, FEATURES_V3)}
RAW, COMPUTE, FEATS = FEATURE_SETS["v1"]


def set_feature_set(name):
    """v1 = motors + vibration (+ current); v2 = v1 + IMU / attitude-rate features; v3 = v2 + battery-aware
    features (fault_features.py)."""
    global RAW, COMPUTE, FEATS
    RAW, COMPUTE, FEATS = FEATURE_SETS[str(name)]

UNITS = 32
PERSIST = 2


def rnn(kind, units, **kw):
    return (tf.keras.layers.GRU if kind == "gru" else tf.keras.layers.LSTM)(units, **kw)


def build_classifier(kind, seq, nf, unroll=False, batch_size=None):
    return tf.keras.Sequential([
        tf.keras.layers.Input((seq, nf), batch_size=batch_size),
        rnn(kind, UNITS, unroll=unroll),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def build_autoencoder(kind, seq, nf, unroll=False, batch_size=None):
    return tf.keras.Sequential([
        tf.keras.layers.Input((seq, nf), batch_size=batch_size),
        rnn(kind, UNITS, unroll=unroll),
        tf.keras.layers.RepeatVector(seq),
        rnn(kind, UNITS, return_sequences=True, unroll=unroll),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(nf)),
    ])


def flight_windows(df, median, scale, seq, cols):
    """Per-flight windows so no window spans two flights. Returns X, and per-window row info (last row)."""
    Xs, info = [], []
    for _, g in df.groupby("flight", sort=True):
        g = g.sort_values("timestamp")
        # airborne segments only; break windows at gaps (ground rows / recorder hiccups > 0.5 s)
        seg = ((g["timestamp"].diff() > 0.5) | (g["airborne"].diff().fillna(0) != 0)).cumsum()
        for _, s in g[g.airborne == 1].groupby(seg[g.airborne == 1]):
            raw = s[RAW].ffill().bfill().to_numpy(np.float32)
            if len(raw) < seq:
                continue
            X = windows(apply_scaler(COMPUTE(raw)[:, cols], median, scale), seq)
            Xs.append(X)
            info.append(s.iloc[seq - 1:][["flight", "timestamp", "phase", "target", "t_since_start", "episode",
                                          "fault_motor"]])
    return np.concatenate(Xs), pd.concat(info, ignore_index=True)


def alarms_with_persistence(score, thr, info):
    """Boolean alarm per window: score > thr for PERSIST consecutive windows of the same flight."""
    above = score > thr
    out = np.zeros_like(above)
    for _, idx in info.groupby("flight").indices.items():
        a = above[idx].astype(int)
        run = np.convolve(a, np.ones(PERSIST, int), mode="full")[:len(a)]
        out[idx] = run >= PERSIST
    return out


def episode_metrics(alarm, info, ramp_s=5.0):
    lat, in_precursor, n_ep = [], 0, 0
    for (_, _), g in info[info.phase.isin([1, 2])].groupby(["flight", "episode"]):
        n_ep += 1
        hit = g[alarm[g.index]]
        if len(hit):
            l = float(hit.t_since_start.iloc[0])
            lat.append(l)
            in_precursor += l < ramp_s
    normal = info.phase == 0
    onsets = 0
    for _, idx in info[normal].groupby("flight").indices.items():
        sel = info[normal].index[idx]
        a = alarm[sel].astype(int)
        onsets += int(np.sum(np.diff(np.concatenate([[0], a])) == 1))
    minutes = normal.sum() / SAMPLE_HZ / 60
    return {"episodes": n_ep, "detected": len(lat), "detected_in_precursor": int(in_precursor),
            "latency_median_s": float(np.median(lat)) if lat else None,
            "latency_p90_s": float(np.percentile(lat, 90)) if lat else None,
            "false_alarms_per_min": onsets / max(minutes, 1e-9), "normal_minutes": float(minutes)}


def pick_threshold(score_val, info_val, max_fa_per_min=0.5):
    for thr in np.unique(np.quantile(score_val, np.linspace(0.5, 0.9999, 400))):
        if episode_metrics(alarms_with_persistence(score_val, thr, info_val), info_val)["false_alarms_per_min"] <= max_fa_per_min:
            return float(thr)
    return float(np.max(score_val))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="flight_data_labeled.csv")
    ap.add_argument("--out-dir", default="models_gru")
    ap.add_argument("--val-flights", nargs="+", required=True)
    ap.add_argument("--test-flights", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="v1")
    ap.add_argument("--models", nargs="+", default=["cls_lstm20", "cls_gru20", "ae_lstm20", "cls_lstm5", "cls_gru5", "ae_gru5"])
    a = ap.parse_args()
    set_feature_set(a.features)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tf.config.set_visible_devices([], "GPU")
    from sklearn.metrics import average_precision_score, roc_auc_score

    df = pd.read_csv(a.data)
    split = np.where(df.flight.isin(a.test_flights), "test", np.where(df.flight.isin(a.val_flights), "val", "train"))
    df["split"] = split
    tr_norm = df[(df.split == "train") & (df.airborne == 1) & (df.phase == 0)]
    f_all = COMPUTE(df[(df.split == "train") & (df.airborne == 1)][RAW].ffill().bfill().to_numpy(np.float32))
    # drop features that are constant in training (e.g. battery current when the simulator does not model it):
    # a dead input carries no information and would be out of distribution on a vehicle that reports it
    cols = np.where(np.nanstd(f_all, axis=0) > 1e-6)[0]
    dropped = [FEATS[i] for i in range(len(FEATS)) if i not in cols]
    print(f"feature set {a.features}; used:", [FEATS[i] for i in cols], "| dropped (constant in training):", dropped,
          flush=True)
    median, scale = fit_scaler(COMPUTE(tr_norm[RAW].ffill().bfill().to_numpy(np.float32))[:, cols])
    np.savez(out / "preprocessing.npz", features=np.array([FEATS[i] for i in cols]), feature_index=cols,
             raw_fields=np.array(RAW), median=median, scale=scale, dropped=np.array(dropped),
             feature_set=np.array(a.features))

    results = {}
    for seq in (20, 5):
        X, info = {}, {}
        for s in ("train", "val", "test"):
            X[s], info[s] = flight_windows(df[df.split == s], median, scale, seq, cols)
        keep = {s: info[s].target.to_numpy() >= 0 for s in X}
        y = {s: info[s].target.to_numpy() for s in X}
        nf = X["train"].shape[-1]
        pos = y["train"][keep["train"]].mean()
        cw = {0: 0.5 / (1 - pos), 1: 0.5 / pos}
        for kind in ("lstm", "gru"):
            name = f"cls_{kind}{seq}"
            if name not in a.models:
                continue
            tf.keras.utils.set_random_seed(int(os.environ.get("SEED", 0)))
            m = build_classifier(kind, seq, nf)
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
            m.fit(X["train"][keep["train"]], y["train"][keep["train"]], epochs=a.epochs, batch_size=64,
                  class_weight=cw, verbose=0,
                  validation_data=(X["val"][keep["val"]], y["val"][keep["val"]]),
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
            m.save(out / f"{name}.h5")
            sv = m.predict(X["val"], verbose=0)[:, 0]
            st = m.predict(X["test"], verbose=0)[:, 0]
            thr = pick_threshold(sv[keep["val"]], info["val"][keep["val"]].reset_index(drop=True))
            k = keep["test"]
            it = info["test"][k].reset_index(drop=True)
            results[name] = {"params": int(m.count_params()), "seq_len": seq, "threshold": thr,
                             "test_roc_auc": float(roc_auc_score(y["test"][k], st[k])),
                             "test_pr_auc": float(average_precision_score(y["test"][k], st[k])),
                             **episode_metrics(alarms_with_persistence(st[k], thr, it), it)}
            print(name, json.dumps(results[name]), flush=True)
        for kind in (("lstm",) if seq == 20 else ("gru",)):
            name = f"ae_{kind}{seq}"
            if name not in a.models:
                continue
            tf.keras.utils.set_random_seed(int(os.environ.get("SEED", 0)))
            m = build_autoencoder(kind, seq, nf)
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
            ntr = X["train"][info["train"].phase.to_numpy() == 0]
            nva = X["val"][info["val"].phase.to_numpy() == 0]
            m.fit(ntr, ntr, epochs=a.epochs, batch_size=64, verbose=0, validation_data=(nva, nva),
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
            m.save(out / f"{name}.h5")
            err = lambda Z: np.mean(np.square(Z - m.predict(Z, verbose=0)), axis=(1, 2))  # noqa: E731
            thr = float(np.quantile(err(nva), 0.99))
            k = keep["test"]
            st = err(X["test"])
            it = info["test"][k].reset_index(drop=True)
            results[name] = {"params": int(m.count_params()), "seq_len": seq, "threshold": thr,
                             "test_roc_auc": float(roc_auc_score(y["test"][k], st[k])),
                             "test_pr_auc": float(average_precision_score(y["test"][k], st[k])),
                             **episode_metrics(alarms_with_persistence(st[k], thr, it), it)}
            print(name, json.dumps(results[name]), flush=True)
        np.save(out / f"repr_windows_seq{seq}.npy", X["train"][:2000])  # representative set for INT8 calibration
        np.savez(out / f"test_windows_seq{seq}.npz", X=X["test"], target=y["test"])

    (out / "metrics.json").write_text(json.dumps({"split": {"val": a.val_flights, "test": a.test_flights},
                                                  "feature_set": a.features,
                                                  "features": [FEATS[i] for i in cols], "dropped": dropped,
                                                  "units": UNITS, "persistence": PERSIST, "results": results}, indent=1))
    cols = ["params", "test_roc_auc", "test_pr_auc", "detected", "episodes", "detected_in_precursor",
            "latency_median_s", "false_alarms_per_min"]
    print(pd.DataFrame(results).T[cols].to_string())


if __name__ == "__main__":
    main()
