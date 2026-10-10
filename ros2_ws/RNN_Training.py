import json
import os
import random

import numpy as np
import pandas as pd

# Work around NumPy 1.26 aliases removed while this image bundles SciPy code
# expecting them during TensorFlow import.
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed

DATA = "/root/ros2_ws/flight_data_normal.csv"
OUT = "/root/ros2_ws/motor_failure_lstm"
SEQ_LEN = 20
SEED = 42

os.makedirs(OUT, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.set_visible_devices([], "GPU")

frame = pd.read_csv(DATA).sort_values("timestamp").reset_index(drop=True)
features = ["vib_x", "vib_y", "vib_z", "m1", "m2", "m3", "m4", "ekf_vel_var", "ekf_pos_var", "ekf_comp_var"]
values = frame[features].replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").to_numpy(dtype=np.float32)
median = np.nanmedian(values, axis=0)
q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
scale = q75 - q25
scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
scaled = np.clip((values - median) / scale, -10.0, 10.0).astype(np.float32)

# Chronological split keeps overlapping windows from leaking across partitions.
def windows(a):
    return np.stack([a[i:i + SEQ_LEN] for i in range(len(a) - SEQ_LEN + 1)])

n = len(scaled)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
train = windows(scaled[:train_end])
val = windows(scaled[train_end - SEQ_LEN + 1:val_end])
test = windows(scaled[val_end - SEQ_LEN + 1:])

model = Sequential([
    Input(shape=(SEQ_LEN, len(features))),
    LSTM(64, activation="tanh", return_sequences=False),
    RepeatVector(SEQ_LEN),
    LSTM(64, activation="tanh", return_sequences=True),
    TimeDistributed(Dense(len(features))),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
history = model.fit(
    train, train, validation_data=(val, val), epochs=50, batch_size=32,
    callbacks=[EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
               ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5)],
    verbose=2,
)

val_pred = model.predict(val, verbose=0)
val_errors = np.mean(np.square(val - val_pred), axis=(1, 2))
threshold = float(np.quantile(val_errors, 0.99))
test_pred = model.predict(test, verbose=0)
test_errors = np.mean(np.square(test - test_pred), axis=(1, 2))
model.save(os.path.join(OUT, "lstm_autoencoder.keras"))
np.savez(os.path.join(OUT, "preprocessing.npz"), features=np.array(features), median=median, scale=scale, seq_len=SEQ_LEN, threshold=threshold)
pd.DataFrame({"window_end_row": np.arange(val_end, n), "reconstruction_mse": test_errors, "anomaly": test_errors > threshold}).to_csv(os.path.join(OUT, "test_anomaly_scores.csv"), index=False)
summary = {
    "dataset": DATA, "rows": n, "label_values": sorted(pd.unique(frame["label"]).tolist()),
    "task": "normal-only LSTM sequence autoencoder for anomaly scoring",
    "features": features, "sequence_length": SEQ_LEN,
    "train_windows": int(len(train)), "validation_windows": int(len(val)), "test_windows": int(len(test)),
    "epochs_run": len(history.history["loss"]), "best_validation_mse": float(min(history.history["val_loss"])),
    "anomaly_threshold_mse_99th_percentile_validation": threshold,
    "test_mse_mean": float(test_errors.mean()), "test_mse_median": float(np.median(test_errors)),
    "test_windows_above_threshold": int(np.sum(test_errors > threshold)),
    "model": os.path.join(OUT, "lstm_autoencoder.keras"),
    "preprocessing": os.path.join(OUT, "preprocessing.npz"),
    "test_scores": os.path.join(OUT, "test_anomaly_scores.csv"),
}
with open(os.path.join(OUT, "training_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
