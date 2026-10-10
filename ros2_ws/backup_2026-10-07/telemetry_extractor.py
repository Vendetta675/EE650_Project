import csv
import os
import time
from collections import deque

import numpy as np
from pymavlink import mavutil

# TensorFlow 2.21 in this image imports an older SciPy that expects these
# NumPy aliases. Defining them before TensorFlow import keeps the model usable
# without changing the container packages.
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int
import tensorflow as tf

WORKSPACE = "/root/ros2_ws"
MODEL_PATH = os.path.join(WORKSPACE, "motor_failure_lstm", "lstm_autoencoder.keras")
PREPROCESSING_PATH = os.path.join(WORKSPACE, "motor_failure_lstm", "preprocessing.npz")
CSV_PATH = os.environ.get("FLIGHT_DATA_OUTPUT", os.path.join(WORKSPACE, "flight_data_live.csv"))
CONNECTION_STRING = os.environ.get("MAVLINK_CONNECTION", "udp:127.0.0.1:14550")
SAMPLE_INTERVAL = 0.1

# This detector is CPU-sized and the container has no usable CUDA runtime.
tf.config.set_visible_devices([], "GPU")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
preprocessing = np.load(PREPROCESSING_PATH)
FEATURES = preprocessing["features"].tolist()
MEDIAN = preprocessing["median"].astype(np.float32)
SCALE = preprocessing["scale"].astype(np.float32)
SEQ_LEN = int(preprocessing["seq_len"])
THRESHOLD = float(preprocessing["threshold"])
window = deque(maxlen=SEQ_LEN)

connection = mavutil.mavlink_connection(CONNECTION_STRING)
connection.wait_heartbeat()
print("Connected to Flight Controller.")


def request_message_interval(message_id, frequency_hz):
    interval_us = int(1e6 / frequency_hz)
    connection.mav.command_long_send(
        connection.target_system, connection.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        message_id, interval_us, 0, 0, 0, 0, 0,
    )


request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_VIBRATION, 10)
request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW, 10)
request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 10)

state = {
    "vib_x": 0.0, "vib_y": 0.0, "vib_z": 0.0,
    "m1": 0.0, "m2": 0.0, "m3": 0.0, "m4": 0.0,
    "ekf_vel_var": 0.0, "ekf_pos_var": 0.0, "ekf_comp_var": 0.0,
}
received = set()
message_fields = {
    "VIBRATION": ("vib_x", "vib_y", "vib_z"),
    "SERVO_OUTPUT_RAW": ("m1", "m2", "m3", "m4"),
    "EKF_STATUS_REPORT": ("ekf_vel_var", "ekf_pos_var", "ekf_comp_var"),
}
fields = ["timestamp"] + FEATURES + ["anomaly_score", "anomaly"]

with open(CSV_PATH, mode="w", newline="", buffering=1) as output:
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    print(f"Logging live anomaly scores to {CSV_PATH}; threshold={THRESHOLD:.6f}.")
    print(f"Waiting for telemetry; scoring starts after {SEQ_LEN} complete samples.")
    last_write_time = time.monotonic()
    last_alert_state = None

    try:
        while True:
            msg = connection.recv_match(
                type=["VIBRATION", "SERVO_OUTPUT_RAW", "EKF_STATUS_REPORT"],
                blocking=True,
            )
            if msg is None:
                continue

            msg_type = msg.get_type()
            if msg_type == "VIBRATION":
                for key, attribute in zip(message_fields[msg_type], ("vibration_x", "vibration_y", "vibration_z")):
                    value = getattr(msg, attribute, None)
                    if value is not None:
                        state[key] = float(value)
                received.add(msg_type)
            elif msg_type == "SERVO_OUTPUT_RAW":
                for key, attribute in zip(message_fields[msg_type], ("servo1_raw", "servo2_raw", "servo3_raw", "servo4_raw")):
                    value = getattr(msg, attribute, None)
                    if value is not None:
                        state[key] = float(value)
                received.add(msg_type)
            elif msg_type == "EKF_STATUS_REPORT":
                for key, attribute in zip(message_fields[msg_type], ("velocity_variance", "pos_horiz_variance", "compass_variance")):
                    value = getattr(msg, attribute, None)
                    if value is not None:
                        state[key] = float(value)
                received.add(msg_type)

            now = time.monotonic()
            if now - last_write_time < SAMPLE_INTERVAL:
                continue
            last_write_time = now

            # Avoid scoring rows built from startup placeholder zeros. Once
            # all message types have arrived, hold their latest values at 10 Hz.
            score = ""
            anomaly = ""
            if len(received) == len(message_fields):
                raw = np.array([state[name] for name in FEATURES], dtype=np.float32)
                scaled = np.clip((raw - MEDIAN) / SCALE, -10.0, 10.0)
                window.append(scaled)
                if len(window) == SEQ_LEN:
                    sequence = np.asarray(window, dtype=np.float32)[None, ...]
                    reconstructed = model(sequence, training=False).numpy()
                    score_value = float(np.mean(np.square(sequence - reconstructed)))
                    is_anomaly = score_value > THRESHOLD
                    score = f"{score_value:.8f}"
                    anomaly = int(is_anomaly)
                    if is_anomaly != last_alert_state:
                        if is_anomaly:
                            print(f"ANOMALY: reconstruction_mse={score_value:.6f} > threshold={THRESHOLD:.6f}")
                        else:
                            print(f"Anomaly cleared: reconstruction_mse={score_value:.6f}")
                        last_alert_state = is_anomaly

            row = {"timestamp": time.time(), **state, "anomaly_score": score, "anomaly": anomaly}
            writer.writerow(row)

    except KeyboardInterrupt:
        print("\nTelemetry extraction and anomaly detection stopped.")
