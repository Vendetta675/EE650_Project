"""Live MAVLink telemetry -> early motor-fault detection with the INT8 TFLite GRU classifier (0.5 s window).

Reads SERVO_OUTPUT_RAW (m1-m4), VIBRATION and BATTERY_STATUS (current; SYS_STATUS as fallback), samples at 10 Hz,
computes the shared features (fault_features.py: PWM of each motor relative to the mean of the other three,
vibration, current, PWM spread), scales them with the training statistics and scores every 0.5 s window.
Alarm = probability above the trained threshold for PERSISTENCE consecutive windows.
Rows (raw fields + fault_probability + alarm) are logged to FLIGHT_DATA_OUTPUT.
Environment: MAVLINK_CONNECTION (default udp:127.0.0.1:14550), FLIGHT_DATA_OUTPUT, MODEL_DIR, MODEL_FILE.
"""
import csv
import json
import os
import time
from collections import deque

import numpy as np
from pymavlink import mavutil

from fault_features import RAW_FIELDS, SAMPLE_HZ, SEQ_LEN, apply_scaler, compute_features

WORKSPACE = "/root/ros2_ws"
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(WORKSPACE, "models_gru"))
MODEL_PATH = os.path.join(MODEL_DIR, os.environ.get("MODEL_FILE", "cls_gru5_int8.tflite"))
CSV_PATH = os.environ.get("FLIGHT_DATA_OUTPUT", os.path.join(WORKSPACE, "flight_data_live.csv"))
CONNECTION_STRING = os.environ.get("MAVLINK_CONNECTION", "udp:127.0.0.1:14550")

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

pre = np.load(os.path.join(MODEL_DIR, "preprocessing.npz"))
COLS, MEDIAN, SCALE = pre["feature_index"], pre["median"].astype(np.float32), pre["scale"].astype(np.float32)
OP = json.load(open(os.path.join(MODEL_DIR, "operating_point.json")))  # threshold + persistence picked on validation
THRESHOLD, PERSISTENCE = float(OP["threshold"]), int(OP["persistence"])
interp = Interpreter(model_path=MODEL_PATH, num_threads=1)
interp.allocate_tensors()
INP, OUT = interp.get_input_details()[0], interp.get_output_details()[0]


def predict(window):
    x = window[None].astype(np.float32)
    if INP["dtype"] == np.int8:
        s, z = INP["quantization"]
        x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
    interp.set_tensor(INP["index"], x)
    interp.invoke()
    y = interp.get_tensor(OUT["index"])
    if OUT["dtype"] == np.int8:
        s, z = OUT["quantization"]
        y = (y.astype(np.float32) - z) * s
    return float(y.reshape(-1)[0])


connection = mavutil.mavlink_connection(CONNECTION_STRING)
connection.wait_heartbeat()
print("Connected to Flight Controller.")


def request_message_interval(message_id, frequency_hz):
    connection.mav.command_long_send(connection.target_system, connection.target_component,
                                     mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                     message_id, int(1e6 / frequency_hz), 0, 0, 0, 0, 0)


for mid in (mavutil.mavlink.MAVLINK_MSG_ID_VIBRATION, mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW,
            mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS, mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS):
    request_message_interval(mid, 2 * SAMPLE_HZ)

state = {k: float("nan") for k in RAW_FIELDS}
window = deque(maxlen=SEQ_LEN)
fields = ["timestamp"] + RAW_FIELDS + ["fault_probability", "alarm"]

with open(CSV_PATH, mode="w", newline="", buffering=1) as output:
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    print(f"Model {MODEL_PATH} ({INP['dtype'].__name__}), window {SEQ_LEN} x {1 / SAMPLE_HZ:.1f} s, "
          f"threshold {THRESHOLD:.3f}, persistence {PERSISTENCE}. Logging to {CSV_PATH}.")
    last_write, run, last_alarm = time.monotonic(), 0, None
    try:
        while True:
            msg = connection.recv_match(type=["VIBRATION", "SERVO_OUTPUT_RAW", "BATTERY_STATUS", "SYS_STATUS"],
                                        blocking=True, timeout=1)
            if msg is not None:
                t = msg.get_type()
                if t == "VIBRATION":
                    state["vib_x"], state["vib_y"], state["vib_z"] = msg.vibration_x, msg.vibration_y, msg.vibration_z
                elif t == "SERVO_OUTPUT_RAW":
                    for i in range(4):
                        state[f"m{i + 1}"] = float(getattr(msg, f"servo{i + 1}_raw"))
                elif t == "BATTERY_STATUS" and msg.current_battery >= 0:
                    state["current"] = msg.current_battery / 100.0
                elif t == "SYS_STATUS" and msg.current_battery >= 0 and np.isnan(state["current"]):
                    state["current"] = msg.current_battery / 100.0
            now = time.monotonic()
            if now - last_write < 1.0 / SAMPLE_HZ:
                continue
            last_write = now
            prob, alarm = "", ""
            if not any(np.isnan(state[k]) for k in RAW_FIELDS if k != "current"):
                raw = np.array([state[k] for k in RAW_FIELDS], np.float32)
                window.append(apply_scaler(compute_features(raw)[COLS], MEDIAN, SCALE))
                if len(window) == SEQ_LEN:
                    p = predict(np.stack(window))
                    run = run + 1 if p > THRESHOLD else 0
                    is_alarm = run >= PERSISTENCE
                    prob, alarm = f"{p:.5f}", int(is_alarm)
                    if is_alarm != last_alarm:
                        print(("FAULT DEVELOPING" if is_alarm else "Fault cleared") + f": p={p:.3f}")
                        last_alarm = is_alarm
            writer.writerow({"timestamp": time.time(), **state, "fault_probability": prob, "alarm": alarm})
    except KeyboardInterrupt:
        print("\nTelemetry extraction and fault detection stopped.")
