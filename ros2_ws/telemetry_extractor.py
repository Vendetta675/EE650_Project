"""Live MAVLink telemetry -> early motor-fault detection (0.5 s sliding window, INT8 TFLite GRU), without ROS.

Same pipeline as the ROS nodes (mavlink_bridge -> lstm_predictor): the shared TelemetryAggregator builds one raw row per
0.1 s (motors, vibration, current, IMU mean/std, attitude rates) and the shared FaultDetector scores it with the model,
feature set, scaling, threshold and persistence of MODEL_DIR. Rows (raw fields + fault_probability + alarm) are logged to
FLIGHT_DATA_OUTPUT.
Environment: MAVLINK_CONNECTION (default udp:127.0.0.1:14550), FLIGHT_DATA_OUTPUT, MODEL_DIR (default models_gru).
"""
import csv
import os
import sys
import time

from pymavlink import mavutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "motor_failure_predictor"))
from motor_failure_predictor.detector import FaultDetector  # noqa: E402
from motor_failure_predictor.fault_features import RAW_FIELDS_V2, SAMPLE_HZ, TelemetryAggregator  # noqa: E402

WORKSPACE = "/root/ros2_ws"
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(WORKSPACE, "models_gru"))
CSV_PATH = os.environ.get("FLIGHT_DATA_OUTPUT", os.path.join(WORKSPACE, "flight_data_live.csv"))
CONNECTION_STRING = os.environ.get("MAVLINK_CONNECTION", "udp:127.0.0.1:14550")

det = FaultDetector(MODEL_DIR)
connection = mavutil.mavlink_connection(CONNECTION_STRING)
connection.wait_heartbeat()
print("Connected to Flight Controller.")
agg = TelemetryAggregator()
agg.request_streams(connection)
fields = ["timestamp"] + RAW_FIELDS_V2 + ["fault_probability", "alarm"]

with open(CSV_PATH, mode="w", newline="", buffering=1) as output:
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    print(f"Model {det.model_path} ({det.feature_set} features), window 0.5 s, threshold {det.threshold:.3f}, "
          f"persistence {det.persistence}. Logging to {CSV_PATH}.")
    next_t, last_alarm = time.time(), None
    try:
        while True:
            msg = connection.recv_match(blocking=True, timeout=0.02)
            if msg is not None:
                agg.update(msg)
            now = time.time()
            if now < next_t:
                continue
            next_t += 1.0 / SAMPLE_HZ
            if now - next_t > 1.0:
                next_t = now + 1.0 / SAMPLE_HZ
            row = agg.row()
            prob, alarm = det.step(row)
            if prob is not None and alarm != last_alarm:
                print(("FAULT DEVELOPING" if alarm else "Fault cleared") + f": p={prob:.3f}")
                last_alarm = alarm
            writer.writerow({"timestamp": now, **row, "fault_probability": "" if prob is None else f"{prob:.5f}",
                             "alarm": "" if prob is None else int(alarm)})
    except KeyboardInterrupt:
        print("\nTelemetry extraction and fault detection stopped.")
