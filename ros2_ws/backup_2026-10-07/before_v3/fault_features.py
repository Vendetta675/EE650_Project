"""Shared feature definition for motor-fault early detection (recording, training, live inference).

Raw telemetry (10 Hz, MAVLink):
  SERVO_OUTPUT_RAW  servo1..4_raw -> m1..m4 (us)        motor i <-> Gazebo rotor_{i-1}_joint
  VIBRATION         vibration_x/y/z -> vib_x/y/z (m/s^2)
  BATTERY_STATUS    current_battery (cA) -> current (A); SYS_STATUS current_battery as fallback

Model features:
  rel_m1..rel_m4   m_i - mean(other three motors). Wind and manoeuvres move all four PWMs together; a motor
                   losing effectiveness is driven above the others by the attitude controller, so its
                   relative PWM creeps up (leading indicator of LOE).
  vib_x/y/z        stutter / ESC desync shows up as vibration.
  current          electrical current (A); NaN-safe (0 if the vehicle does not report it).
  pwm_spread       max(m) - min(m): one number that grows whichever motor degrades.
The EKF variances of the previous pipeline were 0 in every logged row and are dropped.
"""
from __future__ import annotations

import numpy as np

RAW_FIELDS = ["m1", "m2", "m3", "m4", "vib_x", "vib_y", "vib_z", "current"]
FEATURES = ["rel_m1", "rel_m2", "rel_m3", "rel_m4", "vib_x", "vib_y", "vib_z", "current", "pwm_spread"]
SAMPLE_HZ = 10
SEQ_LEN = 5  # 0.5 s at 10 Hz


def compute_features(raw: np.ndarray) -> np.ndarray:
    """raw: (..., len(RAW_FIELDS)) in RAW_FIELDS order -> (..., len(FEATURES)) float32."""
    raw = np.asarray(raw, dtype=np.float32)
    m = raw[..., 0:4]
    total = m.sum(axis=-1, keepdims=True)
    rel = m - (total - m) / 3.0
    vib = raw[..., 4:7]
    current = np.nan_to_num(raw[..., 7:8], nan=0.0)
    spread = m.max(axis=-1, keepdims=True) - m.min(axis=-1, keepdims=True)
    return np.concatenate([rel, vib, current, spread], axis=-1).astype(np.float32)


def fit_scaler(features: np.ndarray):
    """Robust scaling (median / IQR), fitted on normal training rows only."""
    median = np.nanmedian(features, axis=0)
    q25, q75 = np.nanpercentile(features, [25, 75], axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return median.astype(np.float32), scale.astype(np.float32)


def apply_scaler(features: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((features - median) / scale, -10.0, 10.0).astype(np.float32)


def windows(a: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Sliding windows (n - seq_len + 1, seq_len, f); window k ends at row k + seq_len - 1."""
    if len(a) < seq_len:
        return np.zeros((0, seq_len, a.shape[-1]), np.float32)
    idx = np.arange(seq_len)[None, :] + np.arange(len(a) - seq_len + 1)[:, None]
    return a[idx]


# ------------------------------------------------------------------------------------------------ v2 (2026-10-08)
# More MAVLink signals. IMU and attitude rates are streamed at 50 Hz and summarised per 0.1 s step (mean and standard
# deviation), so the model still runs at 10 Hz but sees high-rate content: the stutter of a failing motor raises the
# within-step spread, and its torque imbalance shows up as a yaw/roll/pitch-rate bias.
#   RAW_IMU   xacc/yacc/zacc (mG) -> acc_* (m/s^2); xgyro/ygyro/zgyro (mrad/s) -> gyr_* (rad/s)
#   ATTITUDE  rollspeed/pitchspeed/yawspeed (rad/s) -> rate_*   (EKF-filtered body rates)
RAW_FIELDS_V2 = RAW_FIELDS + [
    "acc_x", "acc_y", "acc_z", "acc_sx", "acc_sy", "acc_sz",
    "gyr_x", "gyr_y", "gyr_z", "gyr_sx", "gyr_sy", "gyr_sz",
    "rate_x", "rate_y", "rate_z", "n_imu"]
# v2 features = the v1 features (first 9 columns, identical definition) + IMU features
FEATURES_V2 = FEATURES + [
    "acc_sx", "acc_sy", "acc_sz",        # within-step accelerometer spread (vibration / stutter)
    "gyr_sx", "gyr_sy", "gyr_sz",        # within-step gyro spread
    "gyr_x", "gyr_y", "gyr_z",           # mean body rates (torque imbalance -> rate bias, esp. yaw)
    "rate_z",                            # EKF yaw rate
    "acc_z_dev"]                         # vertical specific force minus gravity (thrust deficit)
G = 9.80665


def compute_features_v2(raw: np.ndarray) -> np.ndarray:
    """raw: (..., len(RAW_FIELDS_V2)) in RAW_FIELDS_V2 order -> (..., len(FEATURES_V2)) float32."""
    raw = np.asarray(raw, dtype=np.float32)
    f = {k: raw[..., i] for i, k in enumerate(RAW_FIELDS_V2)}
    v1 = compute_features(raw[..., :len(RAW_FIELDS)])
    extra = [f["acc_sx"], f["acc_sy"], f["acc_sz"], f["gyr_sx"], f["gyr_sy"], f["gyr_sz"],
             f["gyr_x"], f["gyr_y"], f["gyr_z"], f["rate_z"], -f["acc_z"] - G]
    extra = np.nan_to_num(np.stack(extra, axis=-1), nan=0.0)
    return np.concatenate([v1, extra], axis=-1).astype(np.float32)


class TelemetryAggregator:
    """Turns a stream of pymavlink messages into one RAW_FIELDS_V2 row per sample period.

    Latest value for motors / vibration / current / attitude rates; mean and std over the samples received since the
    previous row for RAW_IMU. Shared by the flight recorder, the MAVLink->ROS bridge and the live extractor, so
    training and inference see identical raw rows."""

    STREAMS = [("SERVO_OUTPUT_RAW", 20), ("VIBRATION", 20), ("BATTERY_STATUS", 10), ("SYS_STATUS", 10),
               ("RAW_IMU", 50), ("ATTITUDE", 50)]

    def __init__(self):
        self.state = {k: float("nan") for k in RAW_FIELDS_V2}
        self.imu = []

    @staticmethod
    def request_streams(conn, extra=()):
        from pymavlink import mavutil
        for name, hz in list(TelemetryAggregator.STREAMS) + list(extra):
            conn.mav.command_long_send(conn.target_system, conn.target_component,
                                       mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                       getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}"), int(1e6 / hz),
                                       0, 0, 0, 0, 0)

    def update(self, msg) -> None:
        t = msg.get_type()
        s = self.state
        if t == "SERVO_OUTPUT_RAW":
            for i in range(4):
                s[f"m{i + 1}"] = float(getattr(msg, f"servo{i + 1}_raw"))
        elif t == "VIBRATION":
            s["vib_x"], s["vib_y"], s["vib_z"] = msg.vibration_x, msg.vibration_y, msg.vibration_z
        elif t == "BATTERY_STATUS" and msg.current_battery >= 0:
            s["current"] = msg.current_battery / 100.0
        elif t == "SYS_STATUS" and msg.current_battery >= 0 and np.isnan(s["current"]):
            s["current"] = msg.current_battery / 100.0
        elif t == "RAW_IMU":
            self.imu.append((msg.xacc * G / 1000, msg.yacc * G / 1000, msg.zacc * G / 1000,
                             msg.xgyro / 1000, msg.ygyro / 1000, msg.zgyro / 1000))
        elif t == "ATTITUDE":
            s["rate_x"], s["rate_y"], s["rate_z"] = msg.rollspeed, msg.pitchspeed, msg.yawspeed

    def row(self) -> dict:
        """Close the current sample period and return its RAW_FIELDS_V2 values (IMU stats of this period)."""
        s = self.state
        if self.imu:
            a = np.asarray(self.imu, dtype=np.float64)
            mu, sd = a.mean(0), a.std(0)
            for j, k in enumerate(["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]):
                s[k] = float(mu[j])
            for j, k in enumerate(["acc_sx", "acc_sy", "acc_sz", "gyr_sx", "gyr_sy", "gyr_sz"]):
                s[k] = float(sd[j])
        s["n_imu"] = float(len(self.imu))
        self.imu = []
        return dict(s)

    def ready(self, fields=RAW_FIELDS) -> bool:
        return not any(np.isnan(self.state[k]) for k in fields if k != "current")
