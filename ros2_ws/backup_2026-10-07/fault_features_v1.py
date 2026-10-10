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
