"""Shared runtime pieces: the sliding-window fault detector and the fault-response supervisor.

FaultDetector   0.5 s sliding window (5 x 10 Hz rows) -> INT8 TFLite classifier -> fault probability; alarm when the
                probability exceeds the threshold for `persistence` consecutive windows. Model, feature set (v1 / v2),
                scaling, threshold and persistence all come from the model directory (preprocessing.npz,
                operating_point.json), so every consumer behaves identically.
FaultSupervisor two-stage response, driven by the detector at 10 Hz, acting through a pymavlink connection:
                  NORMAL  --alarm-->                         HOLD     stop horizontal motion, hold altitude (GUIDED)
                  HOLD    --no alarm for clear_s-->          NORMAL   (counted as a false alarm, control is released)
                  HOLD    --probability > threshold for confirm_s in total--> DESCEND  vertical velocity descent_speed
                  DESCEND --rel_alt < land_alt-->            LAND     ArduPilot LAND mode for the touchdown
                HOLD is harmless if the alarm was false. On entering DESCEND the supervisor can also release ArduPilot's
                yaw headroom (MOT_YAW_HEADROOM -> 0, restored on LAND/recovery when it was changed): the authority
                reserved for yaw goes to lift and roll/pitch, at the cost of yaw control, which a quadcopter with one
                weak rotor cannot keep anyway.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

from motor_failure_predictor.fault_features import (RAW_FIELDS, RAW_FIELDS_V2, SEQ_LEN, apply_scaler,
                                                     compute_features, compute_features_v2)


def make_interpreter(path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=str(path), num_threads=1)
    it.allocate_tensors()
    return it


class FaultDetector:
    def __init__(self, model_dir, threshold=None, persistence=None):
        d = Path(model_dir)
        pre = np.load(d / "preprocessing.npz")
        op = json.loads((d / "operating_point.json").read_text())
        self.feature_set = str(pre["feature_set"]) if "feature_set" in pre.files else "v1"
        self.raw_fields = RAW_FIELDS_V2 if self.feature_set == "v2" else RAW_FIELDS
        self.compute = compute_features_v2 if self.feature_set == "v2" else compute_features
        self.cols = pre["feature_index"]
        self.median, self.scale = pre["median"].astype(np.float32), pre["scale"].astype(np.float32)
        self.threshold = float(op["threshold"] if threshold is None else threshold)
        self.persistence = int(op["persistence"] if persistence is None else persistence)
        self.model_path = d / op["model"]
        self.it = make_interpreter(self.model_path)
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.buffer = deque(maxlen=SEQ_LEN)
        self.run = 0

    def _predict(self, window):
        x = window[None].astype(np.float32)
        if self.inp["dtype"] == np.int8:
            s, z = self.inp["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])
        if self.out["dtype"] == np.int8:
            s, z = self.out["quantization"]
            y = (y.astype(np.float32) - z) * s
        return float(y.reshape(-1)[0])

    def step(self, raw):
        """raw: dict with self.raw_fields keys, or a sequence in self.raw_fields order. Returns (prob or None, alarm)."""
        if isinstance(raw, dict):
            raw = [raw.get(k, float("nan")) for k in self.raw_fields]
        raw = np.asarray(raw, np.float32)
        if np.isnan(raw[:7]).any():  # motors / vibration not received yet
            return None, False
        self.buffer.append(apply_scaler(self.compute(raw)[self.cols], self.median, self.scale))
        if len(self.buffer) < SEQ_LEN:
            return None, False
        p = self._predict(np.stack(self.buffer))
        self.run = self.run + 1 if p > self.threshold else 0
        return p, self.run >= self.persistence


class FaultSupervisor:
    NORMAL, HOLD, DESCEND, LAND = "NORMAL", "HOLD", "DESCEND", "LAND"

    def __init__(self, conn, threshold, confirm_s=1.0, clear_s=3.0, descent_speed=2.0, land_alt=2.0,
                 rate_hz=10.0, log=print, release_yaw=False):
        self.c, self.threshold = conn, threshold
        self.confirm_n, self.clear_n = int(confirm_s * rate_hz), int(clear_s * rate_hz)
        self.descent_speed, self.land_alt, self.release_yaw = descent_speed, land_alt, release_yaw
        self.log = log
        self.state = self.NORMAL
        self.above = self.quiet = 0
        self.events = []

    @property
    def active(self):
        return self.state != self.NORMAL

    def _mode(self, name):
        self.c.set_mode(self.c.mode_mapping()[name])

    def _velocity(self, vz):
        from pymavlink import mavutil
        # LOCAL_NED velocity target, position / accel / yaw ignored (type mask 0b0000111111000111)
        self.c.mav.set_position_target_local_ned_send(
            0, self.c.target_system, self.c.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111, 0, 0, 0, 0, 0, vz, 0, 0, 0, 0, 0)

    def _goto(self, state, why):
        self.events.append((time.time(), self.state, state, why))
        self.log(f"supervisor: {self.state} -> {state} ({why})")
        self.state = state

    def update(self, prob, alarm, rel_alt):
        """Call once per detector step (10 Hz)."""
        hot = prob is not None and prob > self.threshold
        if self.state == self.NORMAL:
            if alarm:
                self._mode("GUIDED")
                self._velocity(0.0)
                self.above, self.quiet = 0, 0
                self._goto(self.HOLD, f"alarm p={prob:.2f}")
        elif self.state == self.HOLD:
            self._velocity(0.0)
            self.above += hot
            self.quiet = 0 if alarm or hot else self.quiet + 1
            if self.above >= self.confirm_n:
                if self.release_yaw:
                    self.c.mav.param_set_send(self.c.target_system, self.c.target_component, b"MOT_YAW_HEADROOM", 0.0,
                                              9)  # MAV_PARAM_TYPE_REAL32
                self._goto(self.DESCEND, f"confirmed: {self.above} hot windows" +
                           (", yaw headroom released" if self.release_yaw else ""))
            elif self.quiet >= self.clear_n:
                self._goto(self.NORMAL, "alarm cleared (false alarm)")
        elif self.state == self.DESCEND:
            self._velocity(self.descent_speed)  # NED: positive z velocity = down
            if rel_alt is not None and rel_alt < self.land_alt:
                self._mode("LAND")
                self._goto(self.LAND, f"rel_alt {rel_alt:.1f} m")
        return self.state
