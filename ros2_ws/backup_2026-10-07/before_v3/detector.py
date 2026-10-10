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
                With ftc=True the supervisor works with ArduPilot's fault-tolerant controller (CC_TYPE 3):
                  NORMAL --alarm--> ARMED   custom controller switched on with CC3_FAIL_MOT 0: it passes the normal
                                            controller through and watches the mixer at loop rate; a complete rotor
                                            loss switches it to three-rotor control within ~30 ms (onboard, CC3_AUTO)
                                            The mission continues (ftc_continue=True); with ftc_continue=False the
                                            normal HOLD -> DESCEND -> LAND response runs with the FTC armed.
                  any    --"FTC: motor N lost" STATUSTEXT--> FTC   (onboard switch confirmed; supervisor stops commanding)
                  any    --MotorLossIdentifier + detector agrees--> FTC   fallback for a loss without warning: the
                                            supervisor sets CC3_FAIL_MOT and switches the controller on itself. This
                                            path is ~0.3 s late (10 Hz telemetry), which the vehicle may not survive.
                From HOLD / DESCEND, FTC stops the descent (GUIDED then holds position).
MotorLossIdentifier  the failed motor of a quad X after a complete rotor loss: ArduPilot drives that motor to full
                output and its diagonal partner (same spin direction) to minimum within ~0.2 s. Normal flight in the
                training data never reaches 1900 us on any motor (largest deviation from the mean 86 us).
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


DIAGONAL = (1, 0, 3, 2)  # quad X (ArduPilot numbering 1-4): motors 1/2 and 3/4 are diagonal pairs


class MotorLossIdentifier:
    def __init__(self, pwm_high=1900.0, pwm_low=1250.0, rows=2):
        self.high, self.low, self.rows = pwm_high, pwm_low, rows
        self.k, self.n = -1, 0

    def step(self, pwm):
        """pwm: the 4 motor outputs (us) of one 10 Hz row. Returns the failed motor (1-4) or None."""
        m = np.asarray(pwm, np.float64)
        if m.shape != (4,) or np.isnan(m).any():
            self.n = 0
            return None
        k = int(np.argmax(m - m.mean()))
        if m[k] >= self.high and m[DIAGONAL[k]] <= self.low:
            self.n = self.n + 1 if k == self.k else 1
            self.k = k
        else:
            self.n = 0
        return self.k + 1 if self.n >= self.rows else None


class FaultSupervisor:
    NORMAL, HOLD, DESCEND, LAND, ARMED, FTC = "NORMAL", "HOLD", "DESCEND", "LAND", "ARMED", "FTC"

    def __init__(self, conn, threshold, confirm_s=1.0, clear_s=3.0, descent_speed=2.0, land_alt=2.0,
                 rate_hz=10.0, log=print, release_yaw=False, ftc=False, ftc_agree_s=0.5, ftc_require_detector=True,
                 ftc_continue=True):
        self.c, self.threshold = conn, threshold
        self.confirm_n, self.clear_n = int(confirm_s * rate_hz), int(clear_s * rate_hz)
        self.descent_speed, self.land_alt, self.release_yaw = descent_speed, land_alt, release_yaw
        self.ftc, self.ftc_agree_n, self.ftc_require_detector = ftc, max(1, int(ftc_agree_s * rate_hz)), ftc_require_detector
        self.ftc_continue, self.ftc_armed, self.onboard_loss = ftc_continue, False, None
        self.log = log
        self.state = self.NORMAL
        self.above = self.quiet = 0
        self.hot_recent = 0  # detector windows left in which a probability above the threshold still counts
        self.ftc_motor, self.ftc_sends = None, 0
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

    def _controller_on(self):
        from pymavlink import mavutil
        self.c.mav.command_long_send(self.c.target_system, self.c.target_component,
                                     mavutil.mavlink.MAV_CMD_DO_AUX_FUNCTION, 0, 109, 2, 0, 0, 0, 0, 0)

    def _send_ftc(self):
        from pymavlink import mavutil
        self.c.mav.param_set_send(self.c.target_system, self.c.target_component, b"CC3_FAIL_MOT", float(self.ftc_motor),
                                  mavutil.mavlink.MAV_PARAM_TYPE_INT8)
        if not self.ftc_armed or self.ftc_sends == 0:
            self._controller_on()  # switching on again resets the controller's observer: only when needed
        self.ftc_sends += 1

    def _poll_link(self):
        """Drain the link; the onboard FTC announces a detected loss with 'FTC: motor N lost'."""
        for _ in range(200):
            m = self.c.recv_match(blocking=False)
            if m is None:
                break
            if m.get_type() == "STATUSTEXT" and "FTC: motor" in m.text and "lost" in m.text and self.onboard_loss is None:
                try:
                    self.onboard_loss = int(m.text.split("motor")[1].split()[0])
                except (IndexError, ValueError):
                    self.onboard_loss = 0

    def update(self, prob, alarm, rel_alt, failed_motor=None):
        """Call once per detector step (10 Hz). failed_motor: MotorLossIdentifier output (1-4 or None)."""
        hot = prob is not None and prob > self.threshold
        self.hot_recent = self.ftc_agree_n if hot else max(0, self.hot_recent - 1)
        if self.ftc:
            self._poll_link()
        if self.state == self.FTC:
            if self.ftc_motor is not None and self.ftc_sends < 3:  # fallback hand-off: repeat, the link may drop it
                self._send_ftc()
            return self.state
        if self.ftc and self.onboard_loss is not None:
            if self.state in (self.HOLD, self.DESCEND):
                self._velocity(0.0)
            self._goto(self.FTC, f"onboard FTC took over motor {self.onboard_loss}")
            return self.state
        if (self.ftc and failed_motor is not None and self.state != self.LAND
                and (self.hot_recent > 0 or not self.ftc_require_detector)):
            was = self.state
            self.ftc_motor, self.ftc_sends = int(failed_motor), 0
            self._send_ftc()
            if was in (self.HOLD, self.DESCEND):
                self._velocity(0.0)  # stop the descent; GUIDED then holds position
            self._goto(self.FTC, f"motor {self.ftc_motor} lost (outputs), p={prob if prob is not None else float('nan'):.2f}:"
                                 " fault-tolerant controller on")
            return self.state
        if self.state == self.ARMED:
            return self.state  # stays armed: harmless if the alarm was false (pure pass-through)
        if self.state == self.NORMAL and alarm and self.ftc and not self.ftc_armed:
            self._controller_on()
            self.ftc_armed = True
            if self.ftc_continue:
                self._goto(self.ARMED, f"alarm p={prob:.2f}: fault-tolerant controller armed, mission continues")
                return self.state
            self.log("supervisor: fault-tolerant controller armed")
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
