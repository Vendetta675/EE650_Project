"""Closed-loop evaluation of the fault-response supervisor in SITL + Gazebo (one fault episode per flight).

Same prerequisites as fly_fault_episodes.py (gz server + arducopter started by collect_data.sh-style launcher).
Connections: tcp:5760 flight script (takeoff, random GUIDED moves), tcp:5762 telemetry -> TelemetryAggregator ->
FaultDetector at 10 Hz, tcp:5763 FaultSupervisor commands. While the supervisor is active (HOLD / DESCEND / LAND)
the flight script sends no setpoints. With --supervisor off the detector still runs and is logged, but nothing acts.

Flight: takeoff to ALT, normal phase (random moves, NORMAL_MIN..NORMAL_MAX s; holds triggered here are false
alarms), trigger the progressive failure, wait until landed / disarmed or FAULT_HOLD s, recover, land if airborne.
Rows (10 Hz): timestamp, RAW_FIELDS_V2, rel_alt, roll, pitch, armed, prob, alarm, sup_state.
Usage: python3 eval_supervisor.py --model-dir models_gru_v2 --supervisor on --out supervisor_eval/flight_01.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import threading
import time
from pathlib import Path

from pymavlink import mavutil

sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "motor_failure_predictor"))
from motor_failure_predictor.detector import FaultDetector, FaultSupervisor  # noqa: E402
from motor_failure_predictor.fault_features import RAW_FIELDS_V2, TelemetryAggregator  # noqa: E402
from fly_fault_episodes import Copter, trigger  # noqa: E402


def connect(url, sysid):
    for _ in range(60):
        try:
            c = mavutil.mavlink_connection(url, source_system=sysid)
            if c.wait_heartbeat(timeout=10) is not None:
                return c
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError(f"no heartbeat on {url}")


class Loop(threading.Thread):
    """Telemetry -> detector -> (supervisor) at 10 Hz, with logging."""

    def __init__(self, a, stop):
        super().__init__(daemon=True)
        self.a, self.stop = a, stop
        self.det = FaultDetector(a.model_dir)
        self.sup = None
        self.extra = dict(rel_alt=float("nan"), roll=float("nan"), pitch=float("nan"), armed=0)

    def run(self):
        tel = connect(self.a.record, 254)
        agg = TelemetryAggregator()
        agg.request_streams(tel, extra=[("GLOBAL_POSITION_INT", 10)])
        if self.a.supervisor == "on":
            self.sup = FaultSupervisor(connect(self.a.command, 252), self.det.threshold,
                                       descent_speed=self.a.descent_speed, release_yaw=self.a.release_yaw,
                                       log=lambda s: print(f"{time.time():.2f} {s}", flush=True))
        fields = ["timestamp"] + RAW_FIELDS_V2 + list(self.extra) + ["prob", "alarm", "sup_state"]
        M = mavutil.mavlink
        with open(self.a.out, "w", newline="", buffering=1) as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            next_t = time.time()
            while not self.stop.is_set():
                msg = tel.recv_match(blocking=True, timeout=0.02)
                if msg is not None:
                    agg.update(msg)
                    t = msg.get_type()
                    if t == "GLOBAL_POSITION_INT":
                        self.extra["rel_alt"] = msg.relative_alt / 1000.0
                    elif t == "ATTITUDE":
                        self.extra["roll"], self.extra["pitch"] = math.degrees(msg.roll), math.degrees(msg.pitch)
                    elif t == "HEARTBEAT" and msg.get_srcComponent() == 1:
                        self.extra["armed"] = int(bool(msg.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED))
                now = time.time()
                if now < next_t:
                    continue
                next_t += 0.1
                if now - next_t > 1.0:
                    next_t = now + 0.1
                row = agg.row()
                prob, alarm = self.det.step(row)
                state = "OFF"
                if self.sup is not None:
                    airborne = self.extra["armed"] and self.extra["rel_alt"] > 1.0
                    state = self.sup.update(prob, alarm and airborne, self.extra["rel_alt"]) if airborne or \
                        self.sup.active else self.sup.state
                w.writerow({"timestamp": now, **row, **self.extra, "prob": "" if prob is None else round(prob, 4),
                            "alarm": int(alarm), "sup_state": state})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", default="tcp:127.0.0.1:5760")
    ap.add_argument("--record", default="tcp:127.0.0.1:5762")
    ap.add_argument("--command", default="tcp:127.0.0.1:5763")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--supervisor", choices=["on", "off"], default="on")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alt", type=float, default=10.0)
    ap.add_argument("--normal-min", type=float, default=60.0)
    ap.add_argument("--normal-max", type=float, default=90.0)
    ap.add_argument("--fault-hold", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--descent-speed", type=float, default=2.0)
    ap.add_argument("--release-yaw", action="store_true")
    ap.add_argument("--event-log", default="/root/ros2_ws/fault_events.csv", help="plugin event log (trigger confirmation)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cop = Copter(a.control)
    stop = threading.Event()
    loop = Loop(a, stop)
    loop.start()
    cop.wait_ready()
    trigger(False)
    cop.arm_takeoff(a.alt)
    print(f"airborne; supervisor {a.supervisor}", flush=True)

    def wait(seconds):
        t_end = time.time() + seconds
        while time.time() < t_end:
            cop.c.recv_match(blocking=True, timeout=0.5)

    try:
        t_end = time.time() + rng.uniform(a.normal_min, a.normal_max)
        while time.time() < t_end:
            if loop.sup is not None and loop.sup.active:  # false-alarm hold: let the supervisor fly
                wait(0.5)
                continue
            if loop.sup is not None and loop.sup.events and loop.sup.state == "NORMAL":
                cop.mode("GUIDED")  # control released after a false alarm
            cop.goto_offset(rng.uniform(-15, 15), rng.uniform(-15, 15), a.alt + rng.uniform(-2, 2))
            leg_end = min(t_end, time.time() + rng.uniform(4, 9))
            while time.time() < leg_end and not (loop.sup is not None and loop.sup.active):
                cop.c.recv_match(blocking=True, timeout=0.5)
        if loop.sup is not None and loop.sup.active:  # do not start the fault inside a false-alarm hold
            t0 = time.time()
            while loop.sup.active and time.time() - t0 < 10:
                wait(0.5)
            if loop.sup.state == "NORMAL":
                cop.mode("GUIDED")
        print(f"trigger at {time.time():.2f}", flush=True)
        if not trigger(True, a.event_log):
            print("TRIGGER NOT CONFIRMED by the plugin event log", flush=True)
        t_end = time.time() + a.fault_hold
        while time.time() < t_end:
            cop.c.recv_match(blocking=True, timeout=0.5)
            if not cop.c.motors_armed():
                print("disarmed (landed or crashed)", flush=True)
                break
        trigger(False, a.event_log)
        alt = cop.rel_alt()
        print(f"recovered, alt {alt}", flush=True)
        if cop.c.motors_armed() and alt is not None and alt > 1.0:
            cop.land()
    finally:
        trigger(False)
        time.sleep(2)
        stop.set()
        loop.join(timeout=5)
    sup_events = [] if loop.sup is None else [(round(t, 2), s0, s1, why) for t, s0, s1, why in loop.sup.events]
    print("supervisor events:", sup_events, flush=True)


if __name__ == "__main__":
    main()
