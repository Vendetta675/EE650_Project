"""Automated data collection: SITL + Gazebo flight with progressive motor-failure episodes.

Prerequisites (started by collect_data.sh): gz sim server with iris_runway.sdf and the progressive
MotorFailureSystem plugin, and ArduCopter SITL (sim_vehicle.py, gazebo-iris, JSON) without MAVProxy, so
that tcp:5760 (this script, control) and tcp:5762 (recorder) are free.

One flight = takeoff to ALT m, then EPISODES x [normal phase with random GUIDED moves (NORMAL_S, random
length), failure trigger (plugin: LOE ramp + stutter), FAULT_HOLD_S of degraded flight, recover,
STABILISE_S], then land. Every row is recorded at 10 Hz by record_telemetry() in a separate thread; ground
truth comes from the plugin's event log (fault_events.csv) and is joined later by label_precursors.py.
Usage: python3 fly_fault_episodes.py --episodes 8 --out flights/flight_01.csv --seed 1
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
import threading
import time

from pymavlink import mavutil

from fault_features import RAW_FIELDS_V3, TelemetryAggregator

MODEL = "iris_with_gimbal"
TRIGGER = f"/model/{MODEL}/trigger_random_failure"


def trigger(on: bool, event_log: str | None = None, retries: int = 8):
    """Publish the failure trigger. `gz topic -p` is a one-shot publisher that can exit before transport discovery
    reaches the plugin (seen under CPU load: the message is silently dropped). With event_log given, re-publish until
    the plugin logs the matching event ("start" for on, "recover" for off) or retries run out; returns True if
    confirmed. Without event_log: publish once (previous behaviour)."""
    def count(ev):
        try:
            with open(event_log) as f:
                return sum(1 for line in f if f",{ev}," in line)
        except FileNotFoundError:
            return 0
    ev = "start" if on else "recover"
    before = count(ev) if event_log else 0
    for _ in range(retries if event_log else 1):
        subprocess.run(["gz", "topic", "-t", TRIGGER, "-m", "gz.msgs.Boolean", "-p",
                        f"data: {'true' if on else 'false'}"], check=True, capture_output=True, timeout=10)
        if not event_log:
            return True
        t0 = time.time()
        while time.time() - t0 < 1.5:
            if count(ev) > before:
                return True
            time.sleep(0.1)
    return False


def request_rates(conn, rates):
    for msg_id, hz in rates:
        conn.mav.command_long_send(conn.target_system, conn.target_component,
                                   mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, msg_id, int(1e6 / hz),
                                   0, 0, 0, 0, 0)


def record_telemetry(url: str, out_path: str, stop: threading.Event, hz: float = 10.0):
    """10 Hz rows of fault_features.RAW_FIELDS_V3 (via the shared TelemetryAggregator: motors, vibration, current,
    IMU mean/std per step, attitude rates) + rel_alt, roll, pitch, armed for labelling."""
    for attempt in range(60):  # SITL opens SERIAL1 only after SERIAL0 (control) has connected
        try:
            conn = mavutil.mavlink_connection(url, source_system=254)
            if conn.wait_heartbeat(timeout=10) is not None:
                break
        except OSError:
            pass
        time.sleep(2)
    else:
        raise RuntimeError(f"recorder could not connect to {url}")
    M = mavutil.mavlink
    agg = TelemetryAggregator()
    agg.request_streams(conn, extra=[("GLOBAL_POSITION_INT", 10)])
    extra = dict(rel_alt=float("nan"), roll=float("nan"), pitch=float("nan"), armed=0)
    fields = ["timestamp"] + RAW_FIELDS_V3 + list(extra)
    with open(out_path, "w", newline="", buffering=1) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        next_t = time.time()
        while not stop.is_set():
            msg = conn.recv_match(blocking=True, timeout=0.02)
            if msg is not None:
                agg.update(msg)
                t = msg.get_type()
                if t == "GLOBAL_POSITION_INT":
                    extra["rel_alt"] = msg.relative_alt / 1000.0
                elif t == "ATTITUDE":
                    extra["roll"], extra["pitch"] = math.degrees(msg.roll), math.degrees(msg.pitch)
                elif t == "HEARTBEAT" and msg.get_srcComponent() == 1:
                    extra["armed"] = int(bool(msg.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED))
            now = time.time()
            if now >= next_t:
                next_t += 1.0 / hz
                if now - next_t > 1.0:  # fell behind (e.g. blocked): resync instead of bursting
                    next_t = now + 1.0 / hz
                w.writerow({"timestamp": now, **agg.row(), **extra})


class Copter:
    def __init__(self, url):
        self.c = mavutil.mavlink_connection(url, source_system=255)
        self.c.wait_heartbeat(timeout=120)
        M = mavutil.mavlink  # no MAVProxy: this link must request its own streams
        request_rates(self.c, [(M.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 2), (M.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 10),
                               (M.MAVLINK_MSG_ID_HEARTBEAT, 2)])

    def mode(self, name):
        self.c.set_mode(self.c.mode_mapping()[name])
        self._wait(lambda: self.c.flightmode == name, 15, f"mode {name}")

    def _wait(self, cond, timeout, what):
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.c.recv_match(blocking=True, timeout=0.5)
            if cond():
                return
        raise TimeoutError(what)

    def wait_ready(self, timeout=180):
        """Wait until the EKF reports a usable position (pre-arm checks can still fail; arm retries)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = self.c.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=1)
            if m is not None and (m.flags & mavutil.mavlink.EKF_POS_HORIZ_ABS) and (m.flags & mavutil.mavlink.EKF_PRED_POS_HORIZ_ABS):
                return
        raise TimeoutError("EKF not ready")

    def arm_takeoff(self, alt):
        self.mode("GUIDED")
        for _ in range(20):
            self.c.arducopter_arm()
            try:
                self._wait(lambda: self.c.motors_armed(), 5, "arm")
                break
            except TimeoutError:
                time.sleep(2)
        else:
            raise RuntimeError("could not arm")
        self.c.mav.command_long_send(self.c.target_system, self.c.target_component,
                                     mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt)
        self.wait_alt(alt * 0.95, 60)

    def rel_alt(self):
        m = self.c.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
        return None if m is None else m.relative_alt / 1000.0

    def wait_alt(self, alt, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            a = self.rel_alt()
            if a is not None and a >= alt:
                return
        raise TimeoutError("altitude")

    def goto_offset(self, north, east, alt):
        """GUIDED position target relative to home (LOCAL_NED), type mask = position only."""
        self.c.mav.set_position_target_local_ned_send(
            0, self.c.target_system, self.c.target_component, mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000, north, east, -alt, 0, 0, 0, 0, 0, 0, 0, 0)

    def land(self):
        self.mode("LAND")
        t0 = time.time()
        while time.time() - t0 < 90 and self.c.motors_armed():
            self.c.recv_match(blocking=True, timeout=0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", default="tcp:127.0.0.1:5760")
    ap.add_argument("--record", default="tcp:127.0.0.1:5762")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--alt", type=float, default=10.0)
    ap.add_argument("--normal-min", type=float, default=15.0)
    ap.add_argument("--normal-max", type=float, default=30.0)
    ap.add_argument("--fault-hold", type=float, default=12.0, help="seconds after the trigger before recovery")
    ap.add_argument("--stabilise", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event-log", default=None,
                    help="plugin event log: re-publish the trigger until the plugin confirms it (lost triggers in v2)")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    cop = Copter(a.control)  # first: SITL waits for a client on SERIAL0 before it starts
    stop = threading.Event()
    rec = threading.Thread(target=record_telemetry, args=(a.record, a.out, stop), daemon=True)
    rec.start()
    print("connected; waiting for EKF", flush=True)
    cop.wait_ready()
    trigger(False)  # clean state
    cop.arm_takeoff(a.alt)
    print(f"airborne at {a.alt} m", flush=True)
    try:
        for ep in range(a.episodes):
            # normal phase: a few random position targets within 15 m, varying altitude a little
            t_end = time.time() + rng.uniform(a.normal_min, a.normal_max)
            while time.time() < t_end:
                cop.goto_offset(rng.uniform(-15, 15), rng.uniform(-15, 15), a.alt + rng.uniform(-2, 2))
                t_leg = min(t_end, time.time() + rng.uniform(4, 9))
                while time.time() < t_leg:
                    cop.c.recv_match(blocking=True, timeout=0.5)
            # hover still at the trigger in half of the episodes, keep manoeuvring in the other half
            if ep % 2 == 0:
                cop.goto_offset(rng.uniform(-5, 5), rng.uniform(-5, 5), a.alt)
            print(f"episode {ep}: trigger at {time.time():.2f}", flush=True)
            if not trigger(True, a.event_log):
                print(f"episode {ep}: WARNING trigger not confirmed by the plugin", flush=True)
            t_end = time.time() + a.fault_hold
            while time.time() < t_end:
                cop.c.recv_match(blocking=True, timeout=0.5)
            alt = cop.rel_alt()
            trigger(False, a.event_log)
            print(f"episode {ep}: recovered, alt {alt}", flush=True)
            if alt is not None and alt < 2.0:
                print("vehicle lost altitude, ending flight", flush=True)
                break
            t_end = time.time() + a.stabilise
            cop.goto_offset(0, 0, a.alt)
            while time.time() < t_end:
                cop.c.recv_match(blocking=True, timeout=0.5)
        cop.land()
    finally:
        trigger(False)
        time.sleep(1)
        stop.set()
        rec.join(timeout=5)
    print("flight done", flush=True)


if __name__ == "__main__":
    main()
