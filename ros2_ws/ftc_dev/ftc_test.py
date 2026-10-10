"""FTC test flight (SITL + Gazebo): takeoff, complete failure of one rotor, FTC switched on, then hold / waypoints.

Activation here is an oracle (the failed motor is known): CC3_FAIL_MOT is set and the custom controller is switched on
with MAV_CMD_DO_AUX_FUNCTION(109, HIGH) right after the plugin confirms the failure (--activate-delay s later).
Rows at 10 Hz: time, local NED position, velocity, attitude, yaw rate, motors, target waypoint, phase.
Plan mission: AUTO mission (20 m square at --alt, then LAND); the failure is triggered --fail-delay s after the vehicle
starts leg --fail-leg, so the rotor is lost while flying, and the mission must be finished on three rotors.
Usage: python3 ftc_test.py --out run.csv --motor 1 --plan hold|square|land|mission [--no-ftc]
"""
import argparse, csv, math, sys, threading, time
from pymavlink import mavutil
sys.path.insert(0, "/root/ros2_ws")
from fly_fault_episodes import Copter, trigger

def request(conn, msgs):
    for name, hz in msgs:
        conn.mav.command_long_send(conn.target_system, conn.target_component, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                   getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}"), int(1e6 / hz), 0, 0, 0, 0, 0)

def set_param(c, name, value, tries=10):
    """PARAM_SET and confirm with the PARAM_VALUE echo (a set sent too early in boot is silently lost)."""
    for _ in range(tries):
        c.mav.param_set_send(c.target_system, c.target_component, name.encode(), float(value), 9)
        t0 = time.time()
        while time.time() - t0 < 2:
            m = c.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if m is not None and m.param_id == name and abs(m.param_value - value) < 1e-4:
                print(f"param {name} = {m.param_value}", flush=True)
                return
    raise RuntimeError(f"could not set {name}")


SQUARE = [(20, 0), (20, 20), (0, 20), (0, 0)]  # north, east offsets from home (m)


def upload_mission(c, home_lat, home_lon, alt, legs=SQUARE):
    """Mission: seq 0 home (ignored by ArduPilot), seq 1..n waypoints of `legs`, then NAV_LAND at the last one."""
    M = mavutil.mavlink
    def ll(n, e):
        return (int((home_lat + n / 6378137.0 * 180 / math.pi) * 1e7),
                int((home_lon + e / (6378137.0 * math.cos(math.radians(home_lat))) * 180 / math.pi) * 1e7))
    items = [(M.MAV_CMD_NAV_WAYPOINT, *ll(0, 0), 0.0)]
    items += [(M.MAV_CMD_NAV_WAYPOINT, *ll(n, e), alt) for n, e in legs]
    items += [(M.MAV_CMD_NAV_LAND, *ll(*legs[-1]), 0.0)]
    c.mav.mission_count_send(c.target_system, c.target_component, len(items), M.MAV_MISSION_TYPE_MISSION)
    t0 = time.time()
    while time.time() - t0 < 20:
        m = c.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"], blocking=True, timeout=2)
        if m is None:
            continue
        if m.get_type() == "MISSION_ACK":
            if m.type != M.MAV_MISSION_ACCEPTED:
                raise RuntimeError(f"mission rejected: {m.type}")
            print(f"mission uploaded: {len(items) - 2} waypoints + LAND", flush=True)
            return
        cmd, lat, lon, a = items[m.seq]
        c.mav.mission_item_int_send(c.target_system, c.target_component, m.seq, M.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, cmd,
                                    0, 1, 0, 0, 0, 0, lat, lon, a, M.MAV_MISSION_TYPE_MISSION)
    raise TimeoutError("mission upload")


def recorder(url, out, stop, shared):
    c = mavutil.mavlink_connection(url, source_system=254)
    c.wait_heartbeat(timeout=60)
    request(c, [("LOCAL_POSITION_NED", 20), ("ATTITUDE", 20), ("SERVO_OUTPUT_RAW", 20), ("GLOBAL_POSITION_INT", 10)])
    st = dict(x=float("nan"), y=float("nan"), z=float("nan"), vx=0, vy=0, vz=0, roll=0, pitch=0, yaw=0, yawspeed=0,
              m1=0, m2=0, m3=0, m4=0, armed=0)
    with open(out, "w", newline="", buffering=1) as f:
        w = csv.DictWriter(f, fieldnames=["t"] + list(st) + ["tx", "ty", "tz", "phase", "statustext"])
        w.writeheader()
        nxt = time.time(); txt = ""
        while not stop.is_set():
            m = c.recv_match(blocking=True, timeout=0.02)
            if m is not None:
                t = m.get_type()
                if t == "LOCAL_POSITION_NED":
                    st.update(x=m.x, y=m.y, z=m.z, vx=m.vx, vy=m.vy, vz=m.vz)
                elif t == "ATTITUDE":
                    st.update(roll=math.degrees(m.roll), pitch=math.degrees(m.pitch), yaw=math.degrees(m.yaw), yawspeed=m.yawspeed)
                elif t == "SERVO_OUTPUT_RAW":
                    st.update(m1=m.servo1_raw, m2=m.servo2_raw, m3=m.servo3_raw, m4=m.servo4_raw)
                elif t == "HEARTBEAT" and m.get_srcComponent() == 1:
                    st["armed"] = int(bool(m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))
                    shared["armed"] = st["armed"]
                elif t == "STATUSTEXT":
                    txt = m.text; print("FC:", m.text, flush=True)
            if time.time() >= nxt:
                nxt += 0.1
                tx, ty, tz = shared["target"]
                w.writerow({"t": time.time(), **st, "tx": tx, "ty": ty, "tz": tz, "phase": shared["phase"], "statustext": txt})
                txt = ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--motor", type=int, default=1, help="failed motor 1-4 (plugin failure_motor = motor-1)")
    ap.add_argument("--plan", choices=["hold", "square", "land", "mission"], default="hold")
    ap.add_argument("--fail-leg", type=int, default=2, help="mission: leg (waypoint number) during which the failure starts")
    ap.add_argument("--fail-delay", type=float, default=3.0, help="mission: s after the start of that leg")
    ap.add_argument("--no-ftc", action="store_true")
    ap.add_argument("--activate-delay", type=float, default=0.0)
    ap.add_argument("--arm-before", type=float, default=-1.0,
                    help="arm the FTC this many s before the failure (CC on, CC3_FAIL_MOT 0): onboard detection")
    ap.add_argument("--event-log", default="/root/ros2_ws/ftc_dev/events.csv")
    ap.add_argument("--passive", action="store_true",
                    help="after the failure trigger send no more position targets (GUIDED holds the last one): leaves "
                         "the vehicle to the fault supervisor, whose descent the helper would otherwise override")
    ap.add_argument("--alt", type=float, default=10.0)
    ap.add_argument("--param", action="append", default=[], help="NAME=VALUE, set after boot (repeatable)")
    a = ap.parse_args()
    cop = Copter("tcp:127.0.0.1:5760")
    shared = {"target": (0.0, 0.0, -a.alt), "phase": "takeoff"}
    stop = threading.Event()
    th = threading.Thread(target=recorder, args=("tcp:127.0.0.1:5762", a.out, stop, shared), daemon=True); th.start()
    c = cop.c
    request(c, [("LOCAL_POSITION_NED", 10)])
    cop.wait_ready()
    set_param(c, "CC3_FAIL_MOT", float(a.motor if not (a.no_ftc or a.arm_before >= 0) else 0))
    for kv in a.param:
        k, v = kv.split("=")
        set_param(c, k, float(v))
    trigger(False)
    if a.plan == "mission":
        g = c.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=10)
        upload_mission(c, g.lat / 1e7, g.lon / 1e7, a.alt)
    cop.arm_takeoff(a.alt)
    def goto(n, e, wait):
        shared["target"] = (n, e, -a.alt)
        t_end = time.time() + wait
        while time.time() < t_end:
            cop.goto_offset(n, e, a.alt)
            c.recv_match(blocking=True, timeout=0.5)
    if a.plan == "mission":
        return fly_mission(a, cop, c, shared, stop, th)
    if a.arm_before >= 0:
        shared["phase"] = "hover"; goto(0, 0, max(0.0, 10 - a.arm_before))
        c.mav.command_long_send(c.target_system, c.target_component, mavutil.mavlink.MAV_CMD_DO_AUX_FUNCTION, 0,
                                109, 2, 0, 0, 0, 0, 0)
        print("FTC armed (pass-through, onboard detection)", flush=True)
        shared["phase"] = "armed"; goto(0, 0, a.arm_before)
    else:
        shared["phase"] = "hover"; goto(0, 0, 10)
    shared["phase"] = "failure"
    ok = trigger(True, a.event_log)
    print("failure trigger confirmed:", ok, flush=True)
    if not a.no_ftc and a.arm_before < 0:
        time.sleep(a.activate_delay)
        c.mav.command_long_send(c.target_system, c.target_component, mavutil.mavlink.MAV_CMD_DO_AUX_FUNCTION, 0,
                                109, 2, 0, 0, 0, 0, 0)  # aux function 109 (custom controller), HIGH
        print("FTC switched on", flush=True)
    if a.plan != "land":
        shared["phase"] = "hold_after_failure"
        if a.passive:
            t_end = time.time() + 25
            while time.time() < t_end:
                c.recv_match(blocking=True, timeout=0.5)
        else:
            goto(0, 0, 25)
    if a.plan == "square":
        for i, (n, e) in enumerate([(20, 0), (20, 20), (0, 20), (0, 0)]):
            shared["phase"] = f"wp{i + 1}"; goto(n, e, 20)
    shared["phase"] = "land"
    cop.mode("LAND")
    t0 = time.time()
    while time.time() - t0 < 40 and c.motors_armed():
        c.recv_match(blocking=True, timeout=0.5)
    trigger(False)
    stop.set(); th.join(timeout=3)
    print("done", flush=True)

def fly_mission(a, cop, c, shared, stop, th):
    """AUTO mission; failure during leg a.fail_leg; the FTC hand-off is left to the supervisor / onboard detection
    (or, without --no-ftc, the oracle switch right after the trigger, as in the other plans)."""
    request(c, [("MISSION_CURRENT", 5)])
    shared["phase"] = "hover"
    t_end = time.time() + 5
    while time.time() < t_end:
        cop.goto_offset(0, 0, a.alt); c.recv_match(blocking=True, timeout=0.5)
    cop.mode("AUTO")
    leg, leg_t0, failed, t_fail = 0, time.time(), False, None
    t_end = time.time() + 300
    armed = lambda: shared.get("armed", 1) == 1  # the autopilot's own heartbeat, from the recorder link  # noqa: E731
    while time.time() < t_end and armed():
        m = c.recv_match(type=["MISSION_CURRENT", "HEARTBEAT"], blocking=True, timeout=0.5)
        if m is not None and m.get_type() == "MISSION_CURRENT" and m.seq != leg:
            leg, leg_t0 = m.seq, time.time()
            print(f"leg {leg} started", flush=True)
            if leg <= len(SQUARE):
                shared["target"] = (*SQUARE[leg - 1], -a.alt)
            elif not failed or time.time() - t_fail > 1:
                shared["target"] = (*SQUARE[-1], 0.0)
        shared["phase"] = f"wp{leg}" if leg <= len(SQUARE) else "land"
        if not failed and leg == a.fail_leg and time.time() - leg_t0 >= a.fail_delay:
            failed, t_fail = True, time.time()
            shared["phase"] = "failure"
            print("failure trigger confirmed:", trigger(True, a.event_log), flush=True)
            if not a.no_ftc and a.arm_before < 0:
                time.sleep(a.activate_delay)
                c.mav.command_long_send(c.target_system, c.target_component, mavutil.mavlink.MAV_CMD_DO_AUX_FUNCTION, 0,
                                        109, 2, 0, 0, 0, 0, 0)
                print("FTC switched on", flush=True)
    print("mission finished, disarmed" if not armed() else "mission timeout", flush=True)
    if armed():
        cop.mode("LAND")
        t0 = time.time()
        while time.time() - t0 < 40 and c.motors_armed():
            c.recv_match(blocking=True, timeout=0.5)
    trigger(False)
    stop.set(); th.join(timeout=3)
    print("done", flush=True)


if __name__ == "__main__":
    main()
