"""Battery model check (SITL + Gazebo): takeoff, 20 s hover, 1.5 s full-throttle climb, land.
Logs at 10 Hz: SITL battery (BATTERY_STATUS voltage / current / remaining), motor PWM, altitude, vz; the plugin's own
state comes from the gz topic (battery_sim.sh records it). Prints hover current / voltage / PWM and the thrust-to-weight
ratio from the climb: TWR = 1 + a_up / g. Usage: python3 battery_test.py <out.csv>"""
import csv, math, sys, threading, time
import numpy as np
from pymavlink import mavutil
sys.path.insert(0, "/root/ros2_ws"); sys.path.insert(0, "/root/ros2_ws/ftc_dev")
from fly_fault_episodes import Copter, trigger
from ftc_test import request

def recorder(url, out, stop, shared):
    c = mavutil.mavlink_connection(url, source_system=254); c.wait_heartbeat(timeout=60)
    request(c, [("LOCAL_POSITION_NED", 20), ("SERVO_OUTPUT_RAW", 20), ("BATTERY_STATUS", 10), ("ATTITUDE", 10)])
    st = dict(z=float("nan"), vz=0.0, m1=0, m2=0, m3=0, m4=0, volt=float("nan"), amp=float("nan"), remaining=-1, roll=0.0, pitch=0.0)
    with open(out, "w", newline="", buffering=1) as f:
        w = csv.DictWriter(f, fieldnames=["t"] + list(st) + ["phase"]); w.writeheader(); nxt = time.time()
        while not stop.is_set():
            m = c.recv_match(blocking=True, timeout=0.02)
            if m is not None:
                t = m.get_type()
                if t == "LOCAL_POSITION_NED": st.update(z=m.z, vz=m.vz)
                elif t == "SERVO_OUTPUT_RAW": st.update(m1=m.servo1_raw, m2=m.servo2_raw, m3=m.servo3_raw, m4=m.servo4_raw)
                elif t == "BATTERY_STATUS" and m.id == 0:
                    st.update(volt=m.voltages[0] / 1000.0, amp=m.current_battery / 100.0, remaining=m.battery_remaining)
                elif t == "ATTITUDE": st.update(roll=math.degrees(m.roll), pitch=math.degrees(m.pitch))
            if time.time() >= nxt:
                nxt += 0.1; w.writerow({"t": time.time(), **st, "phase": shared["phase"]})

def main():
    out = sys.argv[1]
    cop = Copter("tcp:127.0.0.1:5760"); c = cop.c
    shared = {"phase": "takeoff"}; stop = threading.Event()
    th = threading.Thread(target=recorder, args=("tcp:127.0.0.1:5762", out, stop, shared), daemon=True); th.start()
    cop.wait_ready(); trigger(False); cop.arm_takeoff(10.0)
    shared["phase"] = "hover"; t0 = time.time()
    while time.time() - t0 < 20:
        cop.goto_offset(0, 0, 10.0); c.recv_match(blocking=True, timeout=0.5)
    cop.mode("STABILIZE"); shared["phase"] = "climb"; t0 = time.time()
    while time.time() - t0 < 1.5:  # full throttle stick, attitude centred
        c.mav.rc_channels_override_send(c.target_system, c.target_component, 1500, 1500, 2000, 1500, 0, 0, 0, 0)
        time.sleep(0.05)
    c.mav.rc_channels_override_send(c.target_system, c.target_component, 1500, 1500, 1500, 1500, 0, 0, 0, 0)
    shared["phase"] = "end"; cop.mode("LAND"); time.sleep(4); stop.set(); th.join(timeout=3)
    import pandas as pd
    d = pd.read_csv(out)
    h = d[d.phase == "hover"].iloc[50:]  # last 15 s of the hover
    cl = d[d.phase == "climb"]; cl = cl[(cl.t - cl.t.iloc[0] > 0.3)]
    a_up = -np.polyfit(cl.t - cl.t.iloc[0], cl.vz, 1)[0]
    pwm = h[["m1", "m2", "m3", "m4"]].to_numpy(float).mean()
    print(f"RESULT hover: SITL {h.volt.mean():.2f} V, {h.amp.mean():.1f} A, remaining {int(h.remaining.iloc[-1])} %, "
          f"motor PWM {pwm:.0f} | climb: a_up {a_up:.2f} m/s^2, TWR {1 + a_up / 9.81:.2f}, "
          f"climb current {cl.amp.max():.1f} A, min voltage {cl.volt.min():.2f} V", flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
