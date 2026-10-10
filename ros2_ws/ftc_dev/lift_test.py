"""Maximum-lift test: healthy Iris, GUIDED climb at a velocity it cannot reach (-12 m/s) for 2.5 s; logs vz and motors.
Gives the real all-motor thrust capacity: a_up = 4 T_max_eff / m - g."""
import csv, sys, threading, time
from pymavlink import mavutil
sys.path.insert(0, "/root/ros2_ws"); sys.path.insert(0, "/root/ros2_ws/ftc_dev")
from fly_fault_episodes import Copter, trigger
from ftc_test import recorder
cop = Copter("tcp:127.0.0.1:5760"); c = cop.c
shared = {"target": (0, 0, -10), "phase": "takeoff"}; stop = threading.Event()
th = threading.Thread(target=recorder, args=("tcp:127.0.0.1:5762", sys.argv[1], stop, shared), daemon=True); th.start()
cop.wait_ready(); trigger(False); cop.arm_takeoff(10.0)
shared["phase"] = "hover"; t0 = time.time()
while time.time() - t0 < 6: c.recv_match(blocking=True, timeout=0.5)
cop.mode("STABILIZE")
shared["phase"] = "climb"; t0 = time.time()
while time.time() - t0 < 1.5:   # full throttle stick, roll/pitch/yaw centred
    c.mav.rc_channels_override_send(c.target_system, c.target_component, 1500, 1500, 2000, 1500, 0, 0, 0, 0)
    time.sleep(0.05)
c.mav.rc_channels_override_send(c.target_system, c.target_component, 1500, 1500, 1500, 1500, 0, 0, 0, 0)
shared["phase"] = "end"
cop.mode("LAND"); time.sleep(3); stop.set(); th.join(timeout=3); print("done")
