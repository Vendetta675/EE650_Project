"""Headless copy of iris_runway.sdf for data collection: the Sensors (rendering) system is removed so the
server runs without a GPU/display. IMU and NavSat systems are unchanged, so ArduPilot gets the same inputs;
only the gimbal camera stops rendering."""
import re
src = "/root/ardupilot_gazebo/worlds/iris_runway.sdf"
dst = "/root/ros2_ws/worlds/iris_runway_headless.sdf"
s = open(src).read()
s2 = re.sub(r'\s*<plugin\s+filename="gz-sim-sensors-system".*?</plugin>', "", s, count=1, flags=re.S)
assert s2 != s, "Sensors system not found"
s2 = s2.replace("<world name=", "<!-- headless copy of iris_runway.sdf (make_headless_world.py): Sensors system removed -->\n  <world name=", 1)
open(dst, "w").write(s2)
print("wrote", dst)
