#!/bin/bash
# Stop simulation / test processes: SIGTERM first, then SIGKILL whatever survives (arducopter and the gz sim
# server ignore SIGTERM). Script file, so the patterns never match the calling shell.
#   ./stop_sim.sh             everything, including the driver scripts (collect_data, eval_supervisor, ros_e2e_test)
#   ./stop_sim.sh --sim-only  only Gazebo, SITL, the takeoff helper, ROS nodes and topic echoes (used by the drivers)
SIM="[p]ython3 -$|[a]rducopter|[g]z sim|[g]z-sim|[r]uby.*gz|[f]ly_fault|[e]val_supervisor\.py|[m]otor_failure_predictor/|[r]os2 topic echo"
DRIVERS="[c]ollect_data|[c]ollect_all|[c]ollect_more|[e]val_supervisor\.sh|[e]val_all_supervisor|[r]os_e2e_test"
PAT="$SIM"; [ "$1" = "--sim-only" ] || PAT="$DRIVERS|$SIM"
for p in $(pgrep -f "$PAT"); do kill "$p" 2>/dev/null; done
sleep 3
for p in $(pgrep -f "$PAT"); do kill -9 "$p" 2>/dev/null; done
sleep 1
pgrep -fa "$PAT" || echo "all stopped"
