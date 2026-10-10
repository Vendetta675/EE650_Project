#!/bin/bash
# Paired closed-loop supervisor evaluation (2026-10-08). Final design (fixed after 2 smoke flights): detector
# models_imu_v2 (INT8 GRU-5, IMU features), HOLD -> DESCEND at 1.0 m/s -> LAND below 2 m, no yaw release.
# Each flight number is flown with the supervisor ON and OFF: same failing motor (flight % 4), same seed.
cd /root/ros2_ws
export MODEL_DIR=models_imu_v2 FLY_ARGS="--normal-min 40 --normal-max 60 --descent-speed 1.0"
for spec in "0.7 1" "0.6 5" "0.8 9"; do
  set -- $spec
  for sup in on off; do
    SUPERVISOR=$sup FLIGHT_DIR=supervisor_eval_$sup EVENT_LOG=/root/ros2_ws/fault_events_sup_$sup.csv ./eval_supervisor.sh $1 4 $2
  done
done
echo ALL_DONE
