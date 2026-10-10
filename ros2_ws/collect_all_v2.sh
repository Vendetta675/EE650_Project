#!/bin/bash
# v2 data collection (2026-10-08): same flight plan as v1 (collect_all.sh + collect_more_normal.sh), recorder with IMU.
cd /root/ros2_ws
export FLIGHT_DIR=flights_v2 EVENT_LOG=/root/ros2_ws/fault_events_v2.csv
./collect_data.sh 0.8 5 4 1
./collect_data.sh 0.6 8 1 6
./collect_data.sh 0.7 3 1 14
FLY_ARGS="--normal-min 40 --normal-max 70" ./collect_data.sh 0.8 3 3 17
FLY_ARGS="--normal-min 40 --normal-max 70" ./collect_data.sh 0.6 2 1 20
FLY_ARGS="--normal-min 40 --normal-max 70" ./collect_data.sh 0.7 1 1 22
echo ALL_DONE
