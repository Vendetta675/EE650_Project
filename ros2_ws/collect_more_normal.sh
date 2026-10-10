#!/bin/bash
# Extra flights (2026-10-08) to calibrate the false-alarm rate: long normal phases (40-70 s per episode).
cd /root/ros2_ws
export FLY_ARGS="--normal-min 40 --normal-max 70"
./collect_data.sh 0.8 3 3 17    # flights 17-19
./collect_data.sh 0.6 2 1 20    # flights 20-21
./collect_data.sh 0.7 1 1 22    # flight 22
echo ALL_DONE
