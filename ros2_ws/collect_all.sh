#!/bin/bash
# Full data collection batch (2026-10-08): flight numbers encode severity groups.
cd /root/ros2_ws
./collect_data.sh 0.8 5 4 1     # flights 01-05: survivable LOE, 4 episodes each
./collect_data.sh 0.6 8 1 6     # flights 06-13: LOE to 0.6 -> saturation and loss, 1 episode each
./collect_data.sh 0.7 3 1 14    # flights 14-16: intermediate, 1 episode each
echo ALL_DONE
