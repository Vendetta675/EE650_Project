#!/bin/bash
# Collect labelled flight data: headless Gazebo + ArduCopter SITL + progressive-failure episodes.
# Usage: ./collect_data.sh <loe_final> <n_flights> <episodes_per_flight> [first_flight_index]
#   writes flights/flight_XX.csv (+ sim_logs/), appends ground truth to fault_events.csv.
# Severity is set on a copy of the model (sim_models/iris_with_gimbal), which shadows the original on the
# resource path; the original ardupilot_gazebo model is not modified by this script.
LOE=${1:-0.6}; N=${2:-1}; EP=${3:-1}; FIRST=${4:-1}
WS=/root/ros2_ws
cd $WS
source /opt/ros/jazzy/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$WS/install/motor_failure/lib:/root/ardupilot_gazebo/build
export GZ_SIM_RESOURCE_PATH=$WS/sim_models:$WS/worlds:/root/ardupilot_gazebo/models:/root/ardupilot_gazebo/worlds
FLIGHTS=${FLIGHT_DIR:-flights}; EVLOG=${EVENT_LOG:-$WS/fault_events.csv}
mkdir -p $FLIGHTS sim_logs
sed -i "s#<event_log>[^<]*</event_log>#<event_log>$EVLOG</event_log>#" sim_models/iris_with_gimbal/model.sdf
sed -i "s#<loe_final>[^<]*</loe_final>#<loe_final>$LOE</loe_final>#" sim_models/iris_with_gimbal/model.sdf
for k in $(seq 0 $((N - 1))); do
  i=$(printf "%02d" $((FIRST + k)))
  echo "=== flight $i loe_final=$LOE $(date +%T)"
  if [ "${HEADLESS:-0}" = 1 ]; then   # opt-out: no window, headless world (no GL needed)
    gz sim -s -r -v2 iris_runway_headless.sdf > sim_logs/gz_$i.log 2>&1 &
  else                                 # default: Gazebo GUI on $DISPLAY with the original world
    gz sim -r -v2 iris_runway.sdf > sim_logs/gz_$i.log 2>&1 &
  fi
  sleep 15
  # arducopter directly (sim_vehicle.py would open an xterm without MAVProxy); same defaults as sim_vehicle
  (cd sim_logs && /root/ardupilot/build/sitl/bin/arducopter --model JSON --speedup 1 --slave 0 -I0 \
      --sim-address=127.0.0.1 --home -35.363261,149.165230,584,353 \
      --defaults /root/ardupilot/Tools/autotest/default_params/copter.parm,/root/ardupilot/Tools/autotest/default_params/gazebo-iris.parm \
      > sitl_$i.log 2>&1) &
  sleep 5
  timeout 1800 python3 fly_fault_episodes.py --episodes $EP --out $FLIGHTS/flight_$i.csv --seed $((10#$i)) ${FLY_ARGS:-} \
      2>&1 | grep --line-buffered -v "sleeping" | tee sim_logs/fly_$i.log
  echo "$i,$LOE" >> $FLIGHTS/flight_severity.csv
  $WS/stop_sim.sh --sim-only > /dev/null
  sleep 5
done
echo "=== done $(date +%T)"
