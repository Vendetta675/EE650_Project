#!/bin/bash
# End-to-end test of the ROS 2 pipeline on live SITL with the Gazebo window:
#   mavlink_bridge (tcp:5762) -> /flight_data -> lstm_predictor (models_imu_v2) -> fault_supervisor (tcp:5763)
# A helper takes off to 10 m on tcp:5760 and hovers; the progressive failure is triggered after 30 s; the supervisor
# should go NORMAL -> HOLD -> DESCEND -> LAND. Logs: ros_e2e/.
# Options (environment): LOE=0.8 (final effectiveness; 0.6 = vehicle is lost), MOTOR=-1 (0-3 or random),
#   CUT=-1 (seconds after the 5 s ramp until the motor is cut to zero = complete failure; -1 = never),
#   HEADLESS=1 (no window, headless world), KEEP_OPEN=1 (leave Gazebo + SITL running at the end; stop with ./stop_sim.sh)
WS=/root/ros2_ws; cd $WS
LOE=${LOE:-0.8}; MOTOR=${MOTOR:--1}; CUT=${CUT:--1}
source /opt/ros/jazzy/setup.bash; source install/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$WS/install/motor_failure/lib:/root/ardupilot_gazebo/build
export GZ_SIM_RESOURCE_PATH=$WS/sim_models:$WS/worlds:/root/ardupilot_gazebo/models:/root/ardupilot_gazebo/worlds

if pgrep -f "[a]rducopter|[g]z sim" > /dev/null; then
  echo "A simulator is still running from an earlier run; stopping it first."; $WS/stop_sim.sh --sim-only > /dev/null
fi
restore() { sed -i "s#<event_log>[^<]*</event_log>#<event_log>$WS/fault_events.csv</event_log>#; s#<failure_motor>[^<]*</failure_motor>#<failure_motor>-1</failure_motor>#; s#<cut_after>[^<]*</cut_after>#<cut_after>-1</cut_after>#" $WS/sim_models/iris_with_gimbal/model.sdf; }
cleanup() { restore; [ "${KEEP_OPEN:-0}" = 1 ] && { echo "KEEP_OPEN=1: Gazebo and SITL left running (./stop_sim.sh to stop)."; return; }
            $WS/stop_sim.sh --sim-only > /dev/null; echo "stopped all simulation processes"; }
trap 'echo; echo interrupted; KEEP_OPEN=0; cleanup; exit 1' INT TERM

mkdir -p ros_e2e && cd ros_e2e && rm -f fly.log states.log
sed -i "s#<loe_final>[^<]*</loe_final>#<loe_final>$LOE</loe_final>#; s#<failure_motor>[^<]*</failure_motor>#<failure_motor>$MOTOR</failure_motor>#; s#<cut_after>[^<]*</cut_after>#<cut_after>$CUT</cut_after>#; s#<event_log>[^<]*</event_log>#<event_log>$WS/ros_e2e/events.csv</event_log>#" \
    $WS/sim_models/iris_with_gimbal/model.sdf
if [ "${HEADLESS:-0}" = 1 ]; then
  echo "Gazebo: headless server (HEADLESS=1)"
  gz sim -s -r -v2 iris_runway_headless.sdf > gz.log 2>&1 &
else
  echo "Gazebo: GUI on DISPLAY=$DISPLAY (iris_runway.sdf)"
  gz sim -r -v3 iris_runway.sdf > gz.log 2>&1 &
fi
sleep 15
/root/ardupilot/build/sitl/bin/arducopter --model JSON --speedup 1 --slave 0 -I0 --sim-address=127.0.0.1 \
  --home -35.363261,149.165230,584,353 \
  --defaults /root/ardupilot/Tools/autotest/default_params/copter.parm,/root/ardupilot/Tools/autotest/default_params/gazebo-iris.parm \
  > sitl.log 2>&1 &
sleep 5
python3 - 2>&1 <<PY | grep --line-buffered -v -E "EOF on TCP socket|sleeping" > fly.log &
import sys, time; sys.path.insert(0, "$WS")
from fly_fault_episodes import Copter
c = Copter("tcp:127.0.0.1:5760"); c.wait_ready(); c.arm_takeoff(10.0); print("airborne", flush=True)
t = time.time()
while time.time() - t < 150: c.c.recv_match(blocking=True, timeout=0.5)
PY
sleep 3
ros2 run motor_failure_predictor mavlink_bridge --ros-args -p mavlink_url:=tcp:127.0.0.1:5762 > bridge.log 2>&1 &
ros2 run motor_failure_predictor lstm_predictor --ros-args -p model_dir:=$WS/models_imu_v2 > predictor.log 2>&1 &
ros2 run motor_failure_predictor fault_supervisor --ros-args -p mavlink_url:=tcp:127.0.0.1:5763 -p model_dir:=$WS/models_imu_v2 -p descent_speed:=1.0 > supervisor.log 2>&1 &
timeout 200 ros2 topic echo /fault_supervisor/state std_msgs/msg/String > states.log 2>&1 &
echo "waiting for takeoff (EKF + GPS lock take ~1 min) ..."
t0=$(date +%s)
until grep -q airborne fly.log; do
  sleep 2; [ $(( $(date +%s) - t0 )) -gt 240 ] && { echo "no takeoff after 4 min; see ros_e2e/fly.log, sitl.log, gz.log"; KEEP_OPEN=0; cleanup; exit 1; }
done
echo "airborne $(date +%T); triggering the failure (LOE -> $LOE$( [ "$CUT" != -1 ] && echo ", motor cut $CUT s after the 5 s ramp")) in 30 s"; sleep 30
python3 -c "import sys; sys.path.insert(0,'$WS'); from fly_fault_episodes import trigger; print('failure trigger confirmed by the plugin:', trigger(True, '$WS/ros_e2e/events.csv'))"
sleep 45
python3 -c "import sys; sys.path.insert(0,'$WS'); from fly_fault_episodes import trigger; trigger(False, '$WS/ros_e2e/events.csv')"
echo "--- /flight_data rate:"; timeout 5 ros2 topic hz /flight_data 2>/dev/null | grep average | tail -1
echo "--- supervisor:"; grep -E "supervisor:|Supervisor armed" supervisor.log | sed 's/^\[[A-Z]*\] //'
echo "--- predictor:"; grep -E "Loaded" predictor.log | sed 's/^\[[A-Z]*\] //'; echo "alarm log lines: $(grep -c 'FAULT DEVELOPING' predictor.log)"
echo "--- plugin ground truth (fault_events):"; cut -d, -f1,3,4,6 events.csv | tail -n +2 | tail -3
echo "--- state sequence (10 Hz samples):"; grep data states.log | uniq -c
cleanup
