#!/bin/bash
# Development harness: Gazebo with the GUI (HEADLESS=1: server only) + ArduCopter SITL (with the FTC custom controller enabled) + ftc_test.py.
# Usage: ./ftc_sim.sh <run_name> <ftc_test.py args...>   (complete, instant failure of the given motor)
RUN=$1; shift
WS=/root/ros2_ws; D=$WS/ftc_dev; mkdir -p $D/$RUN
source /opt/ros/jazzy/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$WS/install/motor_failure/lib:/root/ardupilot_gazebo/build
export GZ_SIM_RESOURCE_PATH=${MODELS:-$WS/sim_models}:$WS/worlds:/root/ardupilot_gazebo/models:/root/ardupilot_gazebo/worlds
MOTOR=$(echo "$@" | grep -oP "(?<=--motor )\d" || echo 1)
sed -i "s#<loe_final>[^<]*</loe_final>#<loe_final>0.0</loe_final>#; s#<ramp_duration>[^<]*</ramp_duration>#<ramp_duration>0.05</ramp_duration>#; s#<cut_after>[^<]*</cut_after>#<cut_after>0</cut_after>#; s#<failure_motor>[^<]*</failure_motor>#<failure_motor>$((MOTOR-1))</failure_motor>#; s#<stutter_amplitude>[^<]*</stutter_amplitude>#<stutter_amplitude>0.0</stutter_amplitude>#; s#<event_log>[^<]*</event_log>#<event_log>$D/events.csv</event_log>#" ${MODELS:-$WS/sim_models}/iris_with_gimbal/model.sdf
$WS/stop_sim.sh --sim-only > /dev/null
cd $D/$RUN
if [ "${HEADLESS:-0}" = 1 ]; then
  gz sim -s -r -v2 iris_runway_headless.sdf > gz.log 2>&1 & disown
else
  gz sim -r -v2 iris_runway.sdf > gz.log 2>&1 & disown
fi
sleep 15
/root/ardupilot/build/sitl/bin/arducopter --model JSON --speedup 1 --slave 0 -I0 --sim-address=127.0.0.1 -w \
  --home -35.363261,149.165230,584,353 \
  --defaults /root/ardupilot/Tools/autotest/default_params/copter.parm,/root/ardupilot/Tools/autotest/default_params/gazebo-iris.parm,$D/ftc.parm${EXTRA_PARM:+,$EXTRA_PARM} \
  > sitl.log 2>&1 & disown
sleep 5
timeout 400 python3 $D/ftc_test.py --out $D/$RUN/run.csv --event-log $D/events.csv "$@" 2>&1 | grep -v -E "EOF on TCP|sleeping" | tee test.log
cp eeprom.bin /dev/null 2>&1; ls logs 2>/dev/null | tail -1
$WS/stop_sim.sh --sim-only > /dev/null
# restore the simulation model copy to the data-collection defaults
sed -i "s#<loe_final>[^<]*</loe_final>#<loe_final>0.8</loe_final>#; s#<ramp_duration>[^<]*</ramp_duration>#<ramp_duration>5.0</ramp_duration>#; s#<cut_after>[^<]*</cut_after>#<cut_after>-1</cut_after>#; s#<failure_motor>[^<]*</failure_motor>#<failure_motor>-1</failure_motor>#; s#<stutter_amplitude>[^<]*</stutter_amplitude>#<stutter_amplitude>0.15</stutter_amplitude>#; s#<event_log>[^<]*</event_log>#<event_log>$WS/fault_events.csv</event_log>#" ${MODELS:-$WS/sim_models}/iris_with_gimbal/model.sdf
