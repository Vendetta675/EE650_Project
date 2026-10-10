#!/bin/bash
# End-to-end test of detection -> fault-tolerant control hand-off (Gazebo window by default):
#   progressive failure of one rotor (LOE ramp 1.0 -> LOE over 5 s, then cut to zero CUT s later)
#   mavlink_bridge (udp 14551) -> /flight_data -> lstm_predictor (models_imu_v2) -> fault_supervisor ftc:=true (udp 14552)
#   alarm -> supervisor ARMED: custom controller on in pass-through (CC3_FAIL_MOT 0)
#   cut   -> ArduPilot detects the complete loss onboard (CC3_AUTO) and switches to three-rotor control -> supervisor FTC
# Flight helper + 10 Hz recorder: ftc_dev/ftc_test.py (--no-ftc: it never switches the FTC on itself).
#   PLAN=hold: failure in hover, hold 25 s, land.  PLAN=square: same, then a 20 m square in GUIDED, land.
#   PLAN=mission: AUTO mission (20 m square, then LAND); the failure starts 3 s into leg 2, so the rotor is lost
#   mid-leg while flying, and legs 3-4 and the landing are flown on three rotors.
# Light airframe sim_models_ftc10 (1.0 kg).
# Options (environment): MOTOR=1 (1-4), PLAN=hold|square|mission, MODEL_DIR=models_v3 (detector), DESCENT=1.0 (m/s, the full-battery supervisor descent), LB_DESCENT=1.5 (m/s, low-battery landing below SoC 0.7), SLOW_ALT=0 (m; >0: adaptive, 0.7 m/s below it), CUT_ALT=0 (m; >0: cut the motors below it), SOC=1.0 (initial battery state of charge, 3S 2.2 Ah), LOE=0.6, CUT=2 (s after the 5 s ramp), FTC=1 (0 = supervisor without FTC),
#   HEADLESS=1 (no window), KEEP_OPEN=1. Results: ros_ftc_e2e/ (run.csv, test.log, supervisor.log, states.log, logs/*.BIN)
WS=/root/ros2_ws; D=$WS/ftc_dev; OUT=$WS/ros_ftc_e2e; MODELS=$WS/sim_models_ftc10
MOTOR=${MOTOR:-1}; PLAN=${PLAN:-hold}; SOC=${SOC:-1.0}; LOE=${LOE:-0.6}; CUT=${CUT:-2}; FTC=${FTC:-1}
SDF=$MODELS/iris_with_gimbal/model.sdf
source /opt/ros/jazzy/setup.bash; source $WS/install/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$WS/install/motor_failure/lib:/root/ardupilot_gazebo/build
export GZ_SIM_RESOURCE_PATH=$MODELS:$WS/worlds:/root/ardupilot_gazebo/models:/root/ardupilot_gazebo/worlds

restore_model() {
  sed -i "s#<loe_final>[^<]*</loe_final>#<loe_final>0.8</loe_final>#; s#<ramp_duration>[^<]*</ramp_duration>#<ramp_duration>5.0</ramp_duration>#; s#<cut_after>[^<]*</cut_after>#<cut_after>-1</cut_after>#; s#<failure_motor>[^<]*</failure_motor>#<failure_motor>-1</failure_motor>#; s#<stutter_amplitude>[^<]*</stutter_amplitude>#<stutter_amplitude>0.15</stutter_amplitude>#; s#<event_log>[^<]*</event_log>#<event_log>$WS/fault_events.csv</event_log>#; s#<soc_init>[^<]*</soc_init>#<soc_init>1.0</soc_init>#" $SDF
}
cleanup() {
  restore_model
  [ "${KEEP_OPEN:-0}" = 1 ] && { echo "KEEP_OPEN=1: Gazebo and SITL left running (./stop_sim.sh to stop)."; return; }
  $WS/stop_sim.sh --sim-only > /dev/null; echo "stopped all simulation processes"
}
trap 'echo; echo interrupted; KEEP_OPEN=0; cleanup; exit 1' INT TERM

case $MOTOR in 1|2|3|4) ;; *) echo "MOTOR must be 1-4"; exit 1;; esac
case $PLAN in hold|square|mission) ;; *) echo "PLAN must be hold, square or mission"; exit 1;; esac
if pgrep -f "[a]rducopter|[g]z sim" > /dev/null; then
  echo "A simulator is still running from an earlier run; stopping it first."; $WS/stop_sim.sh --sim-only > /dev/null
fi

sed -i "s#<loe_final>[^<]*</loe_final>#<loe_final>$LOE</loe_final>#; s#<ramp_duration>[^<]*</ramp_duration>#<ramp_duration>5.0</ramp_duration>#; s#<cut_after>[^<]*</cut_after>#<cut_after>$CUT</cut_after>#; s#<failure_motor>[^<]*</failure_motor>#<failure_motor>$((MOTOR-1))</failure_motor>#; s#<stutter_amplitude>[^<]*</stutter_amplitude>#<stutter_amplitude>0.15</stutter_amplitude>#; s#<event_log>[^<]*</event_log>#<event_log>$OUT/events.csv</event_log>#; s#<soc_init>[^<]*</soc_init>#<soc_init>$SOC</soc_init>#" $SDF

mkdir -p $OUT && cd $OUT && rm -rf run.csv test.log states.log events.csv logs battery_state.txt
if [ "${HEADLESS:-0}" = 1 ]; then
  echo "Gazebo: headless server (HEADLESS=1)"
  gz sim -s -r -v2 iris_runway_headless.sdf > gz.log 2>&1 & disown
else
  echo "Gazebo: GUI on DISPLAY=$DISPLAY (iris_runway.sdf)"
  gz sim -r -v3 iris_runway.sdf > gz.log 2>&1 & disown
fi
sleep 15
/root/ardupilot/build/sitl/bin/arducopter --model JSON --speedup 1 --slave 0 -I0 --sim-address=127.0.0.1 -w \
  --serial5=udpclient:127.0.0.1:14551 --serial6=udpclient:127.0.0.1:14552 \
  --home -35.363261,149.165230,584,353 \
  --defaults /root/ardupilot/Tools/autotest/default_params/copter.parm,/root/ardupilot/Tools/autotest/default_params/gazebo-iris.parm,$D/ftc.parm,$D/ftc_e2e.parm \
  > sitl.log 2>&1 & disown
sleep 5
ros2 run motor_failure_predictor mavlink_bridge --ros-args -p mavlink_url:=udp:127.0.0.1:14551 > bridge.log 2>&1 &
ros2 run motor_failure_predictor lstm_predictor --ros-args -p model_dir:=${MODEL_DIR:-$WS/models_v3} > predictor.log 2>&1 &
ros2 run motor_failure_predictor fault_supervisor --ros-args -p mavlink_url:=udp:127.0.0.1:14552 -p model_dir:=${MODEL_DIR:-$WS/models_v3} \
  -p descent_speed:=${DESCENT:-1.0} -p low_batt_descent_speed:=${LB_DESCENT:-1.5} -p low_batt_slow_alt:=${SLOW_ALT:-0.0} -p low_batt_cut_alt:=${CUT_ALT:-0.0} -p ftc:=$([ "$FTC" = 1 ] && echo true || echo false) > supervisor.log 2>&1 &
timeout 600 ros2 topic echo /fault_supervisor/state std_msgs/msg/String > states.log 2>&1 &
timeout 600 gz topic -e -t /model/iris_with_gimbal/battery_state > battery_state.txt 2>&1 &
echo "plan $PLAN, battery SoC $SOC, motor $MOTOR: LOE ramp to $LOE over 5 s, cut $CUT s later; FTC hand-off=$FTC (takeoff after EKF/GPS lock, ~1 min)"
timeout 560 python3 $D/ftc_test.py --out $OUT/run.csv --event-log $OUT/events.csv --motor $MOTOR --plan $PLAN --no-ftc $([ $PLAN = hold ] && echo --passive) 2>&1 \
  | grep --line-buffered -v -E "EOF on TCP|sleeping" | tee test.log
echo "--- supervisor:"; grep -E "supervisor:|Supervisor armed" supervisor.log | sed 's/^\[[A-Z]*\] //'
echo "--- plugin ground truth:"; cut -d, -f2,3,4,6 events.csv | tail -n +2 | tail -4
echo "--- state sequence (10 Hz samples):"; grep data states.log | uniq -c
echo "--- summary"; (cd $D && python3 summarize.py $OUT 2>&1 | tail -2)
[ "$PLAN" = hold ] || (cd $D && python3 nav_summary.py $OUT 2>&1)
cleanup
