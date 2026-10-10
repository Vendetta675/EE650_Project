#!/bin/bash
# Fault-tolerant control demo with the Gazebo window: ArduCopter SITL with the FTC custom controller (CC_TYPE 3),
# light test airframe (sim_models_ftc, 1.2 kg). Takeoff to 10 m, hover 10 s, complete instant failure of one rotor,
# FTC switched on, then hold 25 s (PLAN=hold) or fly a 20 m square at 1 m/s (PLAN=square), then land.
# Options (environment): MOTOR=1 (1-4), PLAN=hold|square, FTC=1 (0 = baseline without FTC: it crashes in ~3 s),
#   HEADLESS=1 (no window), KEEP_OPEN=1 (leave Gazebo + SITL running at the end; stop with ./stop_sim.sh)
# Results: ftc_gui/run.csv (10 Hz), ftc_gui/test.log, ftc_gui/logs/*.BIN (CCF = FTC controller record)
WS=/root/ros2_ws; D=$WS/ftc_dev; OUT=$WS/ftc_gui; MODELS=$WS/sim_models_ftc
MOTOR=${MOTOR:-1}; PLAN=${PLAN:-hold}; FTC=${FTC:-1}
SDF=$MODELS/iris_with_gimbal/model.sdf
source /opt/ros/jazzy/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$WS/install/motor_failure/lib:/root/ardupilot_gazebo/build
export GZ_SIM_RESOURCE_PATH=$MODELS:$WS/worlds:/root/ardupilot_gazebo/models:/root/ardupilot_gazebo/worlds

restore_model() {
  sed -i "s#<loe_final>[^<]*</loe_final>#<loe_final>0.8</loe_final>#; s#<ramp_duration>[^<]*</ramp_duration>#<ramp_duration>5.0</ramp_duration>#; s#<cut_after>[^<]*</cut_after>#<cut_after>-1</cut_after>#; s#<failure_motor>[^<]*</failure_motor>#<failure_motor>-1</failure_motor>#; s#<stutter_amplitude>[^<]*</stutter_amplitude>#<stutter_amplitude>0.15</stutter_amplitude>#; s#<event_log>[^<]*</event_log>#<event_log>$WS/fault_events.csv</event_log>#" $SDF
}
cleanup() {
  restore_model
  [ "${KEEP_OPEN:-0}" = 1 ] && { echo "KEEP_OPEN=1: Gazebo and SITL left running (./stop_sim.sh to stop)."; return; }
  $WS/stop_sim.sh --sim-only > /dev/null; echo "stopped all simulation processes"
}
trap 'echo; echo interrupted; KEEP_OPEN=0; cleanup; exit 1' INT TERM

case $MOTOR in 1|2|3|4) ;; *) echo "MOTOR must be 1-4"; exit 1;; esac
case $PLAN in hold|square) ;; *) echo "PLAN must be hold or square"; exit 1;; esac
if pgrep -f "[a]rducopter|[g]z sim" > /dev/null; then
  echo "A simulator is still running from an earlier run; stopping it first."; $WS/stop_sim.sh --sim-only > /dev/null
fi

# complete, instant failure of the chosen motor (the plugin fires when ftc_test.py triggers it)
sed -i "s#<loe_final>[^<]*</loe_final>#<loe_final>0.0</loe_final>#; s#<ramp_duration>[^<]*</ramp_duration>#<ramp_duration>0.05</ramp_duration>#; s#<cut_after>[^<]*</cut_after>#<cut_after>0</cut_after>#; s#<failure_motor>[^<]*</failure_motor>#<failure_motor>$((MOTOR-1))</failure_motor>#; s#<stutter_amplitude>[^<]*</stutter_amplitude>#<stutter_amplitude>0.0</stutter_amplitude>#; s#<event_log>[^<]*</event_log>#<event_log>$OUT/events.csv</event_log>#" $SDF

mkdir -p $OUT && cd $OUT && rm -f run.csv test.log
if [ "${HEADLESS:-0}" = 1 ]; then
  echo "Gazebo: headless server (HEADLESS=1)"
  gz sim -s -r -v2 iris_runway_headless.sdf > gz.log 2>&1 & disown
else
  echo "Gazebo: GUI on DISPLAY=$DISPLAY (iris_runway.sdf)"
  gz sim -r -v3 iris_runway.sdf > gz.log 2>&1 & disown
fi
sleep 15
/root/ardupilot/build/sitl/bin/arducopter --model JSON --speedup 1 --slave 0 -I0 --sim-address=127.0.0.1 -w \
  --home -35.363261,149.165230,584,353 \
  --defaults /root/ardupilot/Tools/autotest/default_params/copter.parm,/root/ardupilot/Tools/autotest/default_params/gazebo-iris.parm,$D/ftc.parm \
  > sitl.log 2>&1 & disown
sleep 5
echo "motor $MOTOR fails 10 s after reaching 10 m; plan=$PLAN; FTC=$FTC (takeoff after EKF/GPS lock, ~1 min)"
ARGS="--motor $MOTOR --plan $PLAN"; [ "$FTC" = 0 ] && ARGS="$ARGS --no-ftc"
timeout 420 python3 $D/ftc_test.py --out $OUT/run.csv --event-log $OUT/events.csv $ARGS 2>&1 | grep --line-buffered -v -E "EOF on TCP|sleeping" | tee test.log
echo "--- summary"; (cd $D && python3 summarize.py $OUT 2>&1 | tail -2)
cleanup
