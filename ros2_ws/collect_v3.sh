#!/bin/bash
# v3 data collection (2026-10-09): battery model on, randomised per flight (plan from a fixed seed, flights_v3/plan.csv):
#   airframe  stock Iris 1.98 kg (sim_models, default params) / light FTC airframe 1.0 kg (sim_models_ftc10, ftc.parm +
#             ftc_e2e.parm, as deployed with the fault-tolerant controller), alternating
#   battery   initial SoC uniform 0.4-1.0, internal resistance x1-2.5 (aged packs)
#   fault     LOE 0.6 / 0.7 / 0.8 in turn, random motor; every 3rd flight has long normal phases (40-70 s)
# Headless. Every trigger is confirmed against the plugin's event log (fault_events_v3.csv). ~2 h for 28 flights.
# Usage: ./collect_v3.sh [first] [last]     (flight numbers from the plan; default all)
WS=/root/ros2_ws; cd $WS
source /opt/ros/jazzy/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$WS/install/motor_failure/lib:/root/ardupilot_gazebo/build
OUT=$WS/flights_v3; EVLOG=$WS/fault_events_v3.csv; D=/root/ardupilot/Tools/autotest/default_params
mkdir -p $OUT sim_logs_v3
[ -f $OUT/plan.csv ] || python3 - <<PY
import random
r = random.Random(2026)
with open("$OUT/plan.csv", "w") as f:
    f.write("flight,airframe,soc,r_scale,loe,episodes,normal_min,normal_max\n")
    for i in range(1, 29):
        long = i % 3 == 0
        f.write(f"{i:02d},{'light' if i % 2 == 0 else 'stock'},{r.uniform(0.4, 1.0):.2f},{r.uniform(1.0, 2.5):.2f},"
                f"{[0.6, 0.7, 0.8][(i - 1) // 2 % 3]},{2 if long else 3},{40 if long else 15},{70 if long else 30}\n")
PY
FIRST=${1:-1}; LAST=${2:-28}
restore() {  # back to the defaults of each model
  sed -i "s#<event_log>[^<]*</event_log>#<event_log>$WS/fault_events.csv</event_log>#; s#<loe_final>[^<]*</loe_final>#<loe_final>0.8</loe_final>#; s#<soc_init>[^<]*</soc_init>#<soc_init>1.0</soc_init>#; s#<r0_cell>[^<]*</r0_cell>#<r0_cell>0.006</r0_cell>#; s#<r1_cell>[^<]*</r1_cell>#<r1_cell>0.004</r1_cell>#" $WS/sim_models/iris_with_gimbal/model.sdf
  sed -i "s#<event_log>[^<]*</event_log>#<event_log>$WS/fault_events.csv</event_log>#; s#<loe_final>[^<]*</loe_final>#<loe_final>0.8</loe_final>#; s#<soc_init>[^<]*</soc_init>#<soc_init>1.0</soc_init>#; s#<r0_cell>[^<]*</r0_cell>#<r0_cell>0.010</r0_cell>#; s#<r1_cell>[^<]*</r1_cell>#<r1_cell>0.006</r1_cell>#" $WS/sim_models_ftc10/iris_with_gimbal/model.sdf
}
trap 'echo interrupted; $WS/stop_sim.sh --sim-only > /dev/null; restore; exit 1' INT TERM
tail -n +2 $OUT/plan.csv | while IFS=, read -r i air soc rs loe ep nmin nmax; do
  [ $((10#$i)) -lt $FIRST ] || [ $((10#$i)) -gt $LAST ] && continue
  if [ $air = light ]; then M=$WS/sim_models_ftc10; R0=0.010; R1=0.006; PARM=$D/copter.parm,$D/gazebo-iris.parm,$WS/ftc_dev/ftc.parm,$WS/ftc_dev/ftc_e2e.parm
  else M=$WS/sim_models; R0=0.006; R1=0.004; PARM=$D/copter.parm,$D/gazebo-iris.parm; fi
  r0=$(python3 -c "print(round($R0*$rs,4))"); r1=$(python3 -c "print(round($R1*$rs,4))")
  sed -i "s#<event_log>[^<]*</event_log>#<event_log>$EVLOG</event_log>#; s#<loe_final>[^<]*</loe_final>#<loe_final>$loe</loe_final>#; s#<soc_init>[^<]*</soc_init>#<soc_init>$soc</soc_init>#; s#<r0_cell>[^<]*</r0_cell>#<r0_cell>$r0</r0_cell>#; s#<r1_cell>[^<]*</r1_cell>#<r1_cell>$r1</r1_cell>#; s#<failure_motor>[^<]*</failure_motor>#<failure_motor>-1</failure_motor>#; s#<cut_after>[^<]*</cut_after>#<cut_after>-1</cut_after>#; s#<ramp_duration>[^<]*</ramp_duration>#<ramp_duration>5.0</ramp_duration>#; s#<stutter_amplitude>[^<]*</stutter_amplitude>#<stutter_amplitude>0.15</stutter_amplitude>#" $M/iris_with_gimbal/model.sdf
  export GZ_SIM_RESOURCE_PATH=$M:$WS/worlds:/root/ardupilot_gazebo/models:/root/ardupilot_gazebo/worlds
  echo "=== flight $i $air soc=$soc r0x$rs loe=$loe episodes=$ep $(date +%T)"
  $WS/stop_sim.sh --sim-only > /dev/null
  gz sim -s -r -v2 iris_runway_headless.sdf > sim_logs_v3/gz_$i.log 2>&1 &
  sleep 12
  (cd sim_logs_v3 && /root/ardupilot/build/sitl/bin/arducopter --model JSON --speedup 1 --slave 0 -I0 -w \
      --sim-address=127.0.0.1 --home -35.363261,149.165230,584,353 --defaults $PARM > sitl_$i.log 2>&1) &
  sleep 5
  timeout 1200 python3 fly_fault_episodes.py --episodes $ep --out $OUT/flight_$i.csv --seed $((100 + 10#$i)) \
      --normal-min $nmin --normal-max $nmax --event-log $EVLOG < /dev/null 2>&1 | grep --line-buffered -v "sleeping" > sim_logs_v3/fly_$i.log
  grep -E "trigger at|WARNING|lost altitude|flight done|Error" sim_logs_v3/fly_$i.log | tr "\n" ";" ; echo
  echo "$i,$loe" >> $OUT/flight_severity.csv
  $WS/stop_sim.sh --sim-only > /dev/null
  restore
done
echo "=== done $(date +%T)"
