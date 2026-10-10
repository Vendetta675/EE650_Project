#!/bin/bash
# Battery model check, headless: Gazebo + SITL + battery_test.py. Usage: SOC=0.5 MODELS=<dir> ./battery_sim.sh <run>
# SOC = initial state of charge (written to <soc_init> for the run, restored after); MODELS default: sim_models (stock Iris).
# Results: ftc_dev/<run>/ (run.csv: SITL battery + motors, battery_state.txt: the plugin's own state, test.log)
RUN=$1; WS=/root/ros2_ws; D=$WS/ftc_dev; MODELS=${MODELS:-$WS/sim_models}; SDF=$MODELS/iris_with_gimbal/model.sdf
mkdir -p $D/$RUN; source /opt/ros/jazzy/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=$WS/install/motor_failure/lib:/root/ardupilot_gazebo/build
export GZ_SIM_RESOURCE_PATH=$MODELS:$WS/worlds:/root/ardupilot_gazebo/models:/root/ardupilot_gazebo/worlds
sed -i "s#<soc_init>[^<]*</soc_init>#<soc_init>${SOC:-1.0}</soc_init>#" $SDF
$WS/stop_sim.sh --sim-only > /dev/null
cd $D/$RUN && rm -f run.csv battery_state.txt
gz sim -s -r -v3 iris_runway_headless.sdf > gz.log 2>&1 &
sleep 10
timeout 300 gz topic -e -t /model/iris_with_gimbal/battery_state > battery_state.txt 2>&1 &
/root/ardupilot/build/sitl/bin/arducopter --model JSON --speedup 1 --slave 0 -I0 --sim-address=127.0.0.1 -w \
  --home -35.363261,149.165230,584,353 \
  --defaults /root/ardupilot/Tools/autotest/default_params/copter.parm,/root/ardupilot/Tools/autotest/default_params/gazebo-iris.parm${EXTRA_PARM:+,$EXTRA_PARM} \
  > sitl.log 2>&1 &
sleep 5
timeout 300 python3 $D/battery_test.py $D/$RUN/run.csv 2>&1 | grep -v -E "EOF on TCP|sleeping" > test.log
grep -E "RESULT|Traceback|Error" test.log
$WS/stop_sim.sh --sim-only > /dev/null
sed -i "s#<soc_init>[^<]*</soc_init>#<soc_init>1.0</soc_init>#" $SDF
