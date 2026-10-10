# Early motor-fault detection and response: progressive failures, precursor labels, GRU + INT8, IMU features,
# ROS 2 bridge, fault-response supervisor (2026-10-08)

Originals of every changed file are in `backup_2026-10-07/`.

## 1. Progressive failure simulation (Gazebo C++)

`src/motor_failure/src/MotorFailureSystem.cpp` (rebuilt with `colcon build --packages-select motor_failure`).

The plugin no longer zeroes the motor command. A trigger (`/model/iris_with_gimbal/trigger_random_failure`, `gz.msgs.Boolean`) starts
a **loss of effectiveness (LOE)** on a random rotor, or on a fixed one if `<failure_motor>` is set. That joint's `JointForceCmd` is
multiplied by k(t), which ramps from 1.0 to `<loe_final>` (0.6) over `<ramp_duration>` (5 s) of sim time and then holds. A **stutter**
(`<stutter_amplitude>` 0.15 × sin at `<stutter_frequency>` 35 Hz, with phase jitter) is added to the same command. Its amplitude grows
with the degradation. `<cut_after>` (seconds after the ramp, -1 = never) optionally cuts the motor at the end.

Why the force command: with `useForce=1` / `type=VELOCITY`, ArduPilotPlugin writes its rotor-velocity PID output to `JointForceCmd`
(motor torque), and LiftDrag turns rotor speed into thrust. So k scales torque, and the thrust loss is not exactly k. The plugin is
declared after ArduPilotPlugin in the model SDF, so it modifies the command of the current step.

**Ground truth:** every `start`, `cut` and `recover` event is appended to `fault_events.csv` (wall-clock + sim time, motor index, all
parameters). The live state `[active, motor, k, elapsed]` is published on `/model/iris_with_gimbal/motor_fault_state`. Parameters are
listed in the plugin header and in `ardupilot_gazebo/models/iris_with_gimbal/model.sdf`. Defaults are unchanged from the request.

**What the simulator does with it** (smoke test, `smoke/`): the controller compensates by driving the failing motor's PWM up. Its
relative PWM creeps from about 0 to +90 µs. At k ≈ 0.72 the motor saturates at 1950 µs, and with LOE 0.6 the vehicle is lost about
8 s after the trigger. ArduPilot itself reports "Potential Thrust Loss". At LOE 0.8 the copter stays airborne, and the relative-PWM
rise is only +20 to +27 µs.

## 2. Training for early detection

- `fault_features.py` (inside the `motor_failure_predictor` package; the workspace copy is a symlink) defines the features shared by
  recording, training and inference. They are `rel_m1..4` (each motor minus the mean of the other three), `vib_x/y/z`, `current`
  (BATTERY_STATUS, with SYS_STATUS as fallback) and `pwm_spread`. The EKF variances were dropped because they were 0 in every
  previously logged row.
- **Current:** recorded, but this SITL + Gazebo (JSON) setup does not simulate battery current. It is 0 A in every row, so
  `train_fault_models.py` drops constant features automatically. The pipeline uses current as soon as a vehicle reports it.
  No synthetic current was invented.
- **Data:** `collect_data.sh` / `collect_all.sh` run headless Gazebo and ArduCopter SITL, and `fly_fault_episodes.py` flies the
  episodes. Each episode is a normal phase with random GUIDED moves, then a trigger, 12 s of fault, recovery and stabilisation.
  The world is `worlds/iris_runway_headless.sdf`: the Sensors/rendering system is removed (the container has no GL); IMU and NavSat
  are unchanged. 16 flights produced 31 episodes: flights 01-05 at LOE 0.8 (4 episodes each), 06-13 at 0.6, and 14-16 at 0.7.
- **Precursor labelling:** `label_precursors.py` joins the flights with `fault_events.csv`. Labels: phase 1 is the 5 s ramp
  (precursor), phase 2 is degraded until recovery, phase 3 is the 5 s recovery transient (excluded). The target is 1 from the
  *first instant of degradation*, not from loss of thrust. Only airborne rows are used. Output: `flight_data_labeled.csv`.
- The previous `flight_data_live.csv` labels were the autoencoder's own alarms, not ground truth, so they were not reused.

## 3. GRU, 0.5 s window, INT8 TFLite

`train_fault_models.py` trains every variant on identical data, split by flight. Final data: 22 flights, 43 episodes.
Flights 17-22 were added to measure false alarms on more normal flight (they have 40-70 s normal phases).

- train: 01 02 03 06 07 08 09 14 17 22
- val: 04 10 11 15 18 20 (8.7 min of normal flight)
- test: 05 12 13 16 19 21 (11 episodes, 8.4 min of normal flight)

Test results (alarm = 2 consecutive windows above a threshold picked on val for ≤ 0.5 false alarms/min):

| model | params | ROC-AUC | detected | within 5 s precursor | median latency | false alarms / min |
|---|---|---|---|---|---|---|
| LSTM, 20 steps (previous design) | 5,793 | 0.862 | 7/11 | 4 | 3.8 s | 0.24 |
| GRU, 20 steps | 4,577 | 0.834 | 7/11 | 2 | 5.2 s | 0.12 |
| LSTM, 5 steps | 5,793 | 0.860 | 8/11 | 5 | 4.1 s | 1.07 |
| **GRU, 5 steps (deployed)** | **4,577** | **0.878** | **11/11** | **6** | 4.3 s | 1.19 |
| autoencoder LSTM-20 / GRU-5 (normal-only) | 13,832 / 10,632 | 0.46 / 0.49 | 4/11, 5/11 | – | – | 1.2 / 1.7 |

By severity (GRU-5 vs LSTM-20):

- **Severe faults (LOE 0.6/0.7, 4 test episodes):** both detect 4/4, at 3.1-3.5 s vs 3.1-3.8 s. That is around the moment the
  motor saturates (≈ 3.5 s) and several seconds before the vehicle is lost (≈ 8 s).
- **Subtle faults (0.8, 7 episodes):** GRU-5 detects 7/7 (1.1-9.4 s). LSTM-20 detects 3/7 (5.2-6.0 s).

**Deployed:** `models_gru/cls_gru5_int8.tflite` with `operating_point.json`. Persistence is chosen on *validation* only
(`persistence_tradeoff.py`: most precursor detections, then lowest latency), giving 2 windows. The threshold, 0.969, is
re-picked on the INT8 validation scores (`eval_int8.py`). On test: 11/11 detected, 6 within the precursor, median latency
4.3 s, 1.19 false alarms/min. This is identical to the Keras model under the same rule.

**How the three claims held up:**

- **GRU vs LSTM:** the recurrent layer has 23% fewer parameters (4,032 vs 5,248; 21% for the whole model), and accuracy is equal
  or better. Confirmed.
- **5 vs 20 steps:** this is not "4× faster". For severe faults both windows alarm at about the same time, which is set by how fast
  the PWM signature develops. The real gain of the 0.5 s window is *sensitivity to subtle faults* (7/7 vs 3/7 at LOE 0.8). The
  cost is more false alarms, because a short window averages over less manoeuvre noise.
- **INT8** (`export_tflite.py`; unrolled GRU, batch 1, full-integer int8 I/O, calibrated on train windows):

| | .h5 | TFLite fp32 | TFLite int8 |
|---|---|---|---|
| file size | 89 KB | 35 KB | 42 KB |
| weight data | | 19.2 KB | 8.2 KB (int8 + int32 bias) |
| latency, 1 thread, x86 | 25 ms (Keras eager call) | 10 µs | 30 µs |
| ROC-AUC on test | 0.878 | 0.878 | 0.867 |

  The model was already kilobytes, not megabytes. INT8 compresses the weights 2.3×, but the quantization parameters of the
  roughly 124 tensors outweigh that saving at 4.6k parameters. On this x86 CPU int8 is slower than fp32 (XNNPACK fp32 is very
  fast for tiny models). Whether int8 wins on the Pi or Jetson must be measured there: run `export_tflite.py` on the device, or
  time `cls_gru5_*.tflite` with `tflite_runtime`. The large, real speed-up is from TFLite itself versus calling Keras.

The first round (16 flights, 7 test episodes) is kept in `models_gru_v1/`. It gave the same picture: GRU-5 7/7 detected,
1.2-2.0 false alarms/min on 4 min of normal test flight.

## 4. More MAVLink signals: IMU and attitude rates (2026-10-08)

The recorder, the ROS bridge and the live extractor share one `TelemetryAggregator` (`fault_features.py`). It streams
`RAW_IMU` and `ATTITUDE` at 50 Hz, on top of motors, vibration and current, and closes one row every 0.1 s. Each row holds:

- the **mean and standard deviation of acceleration and gyro** within the 0.1 s step;
- the latest EKF body rates.

The model still runs at 10 Hz, but it sees high-rate content. The stutter of a failing motor raises the within-step
spread, and its torque imbalance shows up as a body-rate (yaw) bias. Layout: `RAW_FIELDS_V2` (24 values). The v2
features are the 9 v1 features plus `acc_s{x,y,z}`, `gyr_s{x,y,z}`, `gyr_{x,y,z}`, `rate_z` and `acc_z_dev`. The v1
columns are bit-identical, so v1 models keep working.

**Data:** `collect_all_v2.sh` re-flew the same 22-flight plan with the IMU recorder, into `flights_v2/` and
`fault_events_v2.csv`. In 4 flights (08, 14, 15, 16), and for a few later episodes, the one-shot `gz topic` trigger
never reached the plugin. That leaves 34 episodes instead of 43. The labels are unaffected, because they come from
the plugin's own event log, so those stretches are correctly labelled normal. Fix: `trigger()` now re-publishes until
the event log confirms the start or recovery.

**Result:** GRU-5, same flights and same split, operating point chosen on validation (`models_imu_v1/`,
`models_imu_v2/`, `latency_compare.py`):

| | val: detected in precursor / median latency | test: detected | test: within precursor | test: median latency | test: false alarms / min |
|---|---|---|---|---|---|
| v1 features (PWM + vibration) | 4 / 5.05 s | 6/8 | 4 | 4.0 s | 0.12 |
| **v2 features (+ IMU)** | **7 / 3.10 s** | **8/8** | 4 | 5.1 s | 0.49 |

Per test episode, paired: v2 catches **2 subtle (LOE 0.8) faults that v1 misses**, and v1 catches none that v2
misses. On the 6 episodes both detect, v2 is 0.25 s earlier at the median, but the result is mixed (earlier in 4,
later in 2; extremes −2.2 s and +2.0 s). ROC-AUC rises from 0.923 to 0.942, and LSTM-20 gains in the same way
(median latency 5.6 → 4.5 s). Conclusion: **more MAVLink signals mainly buy sensitivity; the latency gain is small and
inconsistent on 8 test episodes.** The deployed supervisor model is v2 (`models_imu_v2/cls_gru5_int8.tflite`:
threshold 0.453, persistence 8, chosen on val; the INT8 model matches Keras exactly at this operating point).

## 5. MAVLink → ROS 2 bridge and nodes

| node (`ros2 run motor_failure_predictor …`) | role |
|---|---|
| `mavlink_bridge` | MAVLink → `/flight_data` (Float64MultiArray, `RAW_FIELDS_V2` order, dim label "RAW_FIELDS_V2") at 10 Hz, and `/flight_status` [rel_alt, vz, armed]. Param `mavlink_url` (default `udp:127.0.0.1:14551`). |
| `lstm_predictor` | `/flight_data` → `FaultDetector` → `/motor_failure_probability` (Float32) and `/motor_failure_prediction` (Bool). Param `model_dir`. A v2 model needs the 24-value layout; a v1 model also accepts the 8- and old 10-value layouts. |
| `fault_supervisor` | `/motor_failure_prediction` + `/motor_failure_probability` + `/flight_status` → `FaultSupervisor` → MAVLink commands, and `/fault_supervisor/state`. Params `mavlink_url` (default `udp:127.0.0.1:14552`), `model_dir`, `confirm_s`, `clear_s`, `descent_speed`, `land_alt`. |

Shared logic lives in `detector.py` (`FaultDetector`, `FaultSupervisor`), so the ROS path, `telemetry_extractor.py`
and the evaluation behave identically.

**Running it in Gazebo** (GUI, four terminals: `docker exec -it c7ab345f90ed bash`):

```bash
# 1  Gazebo with the progressive-failure plugin
source /opt/ros/jazzy/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=/root/ros2_ws/install/motor_failure/lib:/root/ardupilot_gazebo/build
export GZ_SIM_RESOURCE_PATH=/root/ardupilot_gazebo/models:/root/ardupilot_gazebo/worlds
gz sim -v4 -r iris_runway.sdf
# 2  SITL with two extra MAVLink outputs (bridge, supervisor); then: mode guided / arm throttle / takeoff 10
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console \
    --out udp:127.0.0.1:14551 --out udp:127.0.0.1:14552
# 3  ROS nodes
cd /root/ros2_ws && source install/setup.bash
ros2 run motor_failure_predictor mavlink_bridge &
ros2 run motor_failure_predictor lstm_predictor --ros-args -p model_dir:=/root/ros2_ws/models_imu_v2 &
ros2 run motor_failure_predictor fault_supervisor --ros-args -p model_dir:=/root/ros2_ws/models_imu_v2 -p descent_speed:=1.0
# 4  inject the fault (ground truth goes to fault_events.csv)
gz topic -t /model/iris_with_gimbal/trigger_random_failure -m gz.msgs.Boolean -p "data: true"
```

**End-to-end check of the ROS path** (`ros_e2e_test.sh`; headless, live SITL, LOE 0.8): `/flight_data` arrives at
10.0 Hz from `mavlink_bridge`. `lstm_predictor` loads the v2 INT8 model and alarms once the fault develops.
`fault_supervisor` takes the drone NORMAL → HOLD → DESCEND → LAND from 10 m on its own (log in `ros_e2e/`).

`ros_e2e_test.sh` options (environment variables): `LOE=0.8` (final effectiveness), `MOTOR=-1` (0-3 or random),
`CUT=-1` (seconds after the 5 s ramp until the motor is cut to zero, i.e. a complete failure; -1 = never),
`HEADLESS=1` (no Gazebo window; the GUI is the default), `KEEP_OPEN=1` (leave Gazebo + SITL running; `./stop_sim.sh`
stops them). Yaw control is kept at LOE 0.8 (peak yaw rate 1.4 rad/s, within normal flight) and lost once the motor
saturates: about 2.6 rad/s at LOE 0.6, and a continuous spin after `CUT`. Examples: `LOE=0.6 ./ros_e2e_test.sh`,
`LOE=0.6 MOTOR=0 CUT=0 ./ros_e2e_test.sh` (degrade for 5 s, then cut).

## 6. Fault-response supervisor

`FaultSupervisor` (two stages, so the ~0.5 false alarms per minute cost only a pause):

- **NORMAL → HOLD** on an alarm: GUIDED, zero velocity, hold altitude.
- **HOLD → NORMAL** after 3 s without alarm. This is counted as a false alarm, and control is released.
- **HOLD → DESCEND** after 1 s of above-threshold probability: commanded descent at `descent_speed`.
- **DESCEND → LAND** below 2 m: ArduPilot's LAND mode for the touchdown.
- Option `release_yaw` sets `MOT_YAW_HEADROOM` to 0 on DESCEND.

**Evaluation** (`eval_all_supervisor.sh`, `eval_supervisor.py`, `analyze_supervisor.py`; outputs in
`supervisor_eval_on/` and `supervisor_eval_off/`):

- **Paired design:** each flight number is flown twice, supervisor ON and OFF, with the same failing motor
  (flight number mod 4) and the same manoeuvres (same seed). 4 pairs each at LOE 0.7, 0.6 and 0.8; the 1.0 → k ramp
  takes 5 s; 40-60 s of normal flight first.
- **Final design**, fixed after 2 smoke flights at 0.6: descent 1.0 m/s, no yaw release. The smoke flights were
  2 m/s (hard landing), and 1 m/s plus yaw release (hard landing). Yaw release did not help and would spin
  survivable cases.
- **Outcome criteria:** hard landing = touchdown sink > 1.5 m/s or tilt > 45°; otherwise soft. They were first
  declared as 3 m/s / 60°. Checked on supervisor-off flights only, before any supervisor run, that rule called
  13 of 14 lost vehicles "controlled", so it was tightened. Sink rate, tilt and peak touchdown vibration are reported
  for every episode.

**Results** (`supervisor_eval_results.csv`; 24 flights, every trigger confirmed by the event log):

| LOE | supervisor ON | supervisor OFF (same motor, same manoeuvres) |
|---|---|---|
| 0.6 | 4/4 hard: sink 2.2-3.2 m/s, tilt 10-20°, touchdown at 7.6-9.7 s | 4/4 hard: sink 2.2-3.1 m/s, tilt ~10°, touchdown at 8.2-9.7 s |
| 0.7 | 2 hard, 2 soft; impact vibration 73-107 | 2 hard, 2 soft (same pairs); impact 103-161 |
| 0.8 | **2 soft landings at 0.5 m/s** (impact ~28); 2 still descending under control at the 20 s cut-off | 4/4 stay airborne with the degraded motor |

Detection: first alarm 1.1-5.1 s after the fault starts. Escalation to DESCEND 1 s later in every fault episode.

**False alarms:** 8 HOLDs in 11.4 min of normal flight (0.7/min). **All 8 cleared back to NORMAL within 3.0-4.9 s,
and none escalated to a descent.** The two-stage design does what it is for.

**What the supervisor can and cannot do** (honest reading):

- **LOE 0.6/0.7: no outcome changes.** The faulty motor saturates about 3.5 s after onset. From then on the vehicle is
  thrust-limited and sinks at 1.3-3 m/s whatever is commanded; no MAVLink setpoint adds thrust the motors do not have.
  Detection at 1-4 s leaves too little time to land from 10 m before saturation.
- **At 0.6 the supervisor was slightly worse:** touchdown 0.3-0.9 s earlier in 3 of 4 pairs, tilt up to 20° vs 10°.
  Commanding a descent while thrust-limited adds to the sink.
- **At 0.7, LAND mode for the last 2 m lowered touchdown vibration in 3 of 4 pairs** (mean 91 vs 124). This is
  suggestive at n = 4, not established.
- **The benefit is for survivable faults (0.8):** the supervisor brings a degrading drone down gently (0.5 m/s)
  instead of letting it keep flying with a weakening motor that could fail further. Without it, the 0.8 drones flew on.
- **What would change the 0.6 outcome** is not a MAVLink-level supervisor but a fault-tolerant inner loop: reallocating
  thrust inside the flight controller (e.g. a sliding-mode or control-allocation controller that gives up yaw), which
  was option 3 in the design choice. Earlier detection (the precursor fires at 1-3 s) also only helps if the vehicle
  can still be landed. From 10 m that needs ~5 s of controlled descent.

Limitations: 4 pairs per severity; single altitude (10 m); no wind; the 20 s fault window cut off 2 of the 0.8
landings while still airborne.


## Runtime

- ROS 2: `ros2 run motor_failure_predictor lstm_predictor` reads the INT8 model, `preprocessing.npz` (training statistics, never
  refitted live) and `operating_point.json`. It accepts the 8 raw fields on `/flight_data`, or the previous 10-value layout.
  It publishes `/motor_failure_probability` and `/motor_failure_prediction`.
- MAVLink: `python3 telemetry_extractor.py` (same model and rules; logs raw fields, probability and alarm).

## Reproduce

```bash
cd /root/ros2_ws
./collect_all.sh && ./collect_more_normal.sh      # ~65 min; flights/, fault_events.csv
python3 label_precursors.py --flights flights/flight_[0-9]*.csv --events fault_events.csv --out flight_data_labeled.csv
python3 train_fault_models.py --val-flights flight_04 flight_10 flight_11 flight_15 flight_18 flight_20 \
    --test-flights flight_05 flight_12 flight_13 flight_16 flight_19 flight_21 --out-dir models_gru
python3 persistence_tradeoff.py                    # picks persistence on val, writes operating_point.json
python3 export_tflite.py --model models_gru/cls_gru5.h5 --kind classifier --seq 5
python3 eval_int8.py                               # INT8 threshold on val, episode metrics on test
# IMU features (section 4) and the supervisor (section 6)
./collect_all_v2.sh                                # flights_v2/, fault_events_v2.csv
python3 label_precursors.py --flights flights_v2/flight_[0-9]*.csv --events fault_events_v2.csv --out flight_data_labeled_v2.csv
for fs in v1 v2; do python3 train_fault_models.py --data flight_data_labeled_v2.csv --features $fs --models cls_gru5 cls_lstm20 \
    --out-dir models_imu_$fs --val-flights flight_04 flight_10 flight_11 flight_15 flight_18 flight_20 \
    --test-flights flight_05 flight_12 flight_13 flight_16 flight_19 flight_21; \
    MODEL_DIR=models_imu_$fs DATA=flight_data_labeled_v2.csv python3 persistence_tradeoff.py; done
python3 export_tflite.py --model models_imu_v2/cls_gru5.h5 --kind classifier --seq 5
MODEL_DIR=models_imu_v2 DATA=flight_data_labeled_v2.csv python3 eval_int8.py
python3 latency_compare.py                         # paired per-episode latency, v1 vs v2 features
./eval_all_supervisor.sh                           # 24 paired flights, ~80 min
python3 analyze_supervisor.py --runs supervisor_eval_on/flight_[0-9]*.csv --events fault_events_sup_on.csv \
    --severity supervisor_eval_on/flight_severity.csv --baseline supervisor_eval_off/flight_[0-9]*.csv \
    --baseline-events fault_events_sup_off.csv --baseline-severity supervisor_eval_off/flight_severity.csv \
    --out supervisor_eval_results.csv
```

## Limitations and next steps

1. **False alarms are the main issue.** GRU-5 raises about 1.2 per minute of normal flight on test (10 onsets in 8.4 min),
   against ≤ 0.5 on validation. Doubling the normal calibration data did not change this, so it is a property of the
   0.5 s window, not just noise. Options, in order of expected value:
   - an alarm policy that needs 2 of the next 1 s;
   - running GRU-5 for sensitivity and confirming with the 20-step model, which has 0.1-0.24 false alarms per minute;
   - adding wind and gusts to the simulation, so that "normal" covers what a real vehicle sees.
2. Small test set (11 episodes). Treat per-model differences of one or two episodes as indicative.
3. Simulation only: no wind model, no battery current, and the LOE scales torque rather than thrust. The severity mix
   (0.6 / 0.7 / 0.8) was chosen so that both saturating and survivable faults are represented.
4. The normal-only autoencoders do not separate precursors from manoeuvres here (ROC-AUC ≈ 0.5). The supervised classifier
   with precursor labels is what works. `RNN_Training.py` and `train_lstm.py` are kept unchanged as the previous baselines;
   `train_fault_models.py` supersedes both.
