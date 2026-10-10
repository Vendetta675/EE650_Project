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

## 7. Fault-tolerant control after a complete single-rotor failure (ArduPilot, 2026-10-08)

Based on Gu, Wang, Lu, *Fault-Tolerant Control of a Quadrotor UAV With Single Rotor Failure in Dense Obstacle
Environments*, IEEE Access 2025. The controller runs **inside ArduPilot** as a custom-controller backend, so ArduPilot's
own position control, waypoint logic and failsafes keep working on top of it.

### Result

Complete, instant loss of one rotor at 10 m. FTC is switched on at the same moment. Test airframe: 1.2 kg (see "Thrust").

| run | outcome |
|---|---|
| motor 1, **no FTC** | flips, hits the ground in **3.2 s** |
| motor 1, FTC, hold | **hovers 25 s**: altitude 9.99–10.04 m, position within 0.20 m; then lands at 0.5 m/s, upright, 2 cm from the hover point |
| motors 2 / 3 / 4, FTC, hold | same: 25 s hover and landing. Motor 3 has one 2.9 m drift in the first 3 s, then returns to within 0.1 m |
| motor 1, FTC, 20 m square at 1 m/s | **flies all four legs** at 9.99–10.00 m, then lands |

The vehicle spins about its thrust axis at about 9 rad/s and cones with 12–20° of tilt. That is the physically
unavoidable equilibrium: with three rotors, yaw torque cannot be balanced while also producing lift.

### Thrust: the stock Iris cannot do this

At full throttle the stock Gazebo Iris (1.98 kg) has a thrust-to-weight ratio of about 1.5 (measured: full-throttle
climbs, `ftc_dev/lift_*`). After losing a rotor, the two remaining diagonal rotors must carry almost all the weight,
which needs a ratio of about 2 or more. With FTC the stock Iris stays upright and holds position, but sinks to the
ground in 5.5 s with a motor at 100%.

The test airframe `sim_models_ftc/` is the same Iris with `base_link` reduced from 1.5 kg to 0.72 kg (1.2 kg in total,
thrust-to-weight about 2.3). The motors are unchanged. Raising motor speed instead made Gazebo's rotor velocity loop
numerically unstable.

### What was implemented

All the ArduPilot changes are in `ftc_dev/ardupilot_ftc.patch` (`git diff` in `/root/ardupilot`, 8 files).

- **3-rotor mixer** (`AP_MotorsMatrix`, requested every loop by the controller and active for 200 ms per request):
  - allocates collective thrust, roll and pitch exactly over the three healthy rotors, and leaves yaw free;
  - if the request is infeasible, scales roll and pitch down before giving up collective;
  - reports the torque it actually realised after clipping.
- **Yaw brake** (relaxed hover, Mueller & D'Andrea): the rotor opposite the failed one spins the other way. Above 8 rad/s
  of spin it gets a small thrust (0.05 per rad/s, at most 0.25), and the other two rotors are solved by weighted least
  squares. This caps the spin at about 9 rad/s. Without it, the spin grows linearly in Gazebo (there is no body
  rotational drag) and control is lost after about 8–10 s.
- **FTC controller** `AC_CustomControl_FTC` (`CC_TYPE 3`, parameters `CC3_*`), following the paper:
  - *Reduced attitude, yaw given up* (Eqs. 10–13, 16): the thrust axis tracks the thrust direction that ArduPilot's
    position controller asks for. Error = body z × target z, as an angle-scaled vector in the body frame; z component 0.
  - *Torque law*: Kp·e − Kd·ω (Kp 64, Kd 11), plus gyroscopic compensation ω × Iω, scaled into mixer roll / pitch with
    the arm lengths and the per-rotor hover thrust.
  - *Fixed-time sliding-mode disturbance observer* (Eq. 5) on roll and pitch, with the estimate limited to ±0.3 N m.
  - A GCS warning on activation, and a `CCF` dataflash record at 50 Hz (errors, spin, estimates, outputs).
- **Activation**: `CC3_FAIL_MOT` = failed motor (1–4), then `MAV_CMD_DO_AUX_FUNCTION(109, 2)` (custom-controller switch).
  This is the hand-off point for the detector (sections 3–6). In these tests the failed motor is given directly.

### The decisive finding: a softer position loop

With the paper's law alone the vehicle held 10 m perfectly for about 7 s, then a slow oscillation (about 0.5 Hz in the
earth frame) grew until it flipped. That frequency is the position loop's, not the attitude loop's: ArduPilot's
default position and velocity gains keep re-tilting the target against the coning motion. With softer gains it holds
indefinitely. `ftc_dev/ftc.parm` sets them, together with slow, low-lean navigation, because there is little thrust
margin left:

```
CC_TYPE 3   CC_AXIS_MASK 7   CC3_FAIL_MOT 0   CC3_MASS 1.2
PSC_NE_POS_P 0.5   PSC_NE_VEL_P 0.8   PSC_NE_VEL_I 0.2   PSC_NE_VEL_D 0
PSC_ANGLE_MAX 8    WP_SPD 1.0   WP_ACC 0.5
```

At the default waypoint speed, the first 20 m leg asked for a 51° lean and the vehicle crashed. At 1 m/s it flies the
whole square. These settings also apply in normal flight, which is safe but slower. A production version would switch
to them only when FTC engages.

### Variants tried that did not help (kept as parameters, off by default)

| parameter | idea | result |
|---|---|---|
| `CC3_LAG` | rotate the torque command to compensate rotor lag in the spinning frame | touchdown 6–10 s |
| `CC3_ROT_I` | rotor gyroscopic feed-forward ω × H_rotor | touchdown 9.3 s |
| `CC3_MODE 1` | steer the angular momentum (precession) instead of the thrust axis | touchdown 3.8 s |
| `CC3_MODE 2` | primary-axis reduced attitude (Sun et al. 2021) + INDI with the realised mixer torque | 3–4 s; the roll rate loop saturates against the rotor gyroscopic coupling |
| `CC3_DIST_MAX 1` | larger disturbance-estimate limit | worse |

They are left in the code so the history can be reproduced. Mode 2 is the published method for real vehicles, and
probably needs a rotor model identified from the Gazebo Iris to work here.

### Not implemented from the paper

- Artificial-potential-field obstacle avoidance (Eqs. 17–20).
- The tracking differentiator for the reference (Eqs. 21–22).

ArduPilot's position controller and waypoint navigation take their place. The paper's results are MATLAB-only.

### Run it

```bash
cd /root/ros2_ws
./ftc_gui_test.sh                      # Gazebo window; motor 1 fails at 10 m, FTC holds 25 s, then lands
MOTOR=3 PLAN=square ./ftc_gui_test.sh  # other motor, fly a 20 m square at 1 m/s after the failure
FTC=0 ./ftc_gui_test.sh                # baseline without FTC: crashes in ~3 s
# options: HEADLESS=1 (no window), KEEP_OPEN=1; results in ftc_gui/ (run.csv, test.log, logs/*.BIN with CCF records)
```

Development harness: `ftc_dev/ftc_sim.sh <run> --motor N --plan hold|square|land [--no-ftc] [--param NAME=VAL ...]`
(headless), plus `ftc_dev/summarize.py <run>`.

Rebuild after changing the code: `cd /root/ardupilot && ./waf copter -j10`. The original files are in
`backup_2026-10-07/ardupilot_orig/`.

### Limitations

- Needs a vehicle with a thrust-to-weight ratio of about 2 or more. The stock Iris doesn't have it.
- The position loop must be soft and navigation slow (1 m/s here). Heavier wind was not tested.
- In the tests above, FTC is switched on by hand. Section 8 covers the automatic hand-off from the detector.
- Simulation only. The yaw brake and gains were tuned on this airframe, and a real vehicle needs identified rotor
  dynamics.

## 8. Automatic hand-off: detector → fault-tolerant controller (2026-10-09)

### How it works

A 10 Hz MAVLink link is too slow to switch on the FTC after a rotor is lost. Hand-off delay after an instant, complete
loss, with the failed motor known (oracle, `ftc_dev/AD*`, `BD*`):

| delay | 1.2 kg airframe (`sim_models_ftc`) | 1.0 kg airframe (`sim_models_ftc10`, `CC3_MASS 1.0`) |
|---|---|---|
| 0 s | survives (`AD0`) | survives |
| 0.3 s | crashes in 4.5 s (`AD0.3`) | survives (`BD0.3`) |
| 0.6 s | crashes in 8.7 s (`AD0.6`) | survives (`BD0.6`) |
| 1.0 s | crashes in 3.5 s (`AD1.0`) | – |

The work is therefore split in two:

- **Detector (ROS, early, slow):** the supervisor *arms* the FTC on the first alarm. It switches the custom controller on
  with `CC3_FAIL_MOT 0`, which is a pure pass-through, so a false alarm costs nothing. Flight continues normally.
- **ArduPilot (onboard, at loop rate):** with `CC3_AUTO 1` (the default), the armed controller watches the mixer. When
  one motor is pinned at full output and its diagonal partner at minimum for 30 ms, it switches to three-rotor control
  by itself and announces `FTC: motor N lost (onboard detection)`.
- **Fallback, a loss without a prior alarm:** `MotorLossIdentifier` finds the same pattern in `/flight_data` (2 rows,
  ≥ 1900 / ≤ 1250 µs). If the detector was hot in the last 0.5 s, the supervisor sets `CC3_FAIL_MOT` and switches
  the controller on itself. This path is about 0.3 s late, which only the 1.0 kg airframe survives.

Supervisor states with `ftc:=true`: NORMAL → **ARMED** (alarm) → **FTC** (onboard confirmation or fallback). From HOLD
or DESCEND (`ftc_continue:=false`), FTC stops the descent. Node parameters: `ftc`, `ftc_continue`, `ftc_agree_s`,
`ftc_require_detector`, `ftc_pwm_high`, `ftc_pwm_low`. The ArduPilot changes, onboard detection included, are in
`ftc_dev/ardupilot_ftc.patch` (regenerated; the earlier version is in `backup_2026-10-07/`).

### Result (`ros_ftc_e2e_test.sh`, full ROS chain, 1.0 kg airframe)

Fault profile: the motor ramps to LOE 0.6 over 5 s, then is cut to zero 2 s later, at 10 m.

| run | detector | after the cut |
|---|---|---|
| motor 1, supervisor with FTC hand-off (`ros_ftc_e2e_m1`) | alarm 5.6 s after onset (1.5 s before the cut) → ARMED | onboard switch; supervisor sees it 0.3 s after the cut. Holds at 9.98–10.01 m (median position error 0.06 m) until the hold ends, then lands. Transient at the cut: 64° tilt, 0.6 m height loss, up to 5 m drift |
| motor 3, same (`ros_ftc_e2e_m3`) | alarm → ARMED | same; holds and lands (max sink 1.0 m/s) |
| motor 1, supervisor without FTC (`ros_ftc_e2e_noftc_m1`, previous design) | alarm → HOLD → DESCEND | flips (87°) and hits the ground **2.6 s** after the cut at 9.8 m/s |

| motor 1, **AUTO mission** (`PLAN=mission`, `ros_ftc_e2e_mission_m1`): 20 m square, failure 3 s into leg 2, cut mid-leg | alarm → ARMED → FTC (onboard) | finishes the mission on three rotors: WP2 / WP3 / WP4 reached 6 / 31 / 54 s after the cut. Cross-track error from +3 s on: median 0.05 m, max 0.96 m, altitude 10.2 m. Cut transient 60° tilt, 0.6 m height loss. Lands on the last waypoint (0.0 m) |

Before the cut, arming the FTC (pass-through) did not change the flight: tilt 0.4°, position within 0.07 m.

### Limitations

- On the 1.2 kg airframe, onboard detection also crashed (`ftc_dev/AUTO1`). The ~30 ms detection, plus the time the
  mixer takes to reach the pinned pattern, is already too late there. The hand-off needs the larger thrust margin.
- The detector was trained on the stock Iris. On the light airframe the degrading motor does not saturate (peak 1447 µs),
  and the alarm came only after the ramp (5.6 s), not within it. Retraining on this airframe should move it earlier.
- 2 FTC runs + 1 baseline; one fault profile; no wind. The fallback path (no prior alarm) was not tested through ROS.

Run: `./ros_ftc_e2e_test.sh` (Gazebo window), with options `MOTOR=1-4`, `LOE=0.6`, `CUT=2`, `FTC=0` (previous
supervisor), `HEADLESS=1`, `KEEP_OPEN=1`.

## 9. Battery and propulsion power model (2026-10-09)

Before this, battery current was 0 A in every row and the voltage was constant: the Gazebo plugin never sent a
battery state, so SITL used a fixed `SIM_BATT_VOLTAGE`. The thrust margin also never changed during a flight.

### Model

The model is in `ardupilot_gazebo/include/BatteryModel.hh`. It is wired into `ArduPilotPlugin.cc` and switched on by a
`<battery>` element in the model SDF; without that element the plugin behaves exactly as before. The patch is
`ftc_dev/ardupilot_gazebo_battery.patch`, and the originals are in `backup_2026-10-07/ardupilot_gazebo_orig/`.

- **Pack:** LiPo open-circuit voltage curve by state of charge (SoC), series resistance R0, and one RC polarisation
  branch (R1, τ = 30 s). SoC is coulomb counted.
- **Load:** per rotor, thrust T = k_T·ω², with k_T = 2.0e-5 N/(rad/s)². This was measured: the same value on both
  test airframes from hover PWM. Electrical power follows momentum theory,
  P = T^1.5 / √(2ρA) / (figure of merit 0.6 × drive efficiency 0.75), plus 0.5 A of avionics. The terminal voltage
  solves the constant-power load.
- **Effect on the vehicle:**
  - ESC output is a duty cycle of the battery voltage, so the rotor speed target scales with V / V_ref.
  - The torque available follows the DC motor line τ ≤ τ_stall·(V/V_ref − ω/ω₀), so top speed (and maximum thrust)
    falls with voltage.
  - V_ref is 4.0 V/cell (a fresh pack under hover load), and ω₀ = 775 rad/s, so the top loaded speed of 620 rad/s
    is 80 % of no-load. These are calibrated so that a full battery reproduces the earlier, validated vehicle.
- **To ArduPilot:** voltage and current go out in the JSON packet (`battery.voltage`, `battery.current`; SITL already
  accepts these fields) and are read by the analog battery monitor (`BATT_MONITOR 4` from `copter.parm`).
- **Ground truth:** the plugin publishes `[V, I, SoC, mAh, I_motor x4, ω x4]` at 10 Hz on
  `/model/iris_with_gimbal/battery_state` (`gz.msgs.Float_V`).
- **Packs:**
  - stock Iris (`sim_models`, `ardupilot_gazebo/models`): 3S 5.1 Ah, 6 + 4 mΩ per cell;
  - light FTC airframes (`sim_models_ftc`, `sim_models_ftc10`): 3S 2.2 Ah, 10 + 6 mΩ per cell, `BATT_CAPACITY 2200`
    in `ftc_dev/ftc_e2e.parm`.

### Stock Iris (`ftc_dev/battery_sim.sh`, `battery_summary.py`)

Each run is a 20 s hover, then a 1.5 s full-throttle climb.

| SoC at takeoff | hover voltage / current (SITL) | hover PWM | peak thrust-to-weight (plugin) |
|---|---|---|---|
| no battery model | – | 1567 | ~1.5 |
| 1.0 | 12.0 V / 21.5 A | 1573 | 1.42 at the first calibration; 1.57 with no torque limit |
| 0.5 | 10.9 V / 23.7 A | 1621 | 1.23 |
| 0.2 | 10.6 V / 24.2 A | 1632 | 1.17 |
| 0.05 | 10.4 V at rest | – | ArduPilot refuses to arm: below `BATT_LOW_VOLT` 10.5 V |

The stock Iris rows were measured with the first motor calibration (ω₀ = 754); the FTC rows below use the final one.

A drained pack shows the real signatures:
- hover current rises (the same power at a lower voltage);
- hover PWM rises by about 60 µs, because the controller compensates through its integrator (`MOT_BAT_VOLT_MAX` is 0,
  so there is no explicit compensation);
- thrust margin falls;
- ArduPilot's low-battery arming check now works.

ArduPilot's own "remaining %" is coulomb counted from an assumed full pack, so it reads 94 % on a 20 % pack. This is
the same as on a real vehicle without a voltage-based reset.

### Effect on the fault-tolerant control (`SOC=… ./ros_ftc_e2e_test.sh`)

Setup: 1.0 kg airframe, full detector → supervisor → onboard FTC chain, loss of effectiveness to 0.6 over 5 s, then
the motor is cut, motor 1. The hand-off worked in every run (NORMAL → ARMED → FTC).

| run | after the cut |
|---|---|
| battery model off (`ros_ftc_e2e_nobat`) | survives: holds 9.2–10 m, lands |
| SoC 1.0 (`ros_ftc_e2e_v2bat1.0`) | survives: holds 9.2–10 m, lands |
| SoC 0.5 (`ros_ftc_e2e_v2bat0.5`) | crashes after 5.3 s (flips, 13 m/s) |
| SoC 0.2 (`ros_ftc_e2e_v2bat0.2`) | crashes after 7.4 s; 3.49 V/cell under load, top rotor speed 553 rad/s |

**The finding:** three-rotor flight on this airframe survives only on a near-full battery. The first motor
calibration took about 3 % off the top rotor speed even at full charge, and that alone made every FTC run crash
(`ros_ftc_e2e_bat*`, 3 of 3 plus a repeat). That is how thin the margin is. In practice, FTC needs either more thrust
margin (a lighter vehicle or stronger motors) or a fault response that depends on battery state, e.g. land immediately
rather than continue the mission when the pack is below about 70 %.

### Limitations and next steps

- k_T and the power coefficients are fitted to the Gazebo Iris, not to a real motor or prop. Replace them with a
  thrust-stand table for a real vehicle.
- No temperature effects (cold packs have higher R0) and no ageing over cycles. R0 is a parameter: use 2–3× for
  an aged pack.
- **Not yet done:** recollect training data with randomised SoC (0.3–1.0) and R0, retrain the detector, and recheck
  false alarms. Low-battery flight raises PWM on all motors, which the current detector has never seen.
- `stop_sim.sh` now also stops `gz topic -e` recorders. Before this fix they kept writing into a previous run's
  folder, so the battery figures in the `ros_ftc_e2e_bat*` folders are mixed; the flight results there are unaffected.
