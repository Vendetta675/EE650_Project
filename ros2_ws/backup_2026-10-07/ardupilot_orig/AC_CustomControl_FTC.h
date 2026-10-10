#pragma once

#include "AC_CustomControl_config.h"

#if AP_COPTER_CUSTOMCONTROL_FTC_ENABLED

#include "AC_CustomControl_Backend.h"
#include <Filter/LowPassFilter2p.h>

// Fault-tolerant attitude controller for a quad with one completely failed rotor (CC_TYPE = 3).
// Decoupled degraded strategy of Gu, Wang and Lu (IEEE Access 13, 2025): yaw is given up, only the thrust direction
// is controlled (reduced attitude), with a sliding-mode disturbance observer; the three remaining rotors are allocated
// by AP_MotorsMatrix's single-motor-failure mode. With CC3_FAIL_MOT = 0 the backend passes the main controller through.
class AC_CustomControl_FTC : public AC_CustomControl_Backend {
public:
    AC_CustomControl_FTC(AC_CustomControl& frontend, AP_AHRS_View*& ahrs, AC_AttitudeControl*& att_control, AP_MotorsMulticopter*& motors, float dt);

    Vector3f update(void) override;
    void reset(void) override;

    // user settable parameters
    static const struct AP_Param::GroupInfo var_info[];

protected:
    AP_Int8  _fail_motor;   // failed motor number 1-4, 0 = none
    AP_Float _kp;           // tilt stiffness, 1/s^2
    AP_Float _kd;           // tilt-rate damping, 1/s
    AP_Float _ixx, _iyy, _izz;   // body inertia, kg m^2
    AP_Float _mass;         // vehicle mass, kg
    AP_Float _arm_x, _arm_y;     // rotor offsets from the centre of mass, m
    AP_Int8  _smo_enable;   // 1 = use the disturbance observer
    AP_Float _smo_e1, _smo_e2, _smo_e3, _smo_h;  // observer gains (Eq. 5): eps1, eps2, eps3 and exponent h > 1
    AP_Float _dist_max;     // limit of the disturbance estimate, N m
    AP_Float _lag;          // actuator lag compensated by rotating the torque command by -yaw_rate * lag, s
    AP_Float _yaw_rate_tgt; // spin rate above which the opposite rotor brakes, rad/s (0 = no yaw brake)
    AP_Float _yaw_kr;       // opposite-rotor thrust per rad/s of excess spin
    AP_Float _yaw_tmax;     // maximum opposite-rotor thrust (0..1)
    AP_Float _rot_inertia;  // rotor + propeller inertia about its axis, kg m^2 (0 = no rotor gyroscopic feedforward)
    AP_Float _rot_wmax;     // rotor speed at full output, rad/s
    AP_Int8  _mode;         // 0 = paper reduced-attitude PD law, 1 = angular-momentum steering once spinning
    AP_Float _steer_kp;     // steering gain: precession rate per radian of spin-axis error, 1/s
    AP_Float _steer_kd;     // steering damping on the spin-axis rate
    AP_Float _spin_min;     // spin rate above which mode 1 steers the angular momentum, rad/s
    AP_Float _hk;           // mode 2: primary-axis convergence rate, 1/s
    AP_Float _wk;           // mode 2: rate-loop gain, 1/s
    AP_Float _indi_flt;     // mode 2: INDI filter cutoff, Hz
    AP_Float _hbar_tc;      // mode 2: adaptation time constant of the primary axis, s (0 = fixed at body z)
    AP_Float _hbar_max;     // mode 2: maximum primary-axis tilt from body z, rad
    AP_Float _act_tc;       // mode 2: rotor (actuator) time constant used by INDI, s
    AP_Int8  _auto;         // 1: when armed with FAIL_MOT 0, switch to three-rotor control on an onboard-detected loss
    int8_t   _auto_motor = 0;  // motor (1-4) found by the onboard detection, 0 = none (cleared only on the ground)
    AP_Float _eff;          // control-effectiveness scale on the modelled torque per unit roll / pitch input
    Vector3f _tq_act;       // modelled torque actually acting (realised torque through the actuator lag)

    float _dt;
    Vector3f _mu;           // observer pseudo angular velocity, rad/s
    Vector3f _dist_acc;     // observer disturbance estimate as angular acceleration, rad/s^2
    Vector3f _torque_cmd;   // last commanded torque, N m
    uint16_t _log_counter;
    bool _announced;
    // mode 2 state
    LowPassFilter2pVector3f _gyro_filt;
    LowPassFilter2pVector3f _tq_filt;
    Vector3f _wf_prev;
    Vector3f _hbar;         // primary axis in the body frame (target of the reduced attitude)
    bool _indi_init;
};

#endif  // AP_COPTER_CUSTOMCONTROL_FTC_ENABLED
