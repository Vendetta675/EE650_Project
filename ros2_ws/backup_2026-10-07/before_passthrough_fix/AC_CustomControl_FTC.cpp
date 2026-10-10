#include "AC_CustomControl_config.h"

#if AP_COPTER_CUSTOMCONTROL_FTC_ENABLED

#include "AC_CustomControl_FTC.h"

#include <GCS_MAVLink/GCS.h>
#include <AP_Logger/AP_Logger.h>

// table of user settable parameters
const AP_Param::GroupInfo AC_CustomControl_FTC::var_info[] = {
    // @Param: FAIL_MOT
    // @DisplayName: Failed motor
    // @Description: Motor (1-4) that has completely failed. 0 passes the main attitude controller through.
    // @Range: 0 4
    // @User: Advanced
    AP_GROUPINFO("FAIL_MOT", 1, AC_CustomControl_FTC, _fail_motor, 0),

    // @Param: KP
    // @DisplayName: Tilt stiffness
    // @Description: Reduced-attitude proportional gain (angular acceleration per radian of thrust-axis error)
    // @Units: 1/s/s
    // @User: Advanced
    AP_GROUPINFO("KP", 2, AC_CustomControl_FTC, _kp, 64.0f),

    // @Param: KD
    // @DisplayName: Tilt-rate damping
    // @Description: Damping gain on the roll / pitch body rates
    // @Units: 1/s
    // @User: Advanced
    AP_GROUPINFO("KD", 3, AC_CustomControl_FTC, _kd, 11.0f),

    // @Param: IXX
    // @DisplayName: Inertia x
    // @Units: kg.m.m
    // @User: Advanced
    AP_GROUPINFO("IXX", 4, AC_CustomControl_FTC, _ixx, 0.008f),

    // @Param: IYY
    // @DisplayName: Inertia y
    // @Units: kg.m.m
    // @User: Advanced
    AP_GROUPINFO("IYY", 5, AC_CustomControl_FTC, _iyy, 0.015f),

    // @Param: IZZ
    // @DisplayName: Inertia z
    // @Units: kg.m.m
    // @User: Advanced
    AP_GROUPINFO("IZZ", 6, AC_CustomControl_FTC, _izz, 0.017f),

    // @Param: MASS
    // @DisplayName: Vehicle mass
    // @Units: kg
    // @User: Advanced
    AP_GROUPINFO("MASS", 7, AC_CustomControl_FTC, _mass, 1.8f),

    // @Param: ARM_X
    // @DisplayName: Rotor x offset
    // @Units: m
    // @User: Advanced
    AP_GROUPINFO("ARM_X", 8, AC_CustomControl_FTC, _arm_x, 0.13f),

    // @Param: ARM_Y
    // @DisplayName: Rotor y offset
    // @Units: m
    // @User: Advanced
    AP_GROUPINFO("ARM_Y", 9, AC_CustomControl_FTC, _arm_y, 0.21f),

    // @Param: SMO_EN
    // @DisplayName: Disturbance observer enable
    // @Values: 0:Disabled,1:Enabled
    // @User: Advanced
    AP_GROUPINFO("SMO_EN", 10, AC_CustomControl_FTC, _smo_enable, 1),

    // @Param: SMO_E1
    // @DisplayName: Observer gain eps1
    // @User: Advanced
    AP_GROUPINFO("SMO_E1", 11, AC_CustomControl_FTC, _smo_e1, 15.0f),

    // @Param: SMO_E2
    // @DisplayName: Observer gain eps2
    // @User: Advanced
    AP_GROUPINFO("SMO_E2", 12, AC_CustomControl_FTC, _smo_e2, 5.0f),

    // @Param: SMO_E3
    // @DisplayName: Observer gain eps3
    // @User: Advanced
    AP_GROUPINFO("SMO_E3", 13, AC_CustomControl_FTC, _smo_e3, 50.0f),

    // @Param: SMO_H
    // @DisplayName: Observer exponent h
    // @Description: Exponent of the fixed-time term, must be greater than 1
    // @User: Advanced
    AP_GROUPINFO("SMO_H", 14, AC_CustomControl_FTC, _smo_h, 1.5f),

    // @Param: DIST_MAX
    // @DisplayName: Disturbance estimate limit
    // @Units: N.m
    // @User: Advanced
    AP_GROUPINFO("DIST_MAX", 15, AC_CustomControl_FTC, _dist_max, 0.3f),

    // @Param: LAG
    // @DisplayName: Actuator lag compensation
    // @Description: Rotor thrust lag; the torque command is rotated by -yaw_rate * LAG about body z so that it acts in the intended direction of the spinning frame. 0 disables.
    // @Units: s
    // @User: Advanced
    AP_GROUPINFO("LAG", 16, AC_CustomControl_FTC, _lag, 0.0f),

    // @Param: YAW_RTGT
    // @DisplayName: Yaw brake spin rate
    // @Description: Spin rate above which the rotor opposite the failed one is given thrust to oppose the spin. 0 disables the yaw brake.
    // @Units: rad/s
    // @User: Advanced
    AP_GROUPINFO("YAW_RTGT", 17, AC_CustomControl_FTC, _yaw_rate_tgt, 8.0f),

    // @Param: YAW_KR
    // @DisplayName: Yaw brake gain
    // @Description: Opposite-rotor thrust (0..1) per rad/s of spin above YAW_RTGT
    // @User: Advanced
    AP_GROUPINFO("YAW_KR", 18, AC_CustomControl_FTC, _yaw_kr, 0.05f),

    // @Param: YAW_TMAX
    // @DisplayName: Yaw brake maximum thrust
    // @Description: Maximum opposite-rotor thrust (0..1)
    // @User: Advanced
    AP_GROUPINFO("YAW_TMAX", 19, AC_CustomControl_FTC, _yaw_tmax, 0.25f),

    // @Param: ROT_I
    // @DisplayName: Rotor inertia
    // @Description: Rotor and propeller inertia about the rotor axis, for the rotor gyroscopic feedforward (the two loaded rotors spin the same way after a failure, so their angular momentum no longer cancels). 0 disables.
    // @Units: kg.m.m
    // @User: Advanced
    AP_GROUPINFO("ROT_I", 20, AC_CustomControl_FTC, _rot_inertia, 0.0f),

    // @Param: ROT_WMAX
    // @DisplayName: Rotor speed at full output
    // @Units: rad/s
    // @User: Advanced
    AP_GROUPINFO("ROT_WMAX", 21, AC_CustomControl_FTC, _rot_wmax, 838.0f),

    // @Param: MODE
    // @DisplayName: FTC control law
    // @Description: 0: reduced-attitude PD law of the paper (thrust axis tracks the target). 1: angular-momentum steering once spinning faster than SPIN_MIN (the spin axis, fixed in the body, is precessed onto the target thrust direction; the thrust axis cones around it).
    // @Values: 0:Paper reduced attitude,1:Angular-momentum steering,2:Primary-axis INDI
    // @User: Advanced
    AP_GROUPINFO("MODE", 22, AC_CustomControl_FTC, _mode, 0),

    // @Param: SKP
    // @DisplayName: Steering gain
    // @Units: 1/s
    // @User: Advanced
    AP_GROUPINFO("SKP", 23, AC_CustomControl_FTC, _steer_kp, 3.0f),

    // @Param: SKD
    // @DisplayName: Steering damping
    // @User: Advanced
    AP_GROUPINFO("SKD", 24, AC_CustomControl_FTC, _steer_kd, 1.0f),

    // @Param: SPIN_MIN
    // @DisplayName: Spin rate for angular-momentum steering
    // @Units: rad/s
    // @User: Advanced
    AP_GROUPINFO("SPIN_MIN", 25, AC_CustomControl_FTC, _spin_min, 3.0f),

    // @Param: HK
    // @DisplayName: Mode 2 primary-axis gain
    // @Units: 1/s
    // @User: Advanced
    AP_GROUPINFO("HK", 26, AC_CustomControl_FTC, _hk, 4.0f),

    // @Param: WK
    // @DisplayName: Mode 2 rate-loop gain
    // @Units: 1/s
    // @User: Advanced
    AP_GROUPINFO("WK", 27, AC_CustomControl_FTC, _wk, 20.0f),

    // @Param: INDI_F
    // @DisplayName: Mode 2 INDI filter cutoff
    // @Units: Hz
    // @User: Advanced
    AP_GROUPINFO("INDI_F", 28, AC_CustomControl_FTC, _indi_flt, 25.0f),

    // @Param: HB_TC
    // @DisplayName: Mode 2 primary-axis adaptation time constant
    // @Description: The target primary axis (in the body frame) slowly follows the measured one, so the vehicle settles in its coning equilibrium instead of fighting it. 0 keeps it at body z.
    // @Units: s
    // @User: Advanced
    AP_GROUPINFO("HB_TC", 29, AC_CustomControl_FTC, _hbar_tc, 2.0f),

    // @Param: HB_MAX
    // @DisplayName: Mode 2 maximum primary-axis tilt
    // @Units: rad
    // @User: Advanced
    AP_GROUPINFO("HB_MAX", 30, AC_CustomControl_FTC, _hbar_max, 0.6f),

    // @Param: ACT_TC
    // @DisplayName: Mode 2 actuator time constant
    // @Description: First-order rotor response time used to model the torque actually acting for INDI
    // @Units: s
    // @User: Advanced
    AP_GROUPINFO("ACT_TC", 31, AC_CustomControl_FTC, _act_tc, 0.04f),

    // @Param: EFF
    // @DisplayName: Control effectiveness scale
    // @Description: Scale on the modelled roll / pitch torque per unit mixer input (thrust curve mismatch between the autopilot model and the vehicle)
    // @User: Advanced
    AP_GROUPINFO("EFF", 32, AC_CustomControl_FTC, _eff, 1.0f),

    // @Param: AUTO
    // @DisplayName: Onboard rotor-loss detection
    // @Description: With the custom controller switched on and FAIL_MOT 0 the controller passes the main controller through ("armed", e.g. by a fault predictor). With AUTO 1 it then switches to three-rotor control by itself as soon as the motor mixer shows a complete rotor loss (one motor at full output, its diagonal partner at minimum, for 30 ms).
    // @Values: 0:Disabled,1:Enabled
    // @User: Advanced
    AP_GROUPINFO("AUTO", 33, AC_CustomControl_FTC, _auto, 1),

    AP_GROUPEND
};

AC_CustomControl_FTC::AC_CustomControl_FTC(AC_CustomControl& frontend, AP_AHRS_View*& ahrs, AC_AttitudeControl*& att_control, AP_MotorsMulticopter*& motors, float dt) :
    AC_CustomControl_Backend(frontend, ahrs, att_control, motors, dt),
    _dt(dt)
{
    AP_Param::setup_object_defaults(this, var_info);
}

// return roll, pitch, yaw mixer inputs
Vector3f AC_CustomControl_FTC::update(void)
{
    int8_t failed = _fail_motor;
    switch (_motors->get_spool_state()) {
        case AP_Motors::SpoolState::SHUT_DOWN:
        case AP_Motors::SpoolState::GROUND_IDLE:
            reset();
            _auto_motor = 0;
            return Vector3f{_motors->get_roll(), _motors->get_pitch(), _motors->get_yaw()};
        case AP_Motors::SpoolState::THROTTLE_UNLIMITED:
        case AP_Motors::SpoolState::SPOOLING_UP:
        case AP_Motors::SpoolState::SPOOLING_DOWN:
            break;
    }
    if ((failed < 1 || failed > 4) && _auto == 1) {
        // armed: watch the mixer for a complete rotor loss and latch it until landed
        if (_auto_motor == 0) {
            const int8_t lost = _motors->get_complete_loss_motor();
            if (lost >= 0 && lost < 4) {
                _auto_motor = lost + 1;
                GCS_SEND_TEXT(MAV_SEVERITY_CRITICAL, "FTC: motor %d lost (onboard detection)", _auto_motor);
            }
        }
        failed = _auto_motor;
    }
    if (failed < 1 || failed > 4) {
        // no failed motor declared: behave like the main controller
        return Vector3f{_motors->get_roll(), _motors->get_pitch(), _motors->get_yaw()};
    }
    // three-rotor allocation with yaw released (refreshed every loop, expires if this controller stops); above the spin
    // target the opposite rotor gets thrust in proportion to the excess spin (yaw brake)
    const float spin = fabsf(_ahrs->get_gyro_latest().z);
    float opposite_thrust = -1.0f;
    if (is_positive(_yaw_rate_tgt) && spin > _yaw_rate_tgt) {
        opposite_thrust = MIN(_yaw_kr * (spin - _yaw_rate_tgt), _yaw_tmax);
    }
    _motors->request_single_motor_failure_mode(failed - 1, opposite_thrust);
    if (!_announced) {
        GCS_SEND_TEXT(MAV_SEVERITY_WARNING, "FTC: motor %d failed, yaw released, reduced attitude control", failed);
        _announced = true;
    }

    // ---- reduced attitude error (Eq. 10-12): rotation taking the body thrust axis onto the target thrust axis
    Quaternion q_body, q_target = _att_control->get_attitude_target_quat();
    _ahrs->get_quat_body_to_ned(q_body);
    Matrix3f R, Rt;
    q_body.rotation_matrix(R);      // body -> NED
    q_target.rotation_matrix(Rt);
    const Vector3f zb = R.colz();   // body z axis (thrust is along -z) in NED
    const Vector3f zd = Rt.colz();
    const Vector3f axis_ned = zb % zd;
    const float sin_t = axis_ned.length();
    const float tilt_err = atan2f(sin_t, zb * zd);
    Vector3f err;                   // tilt error vector in the body frame (no yaw component)
    if (sin_t > 1.0e-6f) {
        err = R.mul_transpose(axis_ned) * (tilt_err / sin_t);
    } else if (tilt_err > 1.0f) {   // exactly opposite: any horizontal axis
        err = Vector3f(tilt_err, 0.0f, 0.0f);
    }
    err.z = 0.0f;

    // ---- control torque (Eq. 13 / 16): feedback linearisation incl. gyroscopic term, minus disturbance estimate
    const Vector3f w = _ahrs->get_gyro_latest();
    const Vector3f I(_ixx, _iyy, _izz);
    const Vector3f Iw(I.x * w.x, I.y * w.y, I.z * w.z);
    Vector3f gyro_torque = w % Iw;
    // rotor gyroscopic feedforward: the two loaded rotors (diagonal pair) spin the same way, each carrying about twice
    // its hover share; their net angular momentum H along body z couples roll and pitch (torque w x H). The body spins
    // opposite to the loaded rotors, so H has the sign opposite to the measured yaw rate.
    if (is_positive(_rot_inertia) && fabsf(w.z) > 1.0f) {
        const float t_load = constrain_float(2.0f * _motors->get_throttle(), 0.0f, 1.0f);
        const float h_z = -copysignf(2.0f * _rot_inertia * _rot_wmax * sqrtf(t_load), w.z);
        gyro_torque += w % Vector3f(0.0f, 0.0f, h_z);
    }
    Vector3f alpha(_kp * err.x - _kd * w.x, _kp * err.y - _kd * w.y, 0.0f);
    Vector3f dist_torque(I.x * _dist_acc.x, I.y * _dist_acc.y, 0.0f);
    Vector3f torque(I.x * alpha.x + gyro_torque.x - dist_torque.x,
                    I.y * alpha.y + gyro_torque.y - dist_torque.y,
                    0.0f);

    // mode 1, angular-momentum steering: once spinning, the vehicle is gyroscopically stiff and a torque precesses the
    // angular momentum L (dL/dt = tau) instead of producing an angular acceleration. Steer the spin axis n = L/|L|,
    // which is fixed in the body (the thrust axis cones around it), onto the target thrust direction:
    // tau = s |L| (kp (d - (d.n) n) - kd n_dot), with s the sign that makes n point along body +z.
    if (_mode == 1 && fabsf(w.z) > _spin_min) {
        float h_z = 0.0f;
        if (is_positive(_rot_inertia)) {
            const float t_load = constrain_float(2.0f * _motors->get_throttle(), 0.0f, 1.0f);
            h_z = -copysignf(2.0f * _rot_inertia * _rot_wmax * sqrtf(t_load), w.z);
        }
        const Vector3f L_b(I.x * w.x, I.y * w.y, I.z * w.z + h_z);
        const float L_mag = L_b.length();
        if (L_mag > 1.0e-3f) {
            const float sgn = is_negative(L_b.z) ? -1.0f : 1.0f;
            const Vector3f n_b = L_b * (sgn / L_mag);       // spin axis in the body frame, close to +z
            const Vector3f n_e = R * n_b;
            const Vector3f d_perp = zd - n_e * (zd * n_e);  // where the spin axis should move
            const Vector3f n_dot_e = R * (w % n_b);         // current motion of the spin axis
            const Vector3f tau_e = (d_perp * _steer_kp - n_dot_e * _steer_kd) * (sgn * L_mag);
            torque = R.mul_transpose(tau_e) - dist_torque;
            torque.z = 0.0f;
            // log the spin-axis error instead of the thrust-axis error in this mode
            err = R.mul_transpose(n_e % zd);
        }
    }

    // ---- torque -> mixer input: one unit of roll / pitch input changes each motor's thrust fraction by its +-0.5
    // factor, i.e. a torque of 2 * arm * T_max, with T_max (thrust at full output) from the learned hover throttle
    const float thr_hover = MAX(_motors->get_throttle_hover(), 0.05f);
    const float t_max = _mass * GRAVITY_MSS / (4.0f * thr_hover);
    const float eff = constrain_float(_eff.get(), 0.05f, 5.0f);
    const float tq_roll_max = 2.0f * _arm_y * t_max * eff;
    const float tq_pitch_max = 2.0f * _arm_x * t_max * eff;
    // mode 2, primary-axis reduced attitude + INDI (after Mueller and D'Andrea 2016, Sun et al. 2021):
    // h = desired thrust direction in the body frame, h_dot = -w x h, so the roll / pitch rates that move h towards the
    // primary axis hbar follow exactly, including the spin r: p = (h2_dot + r h1) / h3, q = (r h2 - h1_dot) / h3.
    // The rates are tracked with incremental nonlinear dynamic inversion: torque = realised torque (filtered) +
    // I (alpha_des - filtered angular acceleration), which cancels unmodelled torques (rotor gyroscopics, aerodynamics,
    // CoM offset) from measurement. hbar adapts slowly to the coning equilibrium of the spinning vehicle.
    if (_mode == 2) {
        if (!_indi_init) {
            const float fs = 1.0f / _dt;
            _gyro_filt.set_cutoff_frequency(fs, _indi_flt);
            _tq_filt.set_cutoff_frequency(fs, _indi_flt);
            _gyro_filt.reset(w);
            _tq_filt.reset(Vector3f());
            _tq_act.zero();
            _wf_prev = w;
            _hbar = Vector3f(0.0f, 0.0f, 1.0f);
            _indi_init = true;
        }
        const Vector3f h = R.mul_transpose(zd);
        if (is_positive(_hbar_tc)) {
            const float a = _dt / (_hbar_tc + _dt);
            _hbar.x += a * (h.x - _hbar.x);
            _hbar.y += a * (h.y - _hbar.y);
            const float hxy = norm(_hbar.x, _hbar.y), lim = sinf(_hbar_max);
            if (hxy > lim) {
                _hbar.x *= lim / hxy;
                _hbar.y *= lim / hxy;
            }
            _hbar.z = safe_sqrt(1.0f - sq(_hbar.x) - sq(_hbar.y));
        }
        const float h3 = MAX(h.z, 0.3f);
        const float e1 = h.x - _hbar.x, e2 = h.y - _hbar.y;
        const float hd1 = -_hk * e1, hd2 = -_hk * e2;
        const float p_des = (hd2 + w.z * h.x) / h3;
        const float q_des = (w.z * h.y - hd1) / h3;

        // realised torque of the previous mixer run (N m), filtered like the gyro
        float rr = 0.0f, pp = 0.0f;
        Vector3f tq_real;
        if (_motors->get_single_motor_failure_realized(rr, pp)) {
            tq_real = Vector3f(rr * tq_roll_max, pp * tq_pitch_max, 0.0f);
        }
        // rotors take ~ACT_TC to change speed: the torque acting now is the realised command through that lag
        _tq_act += (tq_real - _tq_act) * (_dt / (MAX(_act_tc.get(), 0.0f) + _dt));
        const Vector3f tq_f = _tq_filt.apply(_tq_act);
        const Vector3f wf = _gyro_filt.apply(w);
        const Vector3f wdot_f = (wf - _wf_prev) / _dt;
        _wf_prev = wf;
        const float ax = _wk * (p_des - w.x), ay = _wk * (q_des - w.y);
        torque = Vector3f(tq_f.x + I.x * (ax - wdot_f.x), tq_f.y + I.y * (ay - wdot_f.y), 0.0f);
#if HAL_LOGGING_ENABLED
        if (_log_counter == 7) {
            // @LoggerMessage: CCI
            // @Description: FTC mode 2 (primary-axis INDI) internals
            // @Field: TimeUS: Time since system startup
            // @Field: H1: desired thrust direction in body x
            // @Field: H2: desired thrust direction in body y
            // @Field: PD: desired roll rate
            // @Field: QD: desired pitch rate
            // @Field: P: measured roll rate
            // @Field: Q: measured pitch rate
            // @Field: WDX: filtered roll acceleration
            // @Field: WDY: filtered pitch acceleration
            // @Field: TFX: filtered acting torque x
            // @Field: TFY: filtered acting torque y
            // @Field: TCX: commanded torque x
            // @Field: TCY: commanded torque y
            AP::logger().WriteStreaming("CCI", "TimeUS,H1,H2,PD,QD,P,Q,WDX,WDY,TFX,TFY,TCX,TCY", "Qffffffffffff",
                                        AP_HAL::micros64(), h.x, h.y, p_des, q_des, w.x, w.y, wdot_f.x, wdot_f.y,
                                        tq_f.x, tq_f.y, torque.x, torque.y);
        }
#endif
        err = Vector3f(e1, e2, 0.0f);
        dist_torque = Vector3f(_hbar.x, _hbar.y, 0.0f);  // logged in DX / DY in this mode
    }

    // actuator-lag compensation for the spinning frame: the torque takes effect ~lag later, when the body has turned by
    // yaw_rate * lag, so rotate the command by the opposite angle about body z
    if (is_positive(_lag)) {
        const float ang = -w.z * _lag;
        const float ca = cosf(ang), sa = sinf(ang);
        torque = Vector3f(ca * torque.x - sa * torque.y, sa * torque.x + ca * torque.y, 0.0f);
    }
    const float u_roll = constrain_float(torque.x / tq_roll_max, -1.0f, 1.0f);
    const float u_pitch = constrain_float(torque.y / tq_pitch_max, -1.0f, 1.0f);
    _torque_cmd = Vector3f(u_roll * tq_roll_max, u_pitch * tq_pitch_max, 0.0f);

    // ---- sliding-mode disturbance observer (Eq. 5), pseudo angular velocity mu, roll / pitch channels
    if (_smo_enable) {
        const Vector3f e = _mu - w;
        const Vector3f exy(e.x, e.y, 0.0f);
        const float en = exy.length();
        const float eps = 1.0e-3f;
        const Vector3f s_half = exy / sqrtf(en + eps);                 // e / |e|^(1/2)
        const Vector3f s_h = exy * powf(en + eps, _smo_h - 1.0f);       // e |e|^(h-1)
        const Vector3f s_sign = exy / (en + eps);                       // e / |e|
        const Vector3f model_acc((_torque_cmd.x - gyro_torque.x) / I.x, (_torque_cmd.y - gyro_torque.y) / I.y, 0.0f);
        _mu += (model_acc + _dist_acc - s_half * _smo_e1 - s_h * _smo_e2) * _dt;
        _mu.z = w.z;
        _dist_acc -= s_sign * (_smo_e3 * _dt);
        const float dmax_x = _dist_max / I.x, dmax_y = _dist_max / I.y;
        _dist_acc.x = constrain_float(_dist_acc.x, -dmax_x, dmax_x);
        _dist_acc.y = constrain_float(_dist_acc.y, -dmax_y, dmax_y);
        _dist_acc.z = 0.0f;
    } else {
        _dist_acc.zero();
    }

#if HAL_LOGGING_ENABLED
    if (++_log_counter >= 8) {  // 50 Hz at a 400 Hz loop
        _log_counter = 0;
        // @LoggerMessage: CCF
        // @Description: FTC custom controller (single rotor failure)
        // @Field: TimeUS: Time since system startup
        // @Field: Mot: failed motor
        // @Field: Err: thrust-axis error
        // @Field: ErX: tilt error x
        // @Field: ErY: tilt error y
        // @Field: Wz: yaw rate
        // @Field: DX: disturbance estimate x
        // @Field: DY: disturbance estimate y
        // @Field: UR: roll input
        // @Field: UP: pitch input
        // @Field: Tmx: thrust at full output per motor
        AP::logger().WriteStreaming("CCF", "TimeUS,Mot,Err,ErX,ErY,Wz,DX,DY,UR,UP,Tmx", "QBfffffffff",
                                    AP_HAL::micros64(), (uint8_t)failed, tilt_err, err.x, err.y, w.z,
                                    dist_torque.x, dist_torque.y, u_roll, u_pitch, t_max);
    }
#endif

    return Vector3f{u_roll, u_pitch, 0.0f};
}

// reset observer and announcement (called when switched on and on the ground)
void AC_CustomControl_FTC::reset(void)
{
    _mu = _ahrs->get_gyro_latest();
    _dist_acc.zero();
    _torque_cmd.zero();
    _announced = false;
    _indi_init = false;
}

#endif  // AP_COPTER_CUSTOMCONTROL_FTC_ENABLED
