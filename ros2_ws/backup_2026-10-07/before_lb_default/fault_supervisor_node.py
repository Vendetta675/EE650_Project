"""ROS 2 node: fault-response supervisor. Subscribes /motor_failure_prediction (Bool), /motor_failure_probability
(Float32) and /flight_status (rel_alt, vz, armed from mavlink_bridge); commands the vehicle over its own MAVLink link
with detector.FaultSupervisor (NORMAL -> HOLD -> DESCEND -> LAND, HOLD -> NORMAL when an alarm clears).
Publishes /fault_supervisor/state (String). Parameters: mavlink_url (default udp:127.0.0.1:14552, e.g. MAVProxy
--out udp:127.0.0.1:14552), model_dir (threshold), confirm_s, clear_s, descent_speed, land_alt,
ftc (false; true = an alarm arms ArduPilot's fault-tolerant controller, CC_TYPE 3, which takes over onboard when a
rotor is lost), ftc_continue (true: the mission continues after the alarm; false: HOLD -> DESCEND -> LAND with the FTC
armed), ftc_agree_s, ftc_require_detector, ftc_pwm_high, ftc_pwm_low (fallback hand-off: failed motor identified from
the /flight_data motor outputs), battery_aware (true), low_soc (0.7), pack_r (0.048 ohm), cells (3), low_batt_land_alt
(1.0 m), low_batt_slow_alt (3.0 m; 0 = constant speed) and low_batt_slow_speed (0.7 m/s): below low_soc an alarm arms the
FTC and lands at once, through the takeover, slowing down below low_batt_slow_alt; low_batt_cut_alt (0 = off) cuts the
motors below that altitude instead of LAND (detector.FaultSupervisor).
"""
import json
import os

import rclpy
from pymavlink import mavutil
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float64MultiArray, String

from motor_failure_predictor.detector import FaultSupervisor, MotorLossIdentifier


class SupervisorNode(Node):
    def __init__(self):
        super().__init__('fault_supervisor')
        for k, v in (('mavlink_url', 'udp:127.0.0.1:14552'), ('model_dir', '/root/ros2_ws/models_gru'),
                     ('confirm_s', 1.0), ('clear_s', 3.0), ('descent_speed', 2.0), ('land_alt', 2.0),
                     ('ftc', False), ('ftc_continue', True), ('ftc_agree_s', 0.5), ('ftc_require_detector', True),
                     ('ftc_pwm_high', 1900.0), ('ftc_pwm_low', 1250.0),
                     ('battery_aware', True), ('low_soc', 0.7), ('pack_r', 0.048), ('cells', 3),
                     ('low_batt_land_alt', 1.0), ('low_batt_slow_alt', 3.0), ('low_batt_slow_speed', 0.7),
                     ('low_batt_cut_alt', 0.0)):
            self.declare_parameter(k, v)
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        op = json.load(open(os.path.join(p('model_dir'), 'operating_point.json')))
        conn = mavutil.mavlink_connection(p('mavlink_url'), source_system=251)
        conn.wait_heartbeat()
        self.sup = FaultSupervisor(conn, float(op['threshold']), p('confirm_s'), p('clear_s'), p('descent_speed'),
                                   p('land_alt'), log=lambda s: self.get_logger().warn(s), ftc=p('ftc'),
                                   ftc_agree_s=p('ftc_agree_s'), ftc_require_detector=p('ftc_require_detector'),
                                   ftc_continue=p('ftc_continue'), battery_aware=p('battery_aware'),
                                   low_soc=p('low_soc'), pack_r=p('pack_r'), cells=p('cells'),
                                   low_batt_land_alt=p('low_batt_land_alt'), low_batt_slow_alt=p('low_batt_slow_alt'),
                                   low_batt_slow_speed=p('low_batt_slow_speed'), low_batt_cut_alt=p('low_batt_cut_alt'))
        self.ident = MotorLossIdentifier(p('ftc_pwm_high'), p('ftc_pwm_low'))
        self.prob, self.rel_alt, self.failed_motor = None, None, None
        self.state_pub = self.create_publisher(String, '/fault_supervisor/state', 10)
        self.create_subscription(Float32, '/motor_failure_probability', self.on_prob, 10)
        self.create_subscription(Float64MultiArray, '/flight_status', self.on_status, 10)
        self.create_subscription(Bool, '/motor_failure_prediction', self.on_alarm, 10)
        self.create_subscription(Float64MultiArray, '/flight_data', self.on_flight_data, 10)
        self.get_logger().info(f'Supervisor armed on {p("mavlink_url")} (threshold {self.sup.threshold:.3f}'
                               f'{", FTC hand-off on" if p("ftc") else ""}).')

    def on_prob(self, msg):
        self.prob = float(msg.data)

    def on_flight_data(self, msg):  # RAW_FIELDS_V3 order: m1..m4 first, current at 7, voltage at 24
        self.failed_motor = self.ident.step(list(msg.data[:4]))
        if len(msg.data) >= 25:
            self.sup.update_battery(float(msg.data[24]), float(msg.data[7]))

    def on_status(self, msg):
        self.rel_alt = float(msg.data[0])

    def on_alarm(self, msg):  # the detector publishes one Bool per window (10 Hz): drive the state machine
        state = self.sup.update(self.prob, bool(msg.data), self.rel_alt, self.failed_motor)
        self.state_pub.publish(String(data=state))


def main(args=None):
    rclpy.init(args=args)
    node = SupervisorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
