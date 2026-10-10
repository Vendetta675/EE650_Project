"""ROS 2 node: fault-response supervisor. Subscribes /motor_failure_prediction (Bool), /motor_failure_probability
(Float32) and /flight_status (rel_alt, vz, armed from mavlink_bridge); commands the vehicle over its own MAVLink link
with detector.FaultSupervisor (NORMAL -> HOLD -> DESCEND -> LAND, HOLD -> NORMAL when an alarm clears).
Publishes /fault_supervisor/state (String). Parameters: mavlink_url (default udp:127.0.0.1:14552, e.g. MAVProxy
--out udp:127.0.0.1:14552), model_dir (threshold), confirm_s, clear_s, descent_speed, land_alt.
"""
import json
import os

import rclpy
from pymavlink import mavutil
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float64MultiArray, String

from motor_failure_predictor.detector import FaultSupervisor


class SupervisorNode(Node):
    def __init__(self):
        super().__init__('fault_supervisor')
        for k, v in (('mavlink_url', 'udp:127.0.0.1:14552'), ('model_dir', '/root/ros2_ws/models_gru'),
                     ('confirm_s', 1.0), ('clear_s', 3.0), ('descent_speed', 2.0), ('land_alt', 2.0)):
            self.declare_parameter(k, v)
        p = lambda k: self.get_parameter(k).value  # noqa: E731
        op = json.load(open(os.path.join(p('model_dir'), 'operating_point.json')))
        conn = mavutil.mavlink_connection(p('mavlink_url'), source_system=251)
        conn.wait_heartbeat()
        self.sup = FaultSupervisor(conn, float(op['threshold']), p('confirm_s'), p('clear_s'), p('descent_speed'),
                                   p('land_alt'), log=lambda s: self.get_logger().warn(s))
        self.prob, self.rel_alt = None, None
        self.state_pub = self.create_publisher(String, '/fault_supervisor/state', 10)
        self.create_subscription(Float32, '/motor_failure_probability', self.on_prob, 10)
        self.create_subscription(Float64MultiArray, '/flight_status', self.on_status, 10)
        self.create_subscription(Bool, '/motor_failure_prediction', self.on_alarm, 10)
        self.get_logger().info(f'Supervisor armed on {p("mavlink_url")} (threshold {self.sup.threshold:.3f}).')

    def on_prob(self, msg):
        self.prob = float(msg.data)

    def on_status(self, msg):
        self.rel_alt = float(msg.data[0])

    def on_alarm(self, msg):  # the detector publishes one Bool per window (10 Hz): drive the state machine
        state = self.sup.update(self.prob, bool(msg.data), self.rel_alt)
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
