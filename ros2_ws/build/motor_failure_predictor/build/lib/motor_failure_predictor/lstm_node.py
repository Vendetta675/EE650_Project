"""ROS 2 node: early motor-fault detection with the INT8 TFLite GRU classifier (0.5 s sliding window).

Subscribes /flight_data (std_msgs/Float64MultiArray, 10 Hz) and runs detector.FaultDetector (model, feature set,
scaling, threshold and persistence from model_dir). Accepted /flight_data layouts:
  25 values = fault_features.RAW_FIELDS_V3 (published by mavlink_bridge; required for a v3 / battery model)
  24 values = fault_features.RAW_FIELDS_V2 (earlier bridge; enough for a v2 / IMU model)
   8 values = fault_features.RAW_FIELDS: m1-m4, vib_x/y/z, current           (v1 model only)
  10 values = previous layout: vib_x/y/z, m1-m4, 3 EKF variances, current = 0 (v1 model only)
Publishes /motor_failure_probability (Float32) and /motor_failure_prediction (Bool, one per window).
Parameters: model_dir, threshold / persistence (override operating_point.json; -1 = use the file).
"""
import os

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float64MultiArray

from motor_failure_predictor.detector import FaultDetector
from motor_failure_predictor.fault_features import RAW_FIELDS, RAW_FIELDS_V2, RAW_FIELDS_V3


class LSTMPredictorNode(Node):
    def __init__(self):
        super().__init__('lstm_predictor_node')
        self.declare_parameter('model_dir', '/root/ros2_ws/models_gru')
        self.declare_parameter('threshold', -1.0)
        self.declare_parameter('persistence', -1)
        d = self.get_parameter('model_dir').value
        thr, per = self.get_parameter('threshold').value, self.get_parameter('persistence').value
        if not os.path.exists(os.path.join(d, 'operating_point.json')):
            self.get_logger().error(f'No model in {d}. Exiting.')
            return
        self.det = FaultDetector(d, None if thr < 0 else thr, None if per < 1 else per)
        self.get_logger().info(f'Loaded {self.det.model_path} ({self.det.feature_set} features, '
                               f'{self.det.inp["dtype"].__name__} input), threshold {self.det.threshold:.3f}, '
                               f'persistence {self.det.persistence}.')
        self.subscription = self.create_subscription(Float64MultiArray, '/flight_data', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(Bool, '/motor_failure_prediction', 10)
        self.score_publisher_ = self.create_publisher(Float32, '/motor_failure_probability', 10)

    def to_raw(self, data):
        n = len(data)
        if n == len(RAW_FIELDS_V3):  # the layouts are nested: v3 = v2 + voltage, v2 = v1 + IMU
            return list(data[:len(self.det.raw_fields)])
        if self.det.feature_set == 'v3':
            return None  # battery voltage required
        if n == len(RAW_FIELDS_V2):
            return list(data) if self.det.feature_set == 'v2' else list(data[:len(RAW_FIELDS)])
        if self.det.feature_set == 'v2':
            return None  # IMU fields required
        if n == len(RAW_FIELDS):
            return list(data)
        if n == 10:  # previous layout: vib xyz, m1-m4, ekf variances
            vx, vy, vz, m1, m2, m3, m4 = data[:7]
            return [m1, m2, m3, m4, vx, vy, vz, 0.0]
        return None

    def listener_callback(self, msg):
        raw = self.to_raw(msg.data)
        if raw is None:
            self.get_logger().warn(f'Unsupported /flight_data length {len(msg.data)} for a '
                                   f'{self.det.feature_set} model', throttle_duration_sec=5.0)
            return
        prob, alarm = self.det.step(np.asarray(raw, np.float32))
        if prob is None:
            return
        self.score_publisher_.publish(Float32(data=prob))
        self.publisher_.publish(Bool(data=bool(alarm)))
        if alarm:
            self.get_logger().warn(f'MOTOR FAULT DEVELOPING! Probability: {prob:.2f}', throttle_duration_sec=1.0)


def main(args=None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    rclpy.init(args=args)
    node = LSTMPredictorNode()
    try:
        if hasattr(node, 'det'):
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
