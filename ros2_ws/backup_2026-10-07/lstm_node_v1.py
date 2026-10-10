"""ROS 2 node: early motor-fault detection with the INT8 TFLite GRU classifier (0.5 s window).

Subscribes /flight_data (std_msgs/Float64MultiArray) at 10 Hz, either
  8 values  = fault_features.RAW_FIELDS: m1, m2, m3, m4, vib_x, vib_y, vib_z, current, or
  10 values = the previous layout: vib_x, vib_y, vib_z, m1, m2, m3, m4, ekf_vel_var, ekf_pos_var, ekf_comp_var
              (EKF variances ignored; current taken as 0).
Publishes /motor_failure_probability (Float32) and /motor_failure_prediction (Bool). The alarm needs the
probability above the threshold for `persistence` consecutive windows (same rule used to pick the threshold).

Preprocessing comes from the training run (preprocessing.npz: feature subset, median / IQR), never refitted on
live data. Parameters: model_path, preprocessing_path, operating_point_path (threshold + persistence chosen on validation
flights, see models_gru/operating_point.json), threshold / persistence (overrides).
The interpreter is tflite_runtime / ai_edge_litert when installed (edge devices), else tf.lite.
"""
import json
import os
from collections import deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float64MultiArray

from motor_failure_predictor.fault_features import RAW_FIELDS, SEQ_LEN, apply_scaler, compute_features

MODELS = '/root/ros2_ws/models_gru'


def make_interpreter(path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=path, num_threads=1)
    it.allocate_tensors()
    return it


class LSTMPredictorNode(Node):
    def __init__(self):
        super().__init__('lstm_predictor_node')
        self.declare_parameter('model_path', os.path.join(MODELS, 'cls_gru5_int8.tflite'))
        self.declare_parameter('preprocessing_path', os.path.join(MODELS, 'preprocessing.npz'))
        self.declare_parameter('operating_point_path', os.path.join(MODELS, 'operating_point.json'))
        self.declare_parameter('threshold', -1.0)
        self.declare_parameter('persistence', -1)
        model_path = self.get_parameter('model_path').value
        if not os.path.exists(model_path):
            self.get_logger().error(f'Model not found at {model_path}. Exiting.')
            return

        pre = np.load(self.get_parameter('preprocessing_path').value)
        self.cols = pre['feature_index']
        self.median = pre['median'].astype(np.float32)
        self.scale = pre['scale'].astype(np.float32)
        op = json.load(open(self.get_parameter('operating_point_path').value))
        thr = float(self.get_parameter('threshold').value)
        self.threshold = thr if thr >= 0 else float(op['threshold'])
        per = int(self.get_parameter('persistence').value)
        self.persistence = per if per > 0 else int(op['persistence'])

        self.interpreter = make_interpreter(model_path)
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.buffer = deque(maxlen=SEQ_LEN)
        self.run = 0
        self.get_logger().info(f'Loaded {model_path} ({self.inp["dtype"].__name__} input), window {SEQ_LEN} '
                               f'steps, threshold {self.threshold:.3f}, persistence {self.persistence}.')

        self.subscription = self.create_subscription(Float64MultiArray, '/flight_data', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(Bool, '/motor_failure_prediction', 10)
        self.score_publisher_ = self.create_publisher(Float32, '/motor_failure_probability', 10)
        self.get_logger().info('Waiting for data on /flight_data...')

    def to_raw(self, data):
        if len(data) == len(RAW_FIELDS):
            return np.asarray(data, np.float32)
        if len(data) == 10:  # previous layout: vib xyz, m1-m4, ekf variances
            vx, vy, vz, m1, m2, m3, m4 = data[:7]
            return np.array([m1, m2, m3, m4, vx, vy, vz, 0.0], np.float32)
        return None

    def predict(self, window):
        x = window[None].astype(np.float32)
        if self.inp['dtype'] == np.int8:
            s, z = self.inp['quantization']
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        self.interpreter.set_tensor(self.inp['index'], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out['index'])
        if self.out['dtype'] == np.int8:
            s, z = self.out['quantization']
            y = (y.astype(np.float32) - z) * s
        return float(y.reshape(-1)[0])

    def listener_callback(self, msg):
        raw = self.to_raw(msg.data)
        if raw is None:
            self.get_logger().warn(f'Expected {len(RAW_FIELDS)} or 10 values, got {len(msg.data)}')
            return
        feats = apply_scaler(compute_features(raw)[self.cols], self.median, self.scale)
        self.buffer.append(feats)
        if len(self.buffer) < SEQ_LEN:
            return
        prob = self.predict(np.stack(self.buffer))
        self.run = self.run + 1 if prob > self.threshold else 0
        is_failure = self.run >= self.persistence

        prob_msg = Float32()
        prob_msg.data = prob
        self.score_publisher_.publish(prob_msg)
        bool_msg = Bool()
        bool_msg.data = is_failure
        self.publisher_.publish(bool_msg)
        if is_failure:
            self.get_logger().warn(f'MOTOR FAULT DEVELOPING! Probability: {prob:.2f}')
        else:
            self.get_logger().debug(f'System normal. Fault probability: {prob:.2f}')


def main(args=None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    rclpy.init(args=args)
    node = LSTMPredictorNode()
    try:
        if hasattr(node, 'interpreter'):
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
