import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float32, Bool
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

class LSTMPredictorNode(Node):
    def __init__(self):
        super().__init__('lstm_predictor_node')
        
        # Load the trained model
        model_path = '/root/ros2_ws/motor_failure_lstm.h5'
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.get_logger().info("Successfully loaded LSTM model.")
        else:
            self.get_logger().error(f"Model not found at {model_path}. Exiting.")
            return

        # Fit the scaler on the original data so it matches the training
        data_path = '/root/ros2_ws/flight_data_live.csv'
        self.scaler = StandardScaler()
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            df = df.fillna(0)
            feature_cols = ['vib_x', 'vib_y', 'vib_z', 'm1', 'm2', 'm3', 'm4', 'ekf_vel_var', 'ekf_pos_var', 'ekf_comp_var']
            data = df[feature_cols].values
            self.scaler.fit(data)
            self.get_logger().info("Fitted scaler on original dataset.")
        else:
            self.get_logger().warn("Original dataset not found. Scaler will not work correctly without fitting.")

        # Sequence buffer for the LSTM
        self.sequence_length = 10
        self.sequence_buffer = []

        # Subscription and Publisher
        # Expecting a Float64MultiArray with the 10 features
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/flight_data',
            self.listener_callback,
            10
        )
        self.publisher_ = self.create_publisher(Bool, '/motor_failure_prediction', 10)
        self.score_publisher_ = self.create_publisher(Float32, '/motor_failure_probability', 10)
        self.get_logger().info("LSTM Predictor Node initialized. Waiting for data on /flight_data...")

    def listener_callback(self, msg):
        if len(msg.data) != 10:
            self.get_logger().warn(f"Expected 10 features, got {len(msg.data)}")
            return
            
        features = np.array(msg.data).reshape(1, -1)
        scaled_features = self.scaler.transform(features)
        
        self.sequence_buffer.append(scaled_features[0])
        
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)
            
        if len(self.sequence_buffer) == self.sequence_length:
            # Prepare input for LSTM
            X = np.array([self.sequence_buffer]) # Shape (1, 10, 10)
            
            # Predict
            prediction_prob = self.model.predict(X, verbose=0)[0][0]
            is_failure = bool(prediction_prob > 0.5)
            
            # Publish
            prob_msg = Float32()
            prob_msg.data = float(prediction_prob)
            self.score_publisher_.publish(prob_msg)
            
            bool_msg = Bool()
            bool_msg.data = is_failure
            self.publisher_.publish(bool_msg)
            
            if is_failure:
                self.get_logger().warn(f"MOTOR FAILURE PREDICTED! Probability: {prediction_prob:.2f}")
            else:
                self.get_logger().info(f"System normal. Failure probability: {prediction_prob:.2f}")

def main(args=None):
    # Set TF log level to suppress info/warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    rclpy.init(args=args)
    node = LSTMPredictorNode()
    if hasattr(node, 'model'):
        rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
