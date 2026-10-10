import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import random
import math

class RandomFlightDataPublisher(Node):
    def __init__(self):
        super().__init__('random_flight_data_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/flight_data', 10)
        # Publish 10 times a second (every 0.1s)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Random Flight Data Publisher started...")
        self.counter = 0

    def timer_callback(self):
        self.counter += 1
        
        # Simulate normal data
        vib_x = random.gauss(0.1, 0.02)
        vib_y = random.gauss(0.02, 0.005)
        vib_z = random.gauss(0.01, 0.002)
        m1 = random.gauss(1560, 5)
        m2 = random.gauss(1560, 5)
        m3 = random.gauss(1560, 5)
        m4 = random.gauss(1560, 5)
        ekf_vel_var = 0.0
        ekf_pos_var = 0.0
        ekf_comp_var = 0.0
        
        # Simulate random anomalies (e.g. vibration spike, motor desync) every ~50 ticks (5 seconds)
        if self.counter % 50 < 5:
            # During the anomaly window, alter values significantly
            vib_x += 0.5
            vib_z += 0.2
            m3 -= 200 # Motor 3 failure simulation
            ekf_vel_var += 0.1
            self.get_logger().warn("Simulating random anomaly injection...")
            
        msg = Float64MultiArray()
        msg.data = [vib_x, vib_y, vib_z, m1, m2, m3, m4, ekf_vel_var, ekf_pos_var, ekf_comp_var]
        
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RandomFlightDataPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
