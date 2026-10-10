import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class MotorFaultDetector(Node):
    def __init__(self):
        super().__init__('motor_fault_detector')
        # Subscribe to the bridged joint states
        self.subscription = self.create_subscription(
            JointState,
            '/world/iris_runway/model/iris_with_gimbal/joint_state',
            self.listener_callback,
            10)
        self.fault_detected = False

    def listener_callback(self, msg):
        try:
            # Find the index of the rotor we want to monitor
            index = msg.name.index('iris_with_standoffs::rotor_0_joint')
            velocity = msg.velocity[index]
            
            # If the rotor speed drops dangerously low while flying
            if abs(velocity) < 5.0 and not self.fault_detected:
                self.get_logger().fatal('FAULT DETECTED: Rotor 0 velocity has dropped below threshold!')
                self.fault_detected = True
                self.trigger_failsafe()
                
        except ValueError:
            pass # Joint not found in this message

    def trigger_failsafe(self):
        self.get_logger().info('Deploying virtual parachute / Cutting power to other motors...')
        # Publish your parachute or failsafe commands here

def main(args=None):
    rclpy.init(args=args)
    node = MotorFaultDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
