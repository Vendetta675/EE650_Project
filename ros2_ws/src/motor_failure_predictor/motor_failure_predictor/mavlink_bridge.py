"""ROS 2 node: MAVLink telemetry -> /flight_data (std_msgs/Float64MultiArray) at 10 Hz.

Uses the shared TelemetryAggregator (fault_features.py), so /flight_data rows are exactly the rows the models were
trained on: fault_features.RAW_FIELDS_V3 order (motors, vibration, current, IMU mean/std per 0.1 s, attitude rates,
battery voltage); the layout's dim label is "RAW_FIELDS_V3" (the first 24 values are the RAW_FIELDS_V2 layout). Also publishes /flight_status [rel_alt (m), vz_down (m/s), armed].
Parameters: mavlink_url (default udp:127.0.0.1:14551, e.g. MAVProxy --out udp:127.0.0.1:14551), rate_hz (10).
"""
import threading
import time

import rclpy
from pymavlink import mavutil
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from motor_failure_predictor.fault_features import RAW_FIELDS_V3, TelemetryAggregator


class MavlinkBridge(Node):
    def __init__(self):
        super().__init__('mavlink_bridge')
        self.declare_parameter('mavlink_url', 'udp:127.0.0.1:14551')
        self.declare_parameter('rate_hz', 10.0)
        url = self.get_parameter('mavlink_url').value
        self.pub = self.create_publisher(Float64MultiArray, '/flight_data', 10)
        self.status_pub = self.create_publisher(Float64MultiArray, '/flight_status', 10)
        self.agg = TelemetryAggregator()
        self.status = [float('nan'), float('nan'), 0.0]
        self.lock = threading.Lock()
        self.get_logger().info(f'Connecting to {url} ...')
        self.conn = mavutil.mavlink_connection(url, source_system=250)
        self.conn.wait_heartbeat()
        self.agg.request_streams(self.conn, extra=[('GLOBAL_POSITION_INT', 10)])
        self.get_logger().info(f'Connected (system {self.conn.target_system}); publishing /flight_data '
                               f'({len(RAW_FIELDS_V3)} values, RAW_FIELDS_V3 order).')
        threading.Thread(target=self.reader, daemon=True).start()
        self.create_timer(1.0 / self.get_parameter('rate_hz').value, self.publish)

    def reader(self):
        while rclpy.ok():
            msg = self.conn.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue
            with self.lock:
                self.agg.update(msg)
                t = msg.get_type()
                if t == 'GLOBAL_POSITION_INT':
                    self.status[0], self.status[1] = msg.relative_alt / 1000.0, msg.vz / 100.0
                elif t == 'HEARTBEAT' and msg.get_srcComponent() == 1:
                    self.status[2] = float(bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED))

    def publish(self):
        with self.lock:
            row = self.agg.row()
            ready = self.agg.ready()
            status = list(self.status)
        if not ready:
            return
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(label='RAW_FIELDS_V3', size=len(RAW_FIELDS_V3), stride=1)]
        msg.data = [float(row[k]) for k in RAW_FIELDS_V3]
        self.pub.publish(msg)
        st = Float64MultiArray()
        st.data = status
        self.status_pub.publish(st)


def main(args=None):
    rclpy.init(args=args)
    node = MavlinkBridge()
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
