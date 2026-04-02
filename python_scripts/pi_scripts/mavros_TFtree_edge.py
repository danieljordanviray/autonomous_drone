#!/usr/bin/env python3
"""
SAR Drone TF Bridge (Real World)

Publishes the transform tree for the drone using Pixhawk odometry:

  odom -> base_link       (dynamic, from MAVROS odometry)
  base_link -> camera_link (static, D455 mount position + 15deg pitch)

The RealSense driver publishes:
  camera_link -> camera_color_optical_frame (optical rotation)
  camera_link -> camera_depth_optical_frame (optical rotation)

map -> odom is published by RTAB-Map (SLAM loop closure).

Full TF chain:
  map -> odom -> base_link -> camera_link -> camera_color_optical_frame
                                          -> camera_depth_optical_frame

Mount measurements (from center of Pixhawk to center of D455 lens):
  Forward:  18.5 cm
  Below:     7.5 cm
  Left:      1.5 cm
  Pitch:    15 deg downward

Run with:
  python3 mavros_tf_bridge.py
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped


class SarDroneTfBridge(Node):
    def __init__(self):
        super().__init__('sar_drone_tf_bridge')

        # Dynamic TF broadcaster (odom -> base_link)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Static TF broadcaster (base_link -> camera_link)
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # Publish the static camera mount transform once
        self.publish_static_transforms()

        # Subscribe to MAVROS odometry for the dynamic drone position
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            Odometry,
            '/mavros/local_position/odom',
            self.odom_callback,
            qos
        )

        self.count = 0
        self.get_logger().info('SAR Drone TF Bridge (Real World) started.')
        self.get_logger().info('  Dynamic: odom -> base_link (from MAVROS)')
        self.get_logger().info('  Static:  base_link -> camera_link (D455 mount)')
        self.get_logger().info('  RealSense driver handles: camera_link -> optical frames')

    def publish_static_transforms(self):
        """
        Static transform: base_link -> camera_link

        D455 mount position relative to Pixhawk center:
          18.5 cm forward  (+X)
           1.5 cm left     (+Y in ROS convention)
           7.5 cm below    (-Z)
          15 deg pitch down (rotation around Y axis)
        """
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'camera_link'

        # Translation in meters
        t.transform.translation.x = 0.185   # 18.5 cm forward
        t.transform.translation.y = 0.015   # 1.5 cm left
        t.transform.translation.z = -0.075  # 7.5 cm below

        # 15 degrees pitch down (rotation around Y axis)
        pitch_rad = math.radians(15)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = math.sin(pitch_rad / 2)
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = math.cos(pitch_rad / 2)

        self.static_broadcaster.sendTransform(t)

    def odom_callback(self, msg):
        """
        Dynamic transform: odom -> base_link

        Publishes the Pixhawk's position as a TF transform
        using the timestamp from the MAVROS message.
        """
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)

        self.count += 1
        if self.count % 100 == 1:
            self.get_logger().info(
                f'TF #{self.count}: pos=({msg.pose.pose.position.x:.2f}, '
                f'{msg.pose.pose.position.y:.2f}, {msg.pose.pose.position.z:.2f})'
            )


def main():
    rclpy.init()
    node = SarDroneTfBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
