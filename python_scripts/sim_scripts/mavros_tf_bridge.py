#!/usr/bin/env python3
"""
SAR Drone TF Bridge

Publishes the complete transform tree for the drone:

  odom -> base_link                              (dynamic, from MAVROS odometry)
  base_link -> camera_link                       (static, mount position + 15deg pitch)
  camera_link -> f450_sar/camera_link/d455_depth (static, optical frame rotation)
  camera_link -> f450_sar/camera_link/d455_color (static, optical frame rotation)

The optical frame rotation converts from ROS body convention
(X-forward, Y-left, Z-up) to camera optical convention
(X-right, Y-down, Z-forward). This is required because RTAB-Map
reprojects depth images assuming the camera frame uses optical
convention (depth is along Z axis of the camera frame).

map -> odom is published by RTAB-Map (SLAM correction).

Run with:
  python3 mavros_tf_bridge.py --ros-args -p use_sim_time:=true
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from transforms3d.euler import euler2quat


class SarDroneTfBridge(Node):
    def __init__(self):
        super().__init__('sar_drone_tf_bridge')

        # Dynamic TF broadcaster (for transforms that change over time)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Static TF broadcaster (for transforms that never change)
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # Publish all static camera transforms once
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
        self.get_logger().info('SAR Drone TF Bridge started.')
        self.get_logger().info('  Dynamic: odom -> base_link (from MAVROS)')
        self.get_logger().info('  Static:  base_link -> camera_link (mount + pitch)')
        self.get_logger().info('  Static:  camera_link -> d455_depth (optical rotation)')
        self.get_logger().info('  Static:  camera_link -> d455_color (optical rotation)')

    def publish_static_transforms(self):
        """
        Publish all static transforms at once.

        Transform 1: base_link -> camera_link
          Camera mount position (0.1m forward, 0.05m below)
          and 15 degree downward pitch. This frame is in ROS
          body convention (X-forward, Z-up).

        Transform 2: camera_link -> d455_depth (optical rotation)
          Rotates from ROS body convention to camera optical convention.
          rpy = (-pi/2, 0, -pi/2)

        Transform 3: camera_link -> d455_color (optical rotation)
          Same optical rotation. In sim these are colocated.
          On real D455 they'd be offset by the stereo baseline.
        """
        transforms = []

        # --- Transform 1: base_link -> camera_link ---
        # Camera mount: 10cm forward, 5cm below, pitched 15deg down
        t1 = TransformStamped()
        t1.header.stamp = self.get_clock().now().to_msg()
        t1.header.frame_id = 'base_link'
        t1.child_frame_id = 'camera_link'

        t1.transform.translation.x = 0.1
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = -0.05

        # 15 degrees pitch down (rotation around Y axis)
        pitch_rad = math.radians(15)
        t1.transform.rotation.x = 0.0
        t1.transform.rotation.y = math.sin(pitch_rad / 2)
        t1.transform.rotation.z = 0.0
        t1.transform.rotation.w = math.cos(pitch_rad / 2)

        transforms.append(t1)

        # --- Transform 2: camera_link -> depth optical frame ---
        # Optical rotation: rpy = (-pi/2, 0, -pi/2)
        # This converts X-forward/Z-up to Z-forward/Y-down
        t2 = TransformStamped()
        t2.header.stamp = self.get_clock().now().to_msg()
        t2.header.frame_id = 'camera_link'
        t2.child_frame_id = 'f450_sar/camera_link/d455_depth'

        t2.transform.translation.x = 0.0
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.0

        # Optical frame rotation: roll=-pi/2, pitch=0, yaw=-pi/2
        q = euler2quat(-math.pi/2, 0, -math.pi/2, 'sxyz')
        t2.transform.rotation.w = q[0]
        t2.transform.rotation.x = q[1]
        t2.transform.rotation.y = q[2]
        t2.transform.rotation.z = q[3]

        transforms.append(t2)

        # --- Transform 3: camera_link -> color optical frame ---
        # Same optical rotation as depth
        t3 = TransformStamped()
        t3.header.stamp = self.get_clock().now().to_msg()
        t3.header.frame_id = 'camera_link'
        t3.child_frame_id = 'f450_sar/camera_link/d455_color'

        t3.transform.translation.x = 0.0
        t3.transform.translation.y = 0.0
        t3.transform.translation.z = 0.0

        t3.transform.rotation.x = q[0]
        t3.transform.rotation.y = q[1]
        t3.transform.rotation.z = q[2]
        t3.transform.rotation.w = q[3]

        transforms.append(t3)

        # Publish all static transforms at once
        self.static_broadcaster.sendTransform(transforms)

    def odom_callback(self, msg):
        """
        Dynamic transform: odom -> base_link

        Every time MAVROS tells us the drone's position, we publish
        it as a TF transform so RViz and other nodes know where the
        drone is in the world.

        Uses the timestamp from the MAVROS message (sim time).
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