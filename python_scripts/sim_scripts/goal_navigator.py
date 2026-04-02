#!/usr/bin/env python3
"""
Goal Navigator with Yaw Control

Click "Publish Point" in RViz toolbar, then click on the map.
The drone rotates to face the goal, then flies there.
Slows down as it approaches to prevent overshoot.
Actively holds position after arrival.

Publishes velocity commands to /cmd_vel_nav (not directly to MAVROS).
The obstacle avoidance node sits between this and MAVROS.

Run with:
  python3 goal_navigator.py --ros-args -p use_sim_time:=true
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TwistStamped
from tf2_ros import Buffer, TransformListener


# ============================================================
# TUNABLE PARAMETERS - Change these to adjust flight behavior
# ============================================================
FLIGHT_SPEED = 1.5            # Max flight speed in m/s
SLOWDOWN_RADIUS = 4.0         # Start slowing down within this distance (meters)
MIN_SPEED = 0.15              # Minimum speed when approaching goal (m/s)
GOAL_ALTITUDE = 1.0           # Altitude to maintain in meters
ARRIVAL_THRESHOLD = 0.3       # Distance to goal considered "arrived" in meters
MAX_VERTICAL_SPEED = 0.5      # Max vertical speed in m/s
ALTITUDE_GAIN = 0.5           # How aggressively to correct altitude
YAW_RATE = 1.0                # Yaw rotation speed in rad/s (~57 deg/s)
YAW_THRESHOLD = 0.08          # Yaw error to start flying (~5 degrees)
YAW_TRACKING_GAIN = 3.0       # Proportional gain for yaw correction while flying
NAV_LOOP_RATE = 10            # Navigation loop frequency in Hz
# ============================================================


class GoalNavigator(Node):
    def __init__(self):
        super().__init__('goal_navigator')

        # TF buffer to look up drone position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribe to clicked points from RViz
        self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.goal_callback,
            10
        )

        # Publisher for velocity commands to obstacle avoidance node
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel_nav',
            10
        )

        # Navigation state
        self.goal = None
        self.state = 'IDLE'  # IDLE, ROTATING, FLYING, ARRIVED
        self._log_counter = 0

        # Timer to run navigation loop
        self.create_timer(1.0 / NAV_LOOP_RATE, self.navigate)

        self.get_logger().info('Goal Navigator started.')
        self.get_logger().info(f'  Flight speed: {FLIGHT_SPEED} m/s')
        self.get_logger().info(f'  Slowdown radius: {SLOWDOWN_RADIUS} m')
        self.get_logger().info(f'  Yaw rate: {math.degrees(YAW_RATE):.0f} deg/s')
        self.get_logger().info(f'  Yaw threshold: {math.degrees(YAW_THRESHOLD):.1f} deg')
        self.get_logger().info(f'  Altitude: {GOAL_ALTITUDE} m')
        self.get_logger().info(f'  Publishing to: /cmd_vel_nav')
        self.get_logger().info('Click "Publish Point" in RViz, then click on the map.')

    def goal_callback(self, msg):
        """Called when you click a point in RViz."""
        self.goal = (msg.point.x, msg.point.y)
        self.state = 'ROTATING'
        self.get_logger().info(
            f'New goal: ({msg.point.x:.2f}, {msg.point.y:.2f}) - Rotating to face goal...'
        )

    def get_drone_pose(self):
        """
        Look up drone's current position and yaw from TF.
        Returns (x, y, z, yaw) or None.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            # Extract yaw from quaternion
            q = transform.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            return (x, y, z, yaw)
        except Exception:
            return None

    def angle_diff(self, target, current):
        """Shortest angular difference in [-pi, pi]."""
        diff = target - current
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def publish_velocity(self, vx, vy, vz, yaw_rate=0.0):
        """Send velocity + yaw rate command to /cmd_vel_nav."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.z = yaw_rate
        self.vel_pub.publish(msg)

    def navigate(self):
        """
        Navigation state machine. Runs at NAV_LOOP_RATE Hz.

        IDLE     -> waiting for goal
        ROTATING -> turning to face goal, no forward movement
        FLYING   -> moving toward goal, yaw tracks direction of travel
        ARRIVED  -> holding position
        """
        # Get current pose
        pose = self.get_drone_pose()
        if pose is None:
            return

        current_x, current_y, current_z, current_yaw = pose

        # Altitude correction (always active)
        dz = GOAL_ALTITUDE - current_z
        vel_z = max(-MAX_VERTICAL_SPEED, min(MAX_VERTICAL_SPEED, dz * ALTITUDE_GAIN))

        # ARRIVED: hold position
        if self.state == 'ARRIVED':
            self.publish_velocity(0.0, 0.0, vel_z, 0.0)
            return

        # IDLE: do nothing
        if self.state == 'IDLE' or self.goal is None:
            return

        goal_x, goal_y = self.goal

        # Compute angle and distance to goal
        dx = goal_x - current_x
        dy = goal_y - current_y
        distance = math.sqrt(dx * dx + dy * dy)
        target_yaw = math.atan2(dy, dx)
        yaw_error = self.angle_diff(target_yaw, current_yaw)

        # ROTATING: turn in place to face goal
        if self.state == 'ROTATING':
            if abs(yaw_error) < YAW_THRESHOLD:
                self.state = 'FLYING'
                self.get_logger().info(
                    f'Rotation complete (error: {math.degrees(yaw_error):.1f}deg). '
                    f'Flying to goal, distance: {distance:.2f}m'
                )
            else:
                # Proportional yaw rate — fast when far, slow when close
                yaw_cmd = yaw_error * YAW_TRACKING_GAIN
                yaw_cmd = max(-YAW_RATE, min(YAW_RATE, yaw_cmd))
                self.publish_velocity(0.0, 0.0, vel_z, yaw_cmd)
            return

        # FLYING: move toward goal with yaw tracking
        if self.state == 'FLYING':
            # Check if arrived
            if distance < ARRIVAL_THRESHOLD:
                self.state = 'ARRIVED'
                self.get_logger().info(
                    f'Arrived at goal! Distance: {distance:.2f}m'
                )
                self.publish_velocity(0.0, 0.0, vel_z, 0.0)
                return

            # Compute direction unit vector
            dir_x = dx / distance
            dir_y = dy / distance

            # Slow down as we approach the goal
            if distance < SLOWDOWN_RADIUS:
                speed = FLIGHT_SPEED * (distance / SLOWDOWN_RADIUS)
                speed = max(speed, MIN_SPEED)
            else:
                speed = FLIGHT_SPEED

            # Compute velocity toward goal
            vel_x = dir_x * speed
            vel_y = dir_y * speed

            # Compute yaw rate to keep facing direction of travel
            yaw_cmd = yaw_error * YAW_TRACKING_GAIN
            yaw_cmd = max(-YAW_RATE, min(YAW_RATE, yaw_cmd))

            # Send velocity + yaw command
            self.publish_velocity(vel_x, vel_y, vel_z, yaw_cmd)

            # Log progress every ~1 second
            self._log_counter += 1
            if self._log_counter % NAV_LOOP_RATE == 0:
                self.get_logger().info(
                    f'Flying: dist={distance:.2f}m, speed={speed:.2f}m/s, '
                    f'yaw_err={math.degrees(yaw_error):.1f}deg, '
                    f'pos=({current_x:.2f}, {current_y:.2f}, {current_z:.2f})'
                )


def main():
    rclpy.init()
    node = GoalNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
