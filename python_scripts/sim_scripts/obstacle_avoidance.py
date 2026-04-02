#!/usr/bin/env python3
"""
Obstacle Avoidance Safety Node with Live Debug Window

Sits between the goal navigator and MAVROS as a safety layer.
Detection region is sized to match the drone's physical width.
Includes braking cooldown and rotation grace period.

Pipeline:
  Goal Navigator -> /cmd_vel_nav -> [THIS NODE] -> /mavros/setpoint_velocity/cmd_vel

Run with:
  python3 obstacle_avoidance.py --ros-args -p use_sim_time:=true

Press 'q' in the debug window to quit.
"""
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PointStamped, TwistStamped
from sensor_msgs.msg import Image


# ============================================================
# TUNABLE PARAMETERS
# ============================================================
SAFETY_DISTANCE = 1.5         # Stop if obstacle closer than this (meters)
WARNING_DISTANCE = 3.0        # Show warning if obstacle closer than this (meters)
BRAKING_COOLDOWN = 2.0        # Keep stopped for this many seconds after obstacle detected
ROTATION_GRACE_PERIOD = 5.0   # Seconds to pause detection after new goal (allow rotation)
DRONE_WIDTH_FRACTION = 0.25   # Detection box width as fraction of image
DRONE_HEIGHT_FRACTION = 0.30  # Detection box height as fraction of image
CHECK_RATE = 10               # How often to process (Hz)
# ============================================================


def ros_image_to_cv2(msg):
    """Convert a ROS Image message to an OpenCV numpy array."""
    if msg.encoding == 'rgb8':
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif msg.encoding == 'bgr8':
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    elif msg.encoding == '32FC1':
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    elif msg.encoding == '16UC1':
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        return arr.astype(np.float32) / 1000.0
    else:
        return None


class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')

        # Latest data
        self.min_depth = float('inf')
        self.latest_cmd = None
        self.latest_color = None
        self.detection_box = None

        # Braking state
        self.obstacle_detected = False
        self.brake_start_time = None

        # Grace period state (pauses detection during rotation)
        self.grace_period_start = None

        # Subscribe to velocity commands from goal navigator
        self.create_subscription(
            TwistStamped,
            '/cmd_vel_nav',
            self.cmd_callback,
            10
        )

        # Subscribe to clicked points to reset state on new goal
        self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.reset_callback,
            10
        )

        # Subscribe to depth image
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            Image,
            '/camera/depth',
            self.depth_callback,
            qos
        )

        # Subscribe to color image for debug window
        self.create_subscription(
            Image,
            '/camera/color',
            self.color_callback,
            qos
        )

        # Publisher to MAVROS
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel',
            10
        )

        # Timer to process and forward commands
        self.create_timer(1.0 / CHECK_RATE, self.process)

        self._log_counter = 0

        self.get_logger().info('Obstacle Avoidance node started.')
        self.get_logger().info(f'  Safety distance: {SAFETY_DISTANCE} m')
        self.get_logger().info(f'  Warning distance: {WARNING_DISTANCE} m')
        self.get_logger().info(f'  Braking cooldown: {BRAKING_COOLDOWN} s')
        self.get_logger().info(f'  Rotation grace period: {ROTATION_GRACE_PERIOD} s')
        self.get_logger().info(f'  Detection box: {DRONE_WIDTH_FRACTION*100:.0f}% W x {DRONE_HEIGHT_FRACTION*100:.0f}% H')
        self.get_logger().info('  Press q in debug window to quit.')

    def cmd_callback(self, msg):
        """Store the latest velocity command from the navigator."""
        self.latest_cmd = msg

    def reset_callback(self, msg):
        """New goal received - pause detection to allow rotation."""
        self.obstacle_detected = False
        self.brake_start_time = None
        self.grace_period_start = time.time()
        self.get_logger().info(
            f'New goal received - detection paused for {ROTATION_GRACE_PERIOD}s rotation.'
        )

    def color_callback(self, msg):
        """Store the latest color image for debug window."""
        img = ros_image_to_cv2(msg)
        if img is not None and len(img.shape) == 3:
            self.latest_color = img

    def depth_callback(self, msg):
        """Process depth image to find minimum distance ahead."""
        try:
            depth_array = ros_image_to_cv2(msg)
            if depth_array is None:
                self.get_logger().warn(
                    f'Unknown depth encoding: {msg.encoding}', throttle_duration_sec=5.0
                )
                return

            h, w = depth_array.shape

            box_w = int(w * DRONE_WIDTH_FRACTION)
            box_h = int(h * DRONE_HEIGHT_FRACTION)

            x1 = (w - box_w) // 2
            y1 = (h - box_h) // 2
            x2 = x1 + box_w
            y2 = y1 + box_h

            self.detection_box = (x1, y1, x2, y2)

            center = depth_array[y1:y2, x1:x2]
            valid = center[(center > 0.1) & np.isfinite(center)]

            if len(valid) > 0:
                self.min_depth = float(np.min(valid))
            else:
                self.min_depth = float('inf')

        except Exception as e:
            self.get_logger().error(f'Depth processing error: {e}')

    def in_grace_period(self):
        """Check if we're in the rotation grace period."""
        if self.grace_period_start is None:
            return False

        elapsed = time.time() - self.grace_period_start
        if elapsed < ROTATION_GRACE_PERIOD:
            return True
        else:
            self.grace_period_start = None
            self.get_logger().info('Grace period ended - obstacle detection active.')
            return False

    def is_braking(self):
        """Check if we're in braking cooldown."""
        if self.brake_start_time is None:
            return False

        elapsed = time.time() - self.brake_start_time
        if elapsed < BRAKING_COOLDOWN:
            return True
        else:
            self.brake_start_time = None
            return False

    def send_stop(self):
        """Send zero velocity command (keep altitude correction)."""
        safe_cmd = TwistStamped()
        safe_cmd.header.stamp = self.get_clock().now().to_msg()
        safe_cmd.header.frame_id = 'map'
        safe_cmd.twist.linear.x = 0.0
        safe_cmd.twist.linear.y = 0.0
        if self.latest_cmd is not None:
            safe_cmd.twist.linear.z = self.latest_cmd.twist.linear.z
        else:
            safe_cmd.twist.linear.z = 0.0
        safe_cmd.twist.angular.z = 0.0
        self.vel_pub.publish(safe_cmd)

    def show_debug_window(self):
        """Display live OpenCV window with camera feed and detection overlay."""
        if self.latest_color is None:
            return

        frame = self.latest_color.copy()
        h, w = frame.shape[:2]

        if self.detection_box is not None:
            x1, y1, x2, y2 = self.detection_box

            # Determine display status
            in_grace = self.in_grace_period()

            if in_grace:
                box_color = (255, 255, 0)  # Cyan - grace period
                status = "ROTATING"
            elif self.obstacle_detected or self.is_braking():
                box_color = (0, 0, 255)    # Red
                status = "STOP"
                if self.is_braking() and not self.obstacle_detected:
                    status = "BRAKING"
            elif self.min_depth < WARNING_DISTANCE:
                box_color = (0, 165, 255)  # Orange
                status = "WARNING"
            else:
                box_color = (0, 255, 0)    # Green
                status = "CLEAR"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Semi-transparent fill
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

            # Status text above box
            cv2.putText(
                frame, f'{status} - {self.min_depth:.2f}m',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2
            )

            # Depth value in center of box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.putText(
                frame, f'{self.min_depth:.2f}m',
                (cx - 30, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        # Top bar with status
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        nav_status = "NAVIGATING" if self.latest_cmd is not None else "IDLE"
        if self.in_grace_period():
            nav_status = "ROTATING (GRACE)"
        elif self.obstacle_detected:
            nav_status = "STOPPED"
        elif self.is_braking():
            nav_status = "BRAKING"
        cv2.putText(
            frame, f'OBSTACLE AVOIDANCE | {nav_status}',
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        # Timer display for grace period or cooldown
        if self.grace_period_start is not None:
            elapsed = time.time() - self.grace_period_start
            remaining = max(0, ROTATION_GRACE_PERIOD - elapsed)
            cv2.putText(
                frame, f'Grace: {remaining:.1f}s',
                (w - 140, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
            )
        elif self.brake_start_time is not None:
            elapsed = time.time() - self.brake_start_time
            remaining = max(0, BRAKING_COOLDOWN - elapsed)
            cv2.putText(
                frame, f'Brake: {remaining:.1f}s',
                (w - 140, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1
            )

        cv2.imshow('Obstacle Avoidance', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Quit requested. Shutting down.')
            raise SystemExit

    def process(self):
        """Check depth and forward or override velocity commands."""
        self.show_debug_window()

        if self.latest_cmd is None:
            return

        # During grace period, pass through all commands without checking
        if self.in_grace_period():
            self.vel_pub.publish(self.latest_cmd)
            return

        # Check for obstacles
        obstacle_now = self.min_depth < SAFETY_DISTANCE

        if obstacle_now:
            if not self.obstacle_detected:
                self.get_logger().warn(
                    f'OBSTACLE DETECTED at {self.min_depth:.2f}m! Stopping.'
                )
                self.obstacle_detected = True
                self.brake_start_time = time.time()

            # Reset cooldown timer while obstacle is still visible
            self.brake_start_time = time.time()
            self.send_stop()

        elif self.is_braking():
            if self.obstacle_detected:
                self.get_logger().info(
                    f'Obstacle cleared. Braking cooldown active...'
                )
                self.obstacle_detected = False

            self.send_stop()

        else:
            if self.obstacle_detected:
                self.get_logger().info(
                    f'Obstacle cleared. Min distance: {self.min_depth:.2f}m. Resuming.'
                )
                self.obstacle_detected = False

            self.vel_pub.publish(self.latest_cmd)

        # Log periodically
        self._log_counter += 1
        if self._log_counter % (CHECK_RATE * 2) == 0:
            braking = self.is_braking()
            status = 'STOPPED' if self.obstacle_detected else ('BRAKING' if braking else 'CLEAR')
            self.get_logger().info(
                f'[{status}] Min depth: {self.min_depth:.2f}m'
            )


def main():
    rclpy.init()
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()