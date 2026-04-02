#!/usr/bin/env python3
from pymavlink import mavutil
import time

conn = mavutil.mavlink_connection('/dev/ttyAMA0', baud=921600)
conn.wait_heartbeat()
print("Connected to Pixhawk")

conn.mav.set_gps_global_origin_send(
    conn.target_system,
    int(41.3 * 1e7),
    int(-72.9 * 1e7),
    0
)
print("Sent SET_GPS_GLOBAL_ORIGIN")

time.sleep(1)

conn.mav.set_home_position_send(
    conn.target_system,
    int(41.3 * 1e7),
    int(-72.9 * 1e7),
    0, 0, 0, 0,
    [0, 0, 0, 0],
    0, 0, 0,
    0
)
print("Sent SET_HOME_POSITION")
print("Done. Close this and start MAVROS.")
