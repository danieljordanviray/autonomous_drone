#!/usr/bin/env python3
"""
Keyboard Drone Controller
==========================
Fly your drone manually with WASD keys via SITL.

Controls:
    W / S       = Forward / Backward
    A / D       = Strafe Left / Right
    Q / E       = Rotate Left / Right
    R / F       = Climb / Descend
    SPACE       = Stop all movement (hover)
    T           = Arm + Takeoff to 4m
    L           = Land
    ESC         = Quit

Usage:
    python keyboard_fly.py

Requires:
    pip install dronekit pynput
"""

import collections
import collections.abc
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable
if not hasattr(collections, 'MutableSequence'):
    collections.MutableSequence = collections.abc.MutableSequence

import time
import os
import sys
os.environ["MAVLINK20"] = "1"

from dronekit import connect
from pymavlink import mavutil
from pynput import keyboard

# ── Config ──
CONNECTION_STRING = '127.0.0.1:14551'
TAKEOFF_ALT = 1.0
STICK_OFFSET = 150

# ── Connect ──
print(f"[*] Connecting to SITL on {CONNECTION_STRING}...")
vehicle = connect(CONNECTION_STRING, wait_ready=False)
time.sleep(3)
print("[✓] Connected!\n")

# ── State ──
keys_pressed = set()
running = True
flying = False  # Only send RC overrides after takeoff

def set_mode(mode_name):
    mode_mapping = {
        'STABILIZE': 0, 'ALT_HOLD': 2, 'GUIDED': 4,
        'LAND': 9, 'RTL': 6, 'LOITER': 5,
    }
    vehicle._master.mav.set_mode_send(
        vehicle._master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_mapping[mode_name]
    )
    time.sleep(1)

def arm_and_takeoff():
    global flying

    # Clear any RC overrides so throttle is neutral
    vehicle.channels.overrides = {}
    keys_pressed.clear()
    time.sleep(0.5)

    print("\n[*] Setting GUIDED mode...")
    set_mode('GUIDED')

    print("[*] Arming...")
    vehicle.armed = True
    timeout = time.time() + 10
    while not vehicle.armed:
        if time.time() > timeout:
            print("[✗] Arming timed out!")
            return
        time.sleep(0.5)
    print("[✓] Armed!")

    print(f"[*] Taking off to {TAKEOFF_ALT}m...")
    vehicle.simple_takeoff(TAKEOFF_ALT)
    while True:
        alt = vehicle.location.global_relative_frame.alt
        if alt >= TAKEOFF_ALT * 0.9:
            break
        time.sleep(0.5)

    print("[✓] At altitude! Switching to LOITER.")
    set_mode('LOITER')
    flying = True
    print("[✓] You have control! Use WASD to fly.\n")

def land():
    global flying
    flying = False
    vehicle.channels.overrides = {}
    keys_pressed.clear()
    print("\n[*] Landing...")
    set_mode('LAND')

def on_press(key):
    global running
    try:
        k = key.char.lower()
        if k == 't':
            arm_and_takeoff()
        elif k == 'l':
            land()
        else:
            keys_pressed.add(k)
    except AttributeError:
        if key == keyboard.Key.space:
            keys_pressed.clear()
        elif key == keyboard.Key.esc:
            running = False

def on_release(key):
    try:
        keys_pressed.discard(key.char.lower())
    except AttributeError:
        pass

# ── Start listener ──
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("╔══════════════════════════════════════════╗")
print("║         KEYBOARD DRONE CONTROLLER        ║")
print("╠══════════════════════════════════════════╣")
print("║  W/S     = Forward / Backward            ║")
print("║  A/D     = Strafe Left / Right            ║")
print("║  Q/E     = Rotate Left / Right            ║")
print("║  R/F     = Climb / Descend                ║")
print("║  SPACE   = Stop all movement              ║")
print("║  T       = Arm + Takeoff                  ║")
print("║  L       = Land                           ║")
print("║  ESC     = Quit                           ║")
print("╚══════════════════════════════════════════╝")
print("\nPress T to arm and takeoff!\n")

# ── Main loop ──
try:
    while running:
        # Only send RC overrides when airborne in LOITER
        if flying:
            ch1 = 1500  # roll
            ch2 = 1500  # pitch
            ch3 = 1500  # throttle
            ch4 = 1500  # yaw

            if 'w' in keys_pressed: ch2 = 1500 - STICK_OFFSET
            if 's' in keys_pressed: ch2 = 1500 + STICK_OFFSET
            if 'a' in keys_pressed: ch1 = 1500 - STICK_OFFSET
            if 'd' in keys_pressed: ch1 = 1500 + STICK_OFFSET
            if 'q' in keys_pressed: ch4 = 1500 - STICK_OFFSET
            if 'e' in keys_pressed: ch4 = 1500 + STICK_OFFSET
            if 'r' in keys_pressed: ch3 = 1500 + STICK_OFFSET
            if 'f' in keys_pressed: ch3 = 1500 - STICK_OFFSET

            vehicle.channels.overrides = {'1': ch1, '2': ch2, '3': ch3, '4': ch4}

        # Single status line that overwrites itself
        alt = vehicle.location.global_relative_frame.alt
        heading = vehicle.heading
        mode = vehicle.mode.name
        active = sorted(keys_pressed) if keys_pressed else '-'

        sys.stdout.write(f"\r  {mode:<10} Alt:{alt:.1f}m  Hdg:{heading:03d}°  Keys:{active}          ")
        sys.stdout.flush()

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

# ── Cleanup ──
print("\n\n[*] Releasing controls...")
vehicle.channels.overrides = {}
vehicle.close()
print("[✓] Done.")
