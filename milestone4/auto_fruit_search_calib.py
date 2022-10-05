# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import ast
import argparse
import time

# import SLAM components
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
# from slam.ekf import EKF
# from slam.robot import Robot
# import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import Alphabot
import measure as measure

def drive_to_point(ppi, turn_diff = 0.0, pos_diff = 0.0):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
   
    wheel_vel = 20 # tick to move the robot
    
    turn_diff = turn_diff*np.pi/180
    
    if turn_diff > 0: # turn left
        turn_time = (abs(turn_diff)*baseline)/(2.0*scale*wheel_vel)
        lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    elif turn_diff < 0: # turn right
        turn_time = (abs(turn_diff)*baseline*1.06)/(2.0*scale*wheel_vel)
        lv, rv = ppi.pibot.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
    print("Turning for {:.2f} seconds".format(turn_time))
    
    if pos_diff > 0:
        drive_time = pos_diff/(scale*wheel_vel)
        lv, rv = operate.pibot.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    print("Driving for {:.2f} seconds".format(drive_time))

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
    
# rotate robot to scan landmarks
def rotate_robot(ppi, num_turns=8):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    wheel_vel = 20 # tick to move the robot
    
    turn_resolution = 2*np.pi/num_turns
    turn_time = (abs(turn_resolution)*baseline)/(2.0*scale*wheel_vel) + 0.024
    print(turn_time)
    
    for _ in range(num_turns):
        lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
        time.sleep(0.2)

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    # parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    args, _ = parser.parse_known_args()

    ppi = Alphabot(args.ip,args.port)

    while True:
        turn_diff,pos_diff = 0.0,0.0
        turn_diff = input("Turning angle: ")
        try:
            turn_diff = float(turn_diff)
        except ValueError:
            print("Please enter a number.")
            continue
        pos_diff = input("Distance: ")
        try:
            pos_diff = float(pos_diff)
        except ValueError:
            print("Please enter a number.")
            continue

        # robot drives to the waypoint
        drive_to_point(ppi,turn_diff,pos_diff)

        # # exit
        # operate.pibot.set_velocity([0, 0])
        # uInput = input("Add a new waypoint? [Y/N]")
        # if uInput == 'N':
            # break