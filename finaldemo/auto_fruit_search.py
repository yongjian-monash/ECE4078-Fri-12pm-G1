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
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
from operate import Operate

# import utility functions
sys.path.insert(0, "util")
from pibot import Alphabot
import measure as measure




def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id-1][0] = x
                    aruco_true_pos[marker_id-1][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point
    
    x_goalpos = waypoint[0]
    y_goalpos = waypoint[1]
    
    robot_x_to_goal = x_goalpos-robot_pose[0]
    robot_y_to_goal = y_goalpos-robot_pose[1]
    robot_angle = robot_pose[2]
    
    
    
    if robot_angle < 0:
        robot_angle = robot_angle + 2*np.pi
    else:
        robot_angle = robot_angle
        
    if (robot_x_to_goal == 0):   
        if (robot_y_to_goal < 0): #means robot needs to turn anticlockwise 90deg
            robot_turn = - robot_angle - np.pi/2
        else:
            robot_turn = -robot_angle + np.pi/2
        
    elif (robot_y_to_goal == 0):
        if(robot_x_to_goal < 0):
            robot_turn = np.pi - robot_angle
        else:
            robot_turn = -robot_angle
            
    else:
        robot_calc =abs(np.arctan2(robot_y_to_goal,robot_x_to_goal))
        if(robot_x_to_goal>0) and (robot_y_to_goal>0):
            robot_turn = robot_calc
        elif(robot_x_to_goal<0) and (robot_y_to_goal>0):
            robot_turn = np.pi - robot_calc
        elif(robot_x_to_goal<0) and (robot_y_to_goal<0):
            robot_turn = np.pi + robot_calc
        elif(robot_x_to_goal>0) and (robot_y_to_goal<0):
            robot_turn = 2*np.pi - robot_calc
    
    turn_cw = robot_turn - robot_angle
    turn_ccw = (2*np.pi - abs(turn_cw)) * -np.sign(turn_cw)
    if abs(turn_cw)<abs(turn_ccw):
        robot_turn = turn_cw
    else:
        robot_turn = turn_ccw
        
    
    wheel_vel = 10 # tick to move the robot
    
    # turn towards the waypoint
    turn_time = abs(robot_turn) * ((baseline/2)/ (scale*wheel_vel)) # replace with your calculation
    print("Turning for {:.2f} seconds".format(turn_time))
    ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    
    lv, rv = operate.pibot.set_velocity([0, np.sign(turn_cw - turn_ccw)], turning_tick=wheel_vel, time=turn_time)
    dt = time.time() - operate.control_clock
    drive_meas = measure.Drive(lv, rv, dt)
    operate.control_clock = time.time()
    lms, operate.aruco_img = operate.aruco_det.detect_marker_positions(operate.img)
    operate.ekf.predict(drive_meas)
    operate.ekf.add_landmarks(lms)
    operate.ekf.update(lms)
    
    distance_to_goal = np.hypot(robot_x_to_goal,robot_y_to_goal)
    # after turning, drive straight to the waypoint
    drive_time = distance_to_goal/ (scale*wheel_vel) # replace with your calculation
    print("Driving for {:.2f} seconds".format(drive_time))
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    
    lv,rv =operate.pibot.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    dt = time.time() - operate.control_clock
    drive_meas = measure.Drive(lv, rv, dt)
    operate.control_clock = time.time()
    lms, operate.aruco_img = operate.aruco_det.detect_marker_positions(operate.img)
    operate.ekf.predict(drive_meas)
    operate.ekf.add_landmarks(lms)
    operate.ekf.update(lms)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose(operate,robot_pose):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    #robot_pose = [0.0,0.0,0.0] # replace with your calculation
    robot_pose[2] = operate.ekf.robot.state[2][0] # Measured from x-axis (theta=0)
    robot_pose[2] = (robot_pose[2] + 2*np.pi) if (robot_pose[2] < 0) else robot_pose[2]
    robot_pose[0] = operate.ekf.robot.state[0][0]
    robot_pose[1] = operate.ekf.robot.state[1][0]
    ####################################################

    return robot_pose

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    parser.add_argument("--map", type=str, default='M4_true_map.txt')    
    args, _ = parser.parse_known_args()

    ppi = Alphabot(args.ip,args.port)
    operate = Operate(args)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    # The following code is only a skeleton code the semi-auto fruit searching task
    while True:
        # enter the waypoints
        # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input), see camera_calibration.py
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        # estimate the robot's pose
        #robot_pose = get_robot_pose(operate,robot_pose)

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose)
        robot_pose = get_robot_pose(operate,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break
            
            
