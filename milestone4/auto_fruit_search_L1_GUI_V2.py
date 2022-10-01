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

# custom
from operate import Operate
import pygame


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
def drive_to_point(waypoint, robot_pose, operate):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    wheel_vel = 20 # tick to move the robot
    
    # compute x and y distance to waypoint
    x_diff = waypoint[0] - robot_pose[0]
    y_diff = waypoint[1] - robot_pose[1]
    
    # wrap robot orientation to (-pi, pi]
    robot_orient = (robot_pose[2]) % (2*np.pi)
    if robot_orient > np.pi:
        robot_orient -= 2*np.pi
    
    # compute min turning angle to waypoint
    turn_diff = np.arctan2(y_diff, x_diff) - robot_orient
    if turn_diff > np.pi:
        turn_diff -= 2*np.pi
    elif turn_diff < -np.pi:
        turn_diff += 2*np.pi
    
    # turn towards the waypoint
    turn_time = 0.0 # replace with your calculation
    turn_time = (abs(turn_diff)*baseline)/(2.0*scale*wheel_vel) # replace with your calculation
    print("Turning for {:.2f} seconds".format(turn_time))
    
    if turn_diff > 0: # turn left
        lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
        turn_drive_meas = measure.Drive(lv, rv, turn_time)
        operate.update_slam(turn_drive_meas)
    elif turn_diff < 0: # turn right
        lv, rv = ppi.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
        turn_drive_meas = measure.Drive(lv, rv, turn_time)
        operate.update_slam(turn_drive_meas)
    # print(operate.ekf.robot.state)
    
    # compute driving distance to waypoint
    pos_diff = np.hypot(x_diff, y_diff)
    
    # after turning, drive straight to the waypoint
    drive_time = 0.0 # replace with your calculation
    drive_time = pos_diff/(scale*wheel_vel)
    print("Driving for {:.2f} seconds".format(drive_time))
    
    lv, rv = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    lin_drive_meas = measure.Drive(lv, rv, drive_time)
    operate.update_slam(lin_drive_meas)
    # print(operate.ekf.robot.state)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose(operate):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    # robot_pose = [0.0,0.0,0.0] # replace with your calculation
    robot_pose = operate.ekf.robot.state.squeeze().tolist()
    ####################################################

    return robot_pose

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

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]
    
    # custom
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)
    n_observed_markers = len(operate.ekf.taglist)
    if n_observed_markers == 0:
        if not operate.ekf_on:
            print('SLAM is running')
            operate.ekf_on = True
        else:
            print('> 2 landmarks is required for pausing')
    elif n_observed_markers < 3:
        print('> 2 landmarks is required for pausing')
    else:
        if not operate.ekf_on:
            operate.request_recover_robot = True
        operate.ekf_on = not operate.ekf_on
        if operate.ekf_on:
            print('SLAM is running')
        else:
            print('SLAM is paused')

    # The following code is only a skeleton code the semi-auto fruit searching task
    while start:
        operate.take_pic()
        operate.draw(canvas)
        pygame.display.update()
        
        lv, rv = ppi.set_velocity([0, 0], tick=0.0, time=0.0)
        drive_meas = measure.Drive(lv, rv, 0.0)
        operate.update_slam(drive_meas)

        # enter the waypoints
        # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input), see camera_calibration.py
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.q:
                x_active = False
                y_active = False
                x = ""
                y = ""
                operate.notification=""
            elif event.type == pygame.KEYDOWN and event.key == pygame.w:
                x_active = True
                x = ""
                y = ""
            elif event.type == pygame.KEYDOWN and x_active:
                if event.key == pygame.K_RETURN:
                    x_active = False
                    y_active = True
                    
                    try:
                        x = float(x)
                    except ValueError:
                        operate.notification="Please enter a number."
                        x_active = False
                        y_active = False
                elif event.key == pygame.K_BACKSPACE:
                    x =  x[:-1]
                else:
                    x += event.unicode
                    operate.notification = f"Waypoint X: {x}, Waypoint Y: {y}"
            elif event.type == pygame.KEYDOWN and y_active:
                if event.key == pygame.K_RETURN:
                    y_active = False
                    
                    try:
                        y = float(y)
                        drive_flag = 1
                    except ValueError:
                        operate.notification="Please enter a number."
                        x_active = False
                        y_active = False
                        
                    if drive_flag:
                        # estimate the robot's pose
                        robot_pose = get_robot_pose(operate)

                        # robot drives to the waypoint
                        waypoint = [x,y]
                        drive_to_point(waypoint,robot_pose,operate)
                        robot_pose = get_robot_pose(operate)
                        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
                        drive_flag = 0
                elif event.key == pygame.K_BACKSPACE:
                    y =  y[:-1]
                else:
                    y += event.unicode
                    operate.notification = f"Waypoint X: {x}, Waypoint Y: {y}"
        
        if self.quit:
            pygame.quit()
            sys.exit()
                    
        # lv, rv = ppi.set_velocity([0, 0], tick=0.0, time=0.0)
        # drive_meas = measure.Drive(lv, rv, 0.0)
        # operate.update_slam(drive_meas)
        
        # # exit
        # ppi.set_velocity([0, 0])
        # uInput = input("Add a new waypoint? [Y/N]")
        # if uInput == 'N':
            # break