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
import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.style.use('ggplot')
from path_planning import DStarLite, GenerateCoord, Node


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
        lv, rv = operate.pibot.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
        turn_drive_meas = measure.Drive(lv, rv, turn_time)
        operate.update_slam(turn_drive_meas)
    elif turn_diff < 0: # turn right
        lv, rv = operate.pibot.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
        turn_drive_meas = measure.Drive(lv, rv, turn_time)
        operate.update_slam(turn_drive_meas)
    # print(operate.ekf.robot.state)
    
    # compute driving distance to waypoint
    pos_diff = np.hypot(x_diff, y_diff)
    
    # after turning, drive straight to the waypoint
    drive_time = 0.0 # replace with your calculation
    drive_time = pos_diff/(scale*wheel_vel)
    print("Driving for {:.2f} seconds".format(drive_time))
    
    lv, rv = operate.pibot.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    lin_drive_meas = measure.Drive(lv, rv, drive_time)
    print(lin_drive_meas)
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
    
def auto_fruit_search():
    # if self.command['auto_fruit_search'] == True:
    gen_cor = GenerateCoord('M4_true_map.txt')
    fruit_list, _, _ = gen_cor.read_true_map()
    obs_fruit_list = []
    spoofed_obs = []
    spoofed_ox = [[],[],[],[]]
    spoofed_oy = [[],[],[],[]]
    prev_len = len(spoofed_obs)
    
    show_animation = True
    pause_time = 0.1
    p_create_random_obstacle = 0
    
    sx, sy, gx, gy, fx, fy, ox, oy, face_angle = gen_cor.generate_points(spoofed_obs)
    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.plot(fx, fy, "or")
        plt.plot()
        plt.grid(True)
        # plt.gca().set_aspect("equal")
        # plt.axis("equal")
        label_column = ['Start', 'Goal', 'Fruits', 'Path taken',
                        'Current computed path', 'Previous computed path',
                        'Obstacles']
        columns = [plt.plot([], [], symbol, color=colour, alpha=alpha)[0]
                   for symbol, colour, alpha in [['o', 'g', 1],
                                                 ['x', 'b', 1],
                                                 ['o', 'r', 1],
                                                 ['-', 'r', 1],
                                                 ['.', 'c', 1],
                                                 ['.', 'c', 0.3],
                                                 ['.', 'k', 1]]]
        plt.legend(columns, label_column, bbox_to_anchor=(1, 1), title="Key:",
                   fontsize="xx-small")
        plt.plot()
        plt.xlim(-18, 18)
        plt.ylim(-18, 18)
        plt.xticks(range(-16, 17))
        plt.yticks(range(-16, 17))
        plt.show()
        # plt.pause(pause_time)
    
    # for i in range(1):
        # # generate starting, goal, and obstacles points
        # sx, sy, gx, gy, fx, fy, ox, oy, face_angle = gen_cor.generate_points(spoofed_obs)
        # # perform fruits detection to detect obstacles (fruits)
        # dect_fruits = self.detect_target()
        # key = [x for x in dect_fruits.keys()]
        # for k in range(len(key)):
            # res = ""
            # for j in key[k]:
                # if j.isalpha():
                    # res="".join([res,j])
            # if res not in fruit_list and res not in obs_fruit_list:
                # obs_fruit_list.append(res)
                # obs_fruit_x = dect_fruits[key[k]]['x']
                # obs_fruit_y = dect_fruits[key[k]]['y']
                # obs_fruit_x = 0.4*round(obs_fruit_x/0.4)
                # obs_fruit_y = 0.4*round(obs_fruit_y/0.4)
                # spoofed_obs.append([obs_fruit_x, obs_fruit_y])
                # print("new obstacles detected:{}" .format([obs_fruit_x, obs_fruit_y]))  
        
        # # recalculate the points if new obstacles are detected 
        # if prev_len < len(spoofed_obs):
            # prev_len = len(spoofed_obs) 
            # spoofed_ox, spoofed_oy = gen_cor.generate_spoofed_obs(spoofed_obs)
            # sx, sy, gx, gy, fx, fy, ox, oy, face_angle = gen_cor.generate_points(spoofed_obs)
        # dstarlite = DStarLite(ox, oy)
        # _, pathx, pathy = dstarlite.main(Node(x=sx[i], y=sy[i]), Node(x=gx[i], y=gy[i]),
                                         # spoofed_ox=spoofed_ox, spoofed_oy=spoofed_oy)
        # print(pathx)
        # print(pathy)
        # print(face_angle[i])
        
        # for j in range(len(pathx)):
            # if j==len(pathx)-1:
                # robot_pose = drive_to_point([pathx[j]/10, pathy[j]/10, face_angle[i])])
            # else:
                # robot_pose = drive_to_point([pathx[j]/10, pathy[j]/10, np.arctan2(pathy[j], pathx[j])])
        # print("Finished driving to waypoint: {}; New robot pose: {}".format([pathx[j], pathy[j], face_angle[i])))
        
    # self.command['auto_fruit_search'] = False

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
            
    lms = []
    for i,lm in enumerate(aruco_true_pos):
        measure_lm = measure.Marker(np.array([[lm[0]],[lm[1]]]),i+1, covariance=(0.0*np.eye(2)))
        lms.append(measure_lm)
    operate.ekf.add_landmarks_init(lms)   
    operate.output.write_map(operate.ekf)
    
    auto_fruit_search()

    # fruit_no = 3
    # fruit_tag_dict = {'redapple':1,'greenapple':2,'orange':3,'mango':4,'capsicum':5,}
    # fruits = []
    # for i,fr in enumerate(fruits_true_pos):
        # print("fruits", fr)
        # measure_fr = measure.Fruits(np.array([[fr[0]],[fr[1]]]),fruit_tag_dict[fruits_list[i]], )
        # fruits.append(measure_fr)
        # print(fruit_tag_dict[fruits_list[i]])
    # operate.ekf.add_fruits(fruits)

    # The following code is only a skeleton code the semi-auto fruit searching task
    # while True:
        # # enter the waypoints
        # # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input), see camera_calibration.py
        # x,y = 0.0,0.0
        # x = input("X coordinate of the waypoint: ")
        # try:
            # x = float(x)
        # except ValueError:
            # print("Please enter a number.")
            # continue
        # y = input("Y coordinate of the waypoint: ")
        # try:
            # y = float(y)
        # except ValueError:
            # print("Please enter a number.")
            # continue

        # # estimate the robot's pose
        # robot_pose = get_robot_pose(operate)

        # # robot drives to the waypoint
        # waypoint = [x,y]
        # drive_to_point(waypoint,robot_pose,operate)
        # robot_pose = get_robot_pose(operate)
        # print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
        
        # # custom
        # lv, rv = operate.pibot.set_velocity([0, 0], tick=0.0, time=0.0)
        # drive_meas = measure.Drive(lv, rv, 0.0)
        # operate.update_slam(drive_meas)

        # # exit
        # operate.pibot.set_velocity([0, 0])
        # uInput = input("Add a new waypoint? [Y/N]")
        # if uInput == 'N':
            # break