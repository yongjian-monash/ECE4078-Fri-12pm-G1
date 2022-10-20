# teleoperate the robot, perform SLAM and object detection

# basic python packages
import numpy as np
import cv2 
import os, sys
import time

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import Alphabot # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf_no_update_marker import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector

# custom added
import SLAM_eval
from a_star import *
from TargetPoseEst import live_fruit_pose

class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = Alphabot(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.06) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False,
                        'output2': False,
                        'auto_fruit_search': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 600
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        
        # if args.ckpt == "":
            # self.detector = None
            # self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        # else:
            # self.detector = Detector(args.ckpt, use_gpu=False)
            # self.network_vis = np.ones((240, 320,3))* 100
            
        self.detector = Detector(None, use_gpu=False)
        self.network_vis = np.ones((240, 320,3))* 100
        
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        
        self.path_planning = None
        self.ox = []
        self.oy = []
        self.waypoints_list = []
        self.spoofed_obs = []
        self.fruit_goals_remain = []
        self.count_rot=0

    # wheel control
    def control(self):       
        if args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas
        
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.yolo_detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
                
                # testing
                self.fruit_detect_update_test()
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False
        # custom function
        if self.command['output2']:
            self.output.write_map2(self.ekf)
            SLAM_eval.display_marker_rmse()
            self.command['output2'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, 
                                position=(h_pad, 240+2*v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [2, 0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-2, 0] 
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 2] 
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -2] 
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
            # output RMSE during run
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                self.command['output2'] = True
        if self.quit:
            pygame.quit()
            sys.exit()
            
    # Generate paths to all fruits in search list
    def generate_path(self):
        ox, oy, a_star = astar_generate(fruit_true_pos,aruco_true_pos)
        sx, sy, gx, gy, fx, fy= generate_points(fruit_goals, aruco_true_pos,fruit_true_pos)
        
        self.waypoints_list = []
        for i in range(len(sx)):
            rx, ry = a_star.planning(sx[i], sy[i], gx[i], gy[i])
            plt.plot(rx,ry,"-r")
            rx.pop(0)
            ry.pop(0)
            temp = [[x/10.0,y/10.0] for x, y in zip(rx, ry)]
            self.waypoints_list.append(temp)
            
        # Feedback
        print(f"Path generated: {self.waypoints_list}")
        
    # Perform fruit search to all fruits in search list
    def auto_fruit_search(self, canvas):
        if self.command['auto_fruit_search']:
            if any(self.waypoints_list):
                if self.waypoints_list[0]:
                    # robot drives to the waypoint
                    self.drive_to_point(self.waypoints_list[0][0], canvas)
                    robot_pose = self.ekf.robot.state.squeeze().tolist()
                    print("Finished driving to waypoint: {}; New robot pose: {}".format(self.waypoints_list[0][0],robot_pose))
                    print()
                    
                    self.waypoints_list[0].pop(0)
                    print(f"New waypoints list: {self.waypoints_list}")

                    if not self.waypoints_list[0]:
                        self.waypoints_list.pop(0)
                        print(f"Remaining fruit goals before: {self.fruit_goals_remain}")
                        self.fruit_goals_remain = np.delete(self.fruit_goals_remain, 0, axis=0)
                        print(f"Remaining fruit goals after: {self.fruit_goals_remain}")
                        print("Fruit reached, robot sleeps for 3 seconds")
                        time.sleep(3)

                    self.command['output2'] = True
                    self.record_data()

                    self.count_rot=self.count_rot+1
                    if self.count_rot==4:
                        self.rotate_robot(num_turns=12)
                        self.count_rot=0
            else:
                print("Waypoints list is empty")
                self.waypoints_list = []
                self.command['auto_fruit_search'] = False
    
    # Detect fruits and update path
    def fruit_detect_update_test(self):
        self.take_pic()
        
        # same as pressing P key
        self.detector_output, self.network_vis = self.detector.yolo_detect_single_image(self.img)
        self.file_output = (self.detector_output, self.ekf)
        self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'
        
        # same as pressin N key
        self.pred_fname = self.output.write_image(self.file_output[0], self.file_output[1])
        # self.notification = f'Prediction is saved to {operate.pred_fname}'
        
        # estimate detected fruit position
        target_est = live_fruit_pose()
        print(f"Detected fruit positions: {target_est}")
        
        update_flag = 0
        for key in target_est:
            print(key)
            print(key[:-2])
            print(fruit_list)
            if key[:-2] not in fruit_list:
                obs_fruit_x = target_est[key]['x']
                obs_fruit_y = target_est[key]['y']
                
                obs_fruit_x = np.round(obs_fruit_x, 1)
                obs_fruit_y = np.round(obs_fruit_y, 1)
                print(obs_fruit_x)
                print(obs_fruit_y)
                
                # snap to grid, not allowed by tutor
                # obs_fruit_x = round_nearest(obs_fruit_x, 0.4)
                # obs_fruit_y = round_nearest(obs_fruit_y, 0.4)
                
                obs_fruit_coord = np.array([obs_fruit_x, obs_fruit_y])
                print(obs_fruit_coord)
                if self.spoofed_obs:
                    if not (obs_fruit_coord == self.spoofed_obs).all(1).any():
                        self.spoofed_obs.append(obs_fruit_coord) # list of array
                        print(f"New obstacles detected at position: {obs_fruit_coord}")  
                        update_flag = 1
                else:
                    self.spoofed_obs.append(obs_fruit_coord) # list of array
                    print(f"New obstacles detected at position: {obs_fruit_coord}")  
                    update_flag = 1
                print(self.spoofed_obs)
        
        print(f"Update flag: {update_flag}")
        
        if update_flag:
            grid_size = 2.0  # [m], change this
            robot_radius = 1.0  # [m], change this
            spoofed_ox, spoofed_oy = generate_spoofed_obs(self.spoofed_obs)
            self.ox.extend(spoofed_ox)
            self.oy.extend(spoofed_oy)
            self.a_star = AStarPlanner(self.ox, self.oy, grid_size, robot_radius) 

            curr_pose = self.ekf.robot.state.squeeze().tolist()
            x = round_nearest(curr_pose[0], 0.2)
            y = round_nearest(curr_pose[1], 0.2)
            curr_pose = [x, y]
            
            sx, sy, gx, gy, fx, fy= generate_points(fruit_goals, aruco_true_pos,fruit_true_pos)
                
            # generate new path, continued from before meeting obstacles
            waypoints_list_new = []
            for i in range(len(sx)):
                rx, ry = self.a_star.planning(sx[i], sy[i], gx[i], gy[i])
                plt.plot(rx,ry,"-r")
                rx.pop(0)
                ry.pop(0)
                temp = [[x/10.0,y/10.0] for x, y in zip(rx, ry)]
                waypoints_list_new.append(temp)
                
            self.waypoints_list = waypoints_list_new
            self.waypoints_list[0].insert(0, curr_pose)
            # Feedback
            print(f"New path generated due to fruit: {self.waypoints_list}")
            
        return update_flag
    
    # Detect fruits and update path
    def fruit_detect_update(self):
        self.take_pic()
        
        # same as pressing P key
        self.detector_output, self.network_vis = self.detector.yolo_detect_single_image(self.img)
        self.file_output = (self.detector_output, self.ekf)
        self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'
        
        # same as pressin N key
        self.pred_fname = self.output.write_image(self.file_output[0], self.file_output[1])
        # self.notification = f'Prediction is saved to {operate.pred_fname}'
        
        # estimate detected fruit position
        target_est = live_fruit_pose()
        print(f"Detected fruit positions: {target_est}")
        
        update_flag = 0
        for key in target_est:
            if key[:-2] not in fruit_list:
                obs_fruit_x = target_est[key]['x']
                obs_fruit_y = target_est[key]['y']
                
                obs_fruit_x = np.round(obs_fruit_x, 1)
                obs_fruit_y = np.round(obs_fruit_y, 1)
                
                # snap to grid, not allowed by tutor
                # obs_fruit_x = round_nearest(obs_fruit_x, 0.4)
                # obs_fruit_y = round_nearest(obs_fruit_y, 0.4)
                
                obs_fruit_coord = np.array([obs_fruit_x, obs_fruit_y])
                if self.spoofed_obs:
                    if not (obs_fruit_coord == self.spoofed_obs).all(1).any():
                        self.spoofed_obs.append(obs_fruit_coord) # list of array
                        print(f"New obstacles detected at position: {obs_fruit_coord}")  
                        update_flag = 1
                else:
                    self.spoofed_obs.append(obs_fruit_coord) # list of array
                    print(f"New obstacles detected at position: {obs_fruit_coord}")  
                    update_flag = 1
                print(self.spoofed_obs)
        
        print(f"Update flag: {update_flag}")
            
        if update_flag:
            grid_size = 2.0  # [m], change this
            robot_radius = 1.0  # [m], change this
            spoofed_ox, spoofed_oy = generate_spoofed_obs(self.spoofed_obs)
            self.ox.extend(spoofed_ox)
            self.oy.extend(spoofed_oy)
            self.a_star = AStarPlanner(self.ox, self.oy, grid_size, robot_radius) 

            curr_pose = self.ekf.robot.state.squeeze().tolist()
            x = round_nearest(curr_pose[0], 0.2)
            y = round_nearest(curr_pose[1], 0.2)
            curr_pose = [x, y]
            
            sx, sy, gx, gy, fx, fy= generate_points(fruit_goals, aruco_true_pos,fruit_true_pos)
                
            # generate new path, continued from before meeting obstacles
            waypoints_list_new = []
            for i in range(len(sx)):
                rx, ry = self.a_star.planning(sx[i], sy[i], gx[i], gy[i])
                plt.plot(rx,ry,"-r")
                rx.pop(0)
                ry.pop(0)
                temp = [[x/10.0,y/10.0] for x, y in zip(rx, ry)]
                waypoints_list_new.append(temp)
                
            self.waypoints_list = waypoints_list_new
            self.waypoints_list[0].insert(0, curr_pose)
            # Feedback
            print(f"New path generated due to fruit: {self.waypoints_list}")
            
        return update_flag
    
    # drive to a waypoint from current position
    def drive_to_point(self, waypoint, canvas):
        # imports camera / wheel calibration parameters 
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt(fileB, delimiter=',')
        
        wheel_vel = 20 # tick to move the robot
        
        # compute x and y distance to waypoint
        robot_pose = self.ekf.robot.state.squeeze().tolist()
        x_diff = waypoint[0] - robot_pose[0]
        y_diff = waypoint[1] - robot_pose[1]
        
        # wrap robot orientation to (-pi, pi]
        robot_orient = (robot_pose[2]) % (2*np.pi)
        if robot_orient > np.pi:
            robot_orient -= 2*np.pi
        
        # compute minimum turning angle to waypoint
        turn_diff = np.arctan2(y_diff, x_diff) - robot_orient
        if turn_diff > np.pi:
            turn_diff -= 2*np.pi
        elif turn_diff < -np.pi:
            turn_diff += 2*np.pi
        
        turn_time = 0.0
        if turn_diff > 0.0: # turn left
            turn_time = (abs(turn_diff)*baseline)/(2.0*scale*wheel_vel)
            lv, rv = self.pibot.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
            turn_drive_meas = measure.Drive(lv, rv, turn_time)
            
            time.sleep(0.5)
            self.take_pic()
            self.update_slam(turn_drive_meas)
            self.waypoint_update()

        elif turn_diff < 0.0: # turn right
            turn_time = (abs(turn_diff)*baseline*1.06)/(2.0*scale*wheel_vel)
            lv, rv = self.pibot.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
            turn_drive_meas = measure.Drive(lv, rv, turn_time)
            
            time.sleep(0.5)
            self.take_pic()
            self.update_slam(turn_drive_meas)
            self.waypoint_update()
            
        print("Turning for {:.2f} seconds".format(turn_time))
        print(f"Position: {operate.ekf.robot.state.squeeze().tolist()}")
            
        # update pygame display
        self.draw(canvas)
        pygame.display.update()
        
        # compute driving distance to waypoint
        pos_diff = np.hypot(x_diff, y_diff)
        
        drive_time = 0.0
        if pos_diff > 0.0:
            # detect if fruit is in path before driving straight
            # update = self.fruit_detect_update()
            # if update:
            #     return
            
            # after turning, drive straight to the waypoint
            drive_time = pos_diff/(scale*wheel_vel)
            lv, rv = self.pibot.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
            lin_drive_meas = measure.Drive(lv, rv, drive_time)
            
            time.sleep(0.5)
            self.take_pic()
            self.update_slam(lin_drive_meas)
            self.waypoint_update()
            
        print("Driving for {:.2f} seconds".format(drive_time))
        print(f"Position: {operate.ekf.robot.state.squeeze().tolist()}")
        
        # update pygame display
        self.draw(canvas)
        pygame.display.update()

        print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
        
    def waypoint_update(self, steps=3):
        for _ in range(steps):
            self.take_pic()
            lv, rv = self.pibot.set_velocity([0, 0], tick=0.0, time=0.0)
            drive_meas = measure.Drive(lv, rv, 0.0)
            self.update_slam(drive_meas)

            # update pygame display
            self.draw(canvas)
            pygame.display.update()
        
    # rotate robot to scan landmarks
    def rotate_robot(self, num_turns=8):
        # imports camera / wheel calibration parameters 
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt(fileB, delimiter=',')
        
        wheel_vel = 20 # tick to move the robot
        
        turn_resolution = 2*np.pi/num_turns
        if(num_turns==8):
            turn_offset=0.024
        elif num_turns==12:
            turn_offset=0.03

        turn_time = (abs(turn_resolution)*baseline)/(2.0*scale*wheel_vel) + turn_offset
        
        for _ in range(num_turns):
            lv, rv = self.pibot.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
            turn_drive_meas = measure.Drive(lv, rv, turn_time-turn_offset)
            
            time.sleep(0.5)
            self.take_pic()
            self.update_slam(turn_drive_meas)

            # update pygame display
            self.draw(canvas)
            pygame.display.update()
            
            self.waypoint_update()

            print(f"Position rotate: {self.ekf.robot.state.squeeze().tolist()}")
            
    
    # Keyboard control for Milestone 4 Level 2
    def update_keyboard_L2(self):
        for event in pygame.event.get():
            # run SLAM
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:

                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'

                # read in the true map
                # fruit_list, fruit_true_pos, aruco_true_pos = read_true_map('M4_true_map.txt')
                lms = []
                for i,lm in enumerate(aruco_true_pos):
                    measure_lm = measure.Marker(np.array([[lm[0]],[lm[1]]]),i+1, covariance=(0.0001*np.eye(2)))
                    lms.append(measure_lm)
                self.ekf.add_landmarks_init(lms)  

            # run path planning algorithm
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.generate_path()
                
            # drive to waypoints
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                # if any(self.waypoints_list):
                #      self.rotate_robot(num_turns=8)
                self.command['auto_fruit_search'] = True
                
                    
            # reset path planning algorithm
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.waypoints_list = []
                self.ekf.reset()
                
                # read in the true map
                # fruit_list, fruit_true_pos, aruco_true_pos = read_true_map('M4_true_map.txt')
                lms = []
                for i,lm in enumerate(aruco_true_pos):
                    measure_lm = measure.Marker(np.array([[lm[0]],[lm[1]]]),i+1, covariance=(0.0001*np.eye(2)))
                    lms.append(measure_lm)
                self.ekf.add_landmarks_init(lms)   
                
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
                
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
                
                
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True

        if self.quit:
            pygame.quit()
            sys.exit()
            
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    args, _ = parser.parse_known_args()
    
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
    fruit_list, fruit_true_pos, aruco_true_pos = read_true_map('aruco_fruit_final.txt')
    search_list = read_search_list()
    fruit_goals = print_target_fruits_pos(search_list, fruit_list, fruit_true_pos)
    operate.fruit_goals_remain = fruit_goals
    
    while start:
        operate.update_keyboard_L2()
        
        # take latest pic and update slam
        operate.take_pic()
        lv, rv = operate.pibot.set_velocity([0, 0], tick=0.0, time=0.0)
        drive_meas = measure.Drive(lv, rv, 0.0)
        operate.update_slam(drive_meas)
        
        # detector testing if overwrite
        operate.detect_target()
        operate.record_data()
        
        # update pygame display
        operate.draw(canvas)
        pygame.display.update()
        
        # perform fruit search
        operate.auto_fruit_search(canvas)
        angle = operate.ekf.robot.state[2][0]
        angle = angle*180/np.pi
        angle = angle % 360
        #print(f"Position_rad: {operate.ekf.robot.state.squeeze().tolist()}")
        print(f"Position: {operate.ekf.robot.state[0][0]},{operate.ekf.robot.state[1][0]},{angle}")

