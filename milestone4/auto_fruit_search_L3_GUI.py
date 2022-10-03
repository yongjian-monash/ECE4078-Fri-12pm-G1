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
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector

# custom added
import SLAM_eval
from path_planning import *
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
        self.count_down = 300
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
        
        # for auto navigation
        self.waypoints_list = []
        self.spoofed_obs = []

    # # wheel control
    # def control(self):       
        # if args.play_data:
            # lv, rv = self.pibot.set_velocity()            
        # else:
            # lv, rv = self.pibot.set_velocity(
                # self.command['motion'])
        # if not self.data is None:
            # self.data.write_keyboard(lv, rv)
        # dt = time.time() - self.control_clock
        # drive_meas = measure.Drive(lv, rv, dt)
        # self.control_clock = time.time()
        # return drive_meas
        
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
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False
        # custom function
        if self.command['output2']:
            # self.output.write_map2(self.ekf)
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
    # def update_keyboard(self):
        # for event in pygame.event.get():
            # # drive forward
            # if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                # self.command['motion'] = [2, 0]
            # # drive backward
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                # self.command['motion'] = [-2, 0] 
            # # turn left
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                # self.command['motion'] = [0, 2] 
            # # drive right
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                # self.command['motion'] = [0, -2] 
            # # stop
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # self.command['motion'] = [0, 0]
            # # save image
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                # self.command['save_image'] = True
            # # save SLAM map
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                # self.command['output'] = True
            # # reset SLAM map
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                # if self.double_reset_comfirm == 0:
                    # self.notification = 'Press again to confirm CLEAR MAP'
                    # self.double_reset_comfirm +=1
                # elif self.double_reset_comfirm == 1:
                    # self.notification = 'SLAM Map is cleared'
                    # self.double_reset_comfirm = 0
                    # self.ekf.reset()
            # # run SLAM
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                # n_observed_markers = len(self.ekf.taglist)
                # if n_observed_markers == 0:
                    # if not self.ekf_on:
                        # self.notification = 'SLAM is running'
                        # self.ekf_on = True
                    # else:
                        # self.notification = '> 2 landmarks is required for pausing'
                # elif n_observed_markers < 3:
                    # self.notification = '> 2 landmarks is required for pausing'
                # else:
                    # if not self.ekf_on:
                        # self.request_recover_robot = True
                    # self.ekf_on = not self.ekf_on
                    # if self.ekf_on:
                        # self.notification = 'SLAM is running'
                    # else:
                        # self.notification = 'SLAM is paused'
            # # run object detector
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                # self.command['inference'] = True
            # # save object detection outputs
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                # self.command['save_inference'] = True
            # # quit
            # elif event.type == pygame.QUIT:
                # self.quit = True
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                # self.quit = True
            # # output RMSE during run
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                # self.command['output2'] = True
        # if self.quit:
            # pygame.quit()
            # sys.exit()
            
    def round_nearest(x, a):
        return round(round(x / a) * a, -int(math.floor(math.log10(a))))
            
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
            
    def live_detect_update(self, fruit_list):
        if any(self.waypoints_list):
            self.take_pic()
            self.detector_output, self.network_vis = self.detector.yolo_detect_single_image(self.img)
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'
            self.pred_fname = self.output.write_image(self.file_output[0], self.file_output[1])
            # self.notification = f'Prediction is saved to {operate.pred_fname}'
            target_est = live_fruit_pose()
            
            update_flag = 0
            for key in target_est:
                if key[:-2] not in fruit_list:
                    obs_fruit_x = target_est[key]['x']
                    obs_fruit_y = target_est[key]['y']
                    obs_fruit_x = round_nearest(obs_fruit_x, 0.4)
                    obs_fruit_y = round_nearest(obs_fruit_y, 0.4)
                    self.spoofed_obs.append([obs_fruit_x, obs_fruit_y])
                    print("new obstacles detected:{}" .format([obs_fruit_x, obs_fruit_y]))  
                    update_flag = 1
                    
            if update_flag:
                spoofed_ox, spoofed_oy = gen_cor.generate_spoofed_obs(self.spoofed_obs)
                # sx, sy, gx, gy, fx, fy, ox, oy, face_angle = gen_cor.generate_points(self.spoofed_obs)
                
                sx = []
                sy = []
                gx = []
                gy = []
                
                # update latest starting and goal positions so robot does not revisit reached goal
                for waypoint in self.waypoints_list:
                    # self.waypoints_list is a list of lists of lists
                    # outer list is list of [[x1,y1], [x2,y2], ...] points to go from start to goal
                    # inner list is [x, y] 
                    sx.append(waypoint[0][0])
                    sy.append(waypoint[0][1])
                    gx.append(waypoint[-1][0])
                    gy.append(waypoint[-1][1])
                    
                # generate new path, continued from before meeting obstacles
                waypoints_list_new = []
                for i in range(len(sx)):
                    _, pathx, pathy = dstarlite.main(Node(x=sx[i], y=sy[i]), Node(x=gx[i], y=gy[i]), spoofed_ox=spoofed_ox, spoofed_oy=spoofed_oy)
                    pathx.pop(0)
                    pathy.pop(0)
                    temp = [[x/10.0,y/10.0] for x, y in zip(pathx, pathy)]
                    waypoints_list_new.append(temp)
                self.waypoints_list = waypoints_list_new

    # Waypoint navigation
    # the robot automatically drives to a given [x,y] coordinate
    def drive_to_point(self, waypoint, canvas, fruit_list):
        # imports camera / wheel calibration parameters 
        fileS = "calibration/param/scale.txt"
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = "calibration/param/baseline.txt"
        baseline = np.loadtxt(fileB, delimiter=',')
        
        ####################################################
        wheel_vel = 20 # tick to move the robot
        
        # compute x and y distance to waypoint
        robot_pose = self.ekf.robot.state.squeeze().tolist()
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
        
        if turn_diff > 0.0: # turn left
            lv, rv = self.pibot.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
            turn_drive_meas = measure.Drive(lv, rv, turn_time)
            self.update_slam(turn_drive_meas)
        elif turn_diff < 0.0: # turn right
            lv, rv = self.pibot.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
            turn_drive_meas = measure.Drive(lv, rv, turn_time)
            self.update_slam(turn_drive_meas)
        
        # take latest pic and update slam
        for _ in range(3):
            lv, rv = self.pibot.set_velocity([0, 0], tick=0.0, time=0.0)
            drive_meas = measure.Drive(lv, rv, 0.0)
            self.update_slam(drive_meas)
        
        # live_detect_update(self, fruit_list)
        
        # update pygame display
        self.draw(canvas)
        pygame.display.update()
        
        # compute driving distance to waypoint
        pos_diff = np.hypot(x_diff, y_diff)
        
        # after turning, drive straight to the waypoint
        drive_time = 0.0 # replace with your calculation
        drive_time = pos_diff/(scale*wheel_vel)
        print("Driving for {:.2f} seconds".format(drive_time))
        
        if pos_diff > 0.0
            lv, rv = self.pibot.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
            lin_drive_meas = measure.Drive(lv, rv, drive_time)
            self.update_slam(lin_drive_meas)
        
        # take latest pic and update slam
        for _ in range(3):
            lv, rv = self.pibot.set_velocity([0, 0], tick=0.0, time=0.0)
            drive_meas = measure.Drive(lv, rv, 0.0)
            self.update_slam(drive_meas)
        
        # live_detect_update(self, fruit_list)
        
        # update pygame display
        self.draw(canvas)
        pygame.display.update()
        ####################################################

        print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
            
    def generate_path(self):
        global gen_cor
        gen_cor = GenerateCoord('M4_true_map.txt')
        # fruit_list, _, _ = gen_cor.read_true_map()
        # obs_fruit_list = []
        spoofed_obs = []
        spoofed_ox = [[],[],[],[]]
        spoofed_oy = [[],[],[],[]]
        # prev_len = len(spoofed_obs)

        sx, sy, gx, gy, fx, fy, ox, oy, face_angle = gen_cor.generate_points(spoofed_obs)
        global dstarlite
        dstarlite = DStarLite(ox, oy)
        
        self.waypoints_list = []
        for i in range(len(sx)):
            _, pathx, pathy = dstarlite.main(Node(x=sx[i], y=sy[i]), Node(x=gx[i], y=gy[i]), spoofed_ox=spoofed_ox, spoofed_oy=spoofed_oy)
            pathx.pop(0)
            pathy.pop(0)
            temp = [[x/10.0,y/10.0] for x, y in zip(pathx, pathy)]
            self.waypoints_list.append(temp)
        print(self.waypoints_list)
        print("Path generated")
        
    def auto_fruit_search(self, canvas, fruit_list):
        if self.command['auto_fruit_search']:
            if any(self.waypoints_list):
                if self.waypoints_list[0]:
                    # robot drives to the waypoint
                    drive_to_point(self, self.waypoints_list[0][0], canvas, fruit_list)
                    robot_pose = self.ekf.robot.state.squeeze().tolist()
                    print("Finished driving to waypoint: {}; New robot pose: {}".format(self.waypoints_list[0][0],robot_pose))
                    
                    self.waypoints_list[0].pop(0)
                    print(self.waypoints_list)

                    if not self.waypoints_list[0]:
                        self.waypoints_list.pop(0)
                        print("Sleep 3 seconds")
                        time.sleep(3)
            else:
                print("Waypoints list is empty")
                self.waypoints_list = []
                self.spoofed_obs = []
                self.command['auto_fruit_search'] = False
            
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
                _, _, aruco_true_pos = read_true_map('M4_true_map.txt')
                lms = []
                for i,lm in enumerate(aruco_true_pos):
                    measure_lm = measure.Marker(np.array([[lm[0]],[lm[1]]]),i+1, covariance=(0.0*np.eye(2)))
                    lms.append(measure_lm)
                self.ekf.add_landmarks_init(lms)   
                
            # run path planning algorithm
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.generate_path()
                
            # drive to waypoints
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                self.command['auto_fruit_search'] = True
                    
            # reset path planning algorithm
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.waypoints_list = []
                self.spoofed_obs = []
                self.ekf.reset()
                
                # read in the true map
                _, _, aruco_true_pos = read_true_map('M4_true_map.txt')
                lms = []
                for i,lm in enumerate(aruco_true_pos):
                    measure_lm = measure.Marker(np.array([[lm[0]],[lm[1]]]),i+1, covariance=(0.0*np.eye(2)))
                    lms.append(measure_lm)
                self.ekf.add_landmarks_init(lms)   
                
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
    robot_pose = [0.0,0.0,0.0]
    # read in the true map
    fruit_list, _, _ = read_true_map('M4_true_map.txt')

    while start:
        operate.update_keyboard_L2()
        
        # take latest pic and update slam
        operate.take_pic()
        lv, rv = operate.pibot.set_velocity([0, 0], tick=0.0, time=0.0)
        drive_meas = measure.Drive(lv, rv, 0.0)
        operate.update_slam(drive_meas)
        
        # update pygame display
        operate.draw(canvas)
        pygame.display.update()
        
        # perform fruit search
        operate.auto_fruit_search(canvas, fruit_list)
        

