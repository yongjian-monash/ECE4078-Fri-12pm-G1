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
import SLAM_eval_M5
from TargetPoseEst import live_fruit_pose_M5

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
                        'evaluate': False,
                        'save_fruits': False
                        'offset_mode': False
                        'offset_plus': False
                        'offset_minus': False}
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
        
        self.offset_mode = 0 # 1 - rotation, 2 - translation x, 3 - translation y
        self.offset_rot = 0
        self.offset_trans_x = 0
        self.offset_trans_y = 0

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
            self.current_robot_pose = self.getting_robot_pose()
            self.current_slam_pose = self.ekf.robot.state
            SLAM_eval_M5.givecoord_test(self.ekf.robot.state,self.current_robot_pose)

            self.update_plot(canvas,'pics/test_plot_markers.png')
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
            self.output.write_map2(self.ekf)
            SLAM_eval_M5.display_marker_rmse(self.ekf.robot.state)
            # self.ekf.robot.state=0
            self.command['output2'] = False

        # if self.command['evaluate']:
            # #SLAM_eval_M5.givecoord(self.ekf.robot.state)  #use this for final demo
            # SLAM_eval_M5.givecoord_test(self.ekf.robot.state)
            # self.update_plot(canvas,'pics/test_plot_markers.png')
            # self.command['evaluate'] = False

        if self.command['save_fruits']:
            live_fruit_pose_M5()
            self.update_plot(canvas,'pics/test_plot_fruits.png')
            self.command['save_fruits'] = False
            
        if self.command['offset_mode']: # toggle offset mode
            self.offset_mode += 1
            if self.offset_mode > 3:
                self.offset_mode = 1
                
            if self.offset_mode = 1:
                self.notification = f'Rotation offset'
            else if self.offset_mode = 2:
                self.notification = f'Trans_x offset'
            else if self.offset_mode = 3:
                self.notification = f'Trans_y offset'
                
            self.command['offset_mode'] = False
                
        if self.command['offset_plus']: # increase offset
            if self.offset_mode = 1:
                self.offset_rot += 5*np.pi/180
            else if self.offset_mode = 2:
                self.trans_x += 0.1
            else if self.offset_mode = 3:
                self.trans_y += 0.1
                
            SLAM_eval_M5.givecoord_test(self.current_slam_pose, self.current_robot_pose, offset_rot = self.offset_rot, offset_x = self.offset_trans_x, offset_y = self.offset_trans_y)
            self.update_plot(canvas,'pics/test_plot_markers.png')
    
            self.command['offset_plus'] = False
                
        if self.command['offset_minus']: # decrease offset
            if self.offset_mode = 1:
                self.offset_rot -= 5*np.pi/180
            else if self.offset_mode = 2:
                self.trans_x -= 0.1
            else if self.offset_mode = 3:
                self.trans_y -= 0.1
                
            SLAM_eval_M5.givecoord_test(self.current_slam_pose, self.current_robot_pose, offset_rot = self.offset_rot, offset_x = self.offset_trans_x, offset_y = self.offset_trans_y)
            self.update_plot(canvas,'pics/test_plot_markers.png')
                
            self.command['offset_minus'] = False


    def getting_robot_pose(self):
        while True:
            x,y,rad = 0.0,0.0,0.0
            x = input("X coordinate of the robot at final position: ")
            try:
                x = float(x)
            except ValueError:
                print("Please enter a number.")
                continue
            y = input("Y coordinate of the robot at final position: ")
            try:
                y = float(y)
            except ValueError:
                print("Please enter a number.")
                continue
            deg = input("Angle (degrees) of the robot at final position: ")
            try:
                deg = float(deg)
                rad = deg*np.pi/180
            except ValueError:
                print("Please enter a number.")
                continue
            break
        robot_pose = np.hstack([x, y, rad])
        return robot_pose

    # paint the GUI
    def update_plot(self,canvas,file_name):
        v_pad = 40
        h_pad = 20        
        plot_markers = cv2.resize(cv2.imread(f'{file_name}'),(640,480))
        plot_markers = cv2.cvtColor(plot_markers,cv2.COLOR_BGR2RGB)
        plot_markers = pygame.surfarray.make_surface(np.rot90(plot_markers))
        plot_markers = pygame.transform.flip(plot_markers,True,False)
        canvas.blit(plot_markers,(2*h_pad+320+350,60))

        return canvas

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

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                self.command['evaluate'] = True

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                self.command['save_fruits'] = True

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
            # # output RMSE during run
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                # self.command['output2'] = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_z:
                self.command['offset_mode'] = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_x:
                self.command['offset_plus'] = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                self.command['offset_minus'] = True
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
    
    width, height = 1380, 660
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

    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        print(f"Position: {operate.ekf.robot.state.squeeze().tolist()}")




