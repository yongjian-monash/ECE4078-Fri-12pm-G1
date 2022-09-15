import glob
import torch
import cv2
import numpy as np
import os, sys
import json

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
import util.DatasetHandler as dh

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot

def yolo_detect_single_image(img):
    results = yolo_model(img, size=640)
    
    ########## Retrieve yolov5 output and convert to resnet output ##########
    # results.save() # save yolov5 output image
    results.render() # render results.ims[0] to return image with bounding box
    bbox_img = results.ims[0] # image with bounding box
    detected_obj = (results.xyxy[0]).numpy()
    
    # remove duplicate
    _, indices = np.unique((detected_obj[:, 5]).astype(int), return_index=True, axis=0)
    test = detected_obj[np.sort(indices), :]
    
    # yolov5 output format = [xmin, ymin, xmax, ymax, confidence, class, name]
    # resnet output format = image of 0 for background, class number for each object
    pred = np.uint8(np.zeros((img.shape[0], img.shape[1])))
    num_obj = detected_obj.shape[0]
    for i in range(num_obj):
        p1 = (int(detected_obj[i][0]), int(detected_obj[i][1]))
        p2 = (int(detected_obj[i][2]), int(detected_obj[i][3]))
        obj_class = int(detected_obj[i][5])
        print(obj_class)
        cv2.rectangle(pred, p1, p2, obj_class+1, -1)
    #########################################################################
    
    # colour_map = visualise_output(pred)
    
    return pred, bbox_img # original yolov5 output
    # return pred, colour_map

def init_ekf(datadir, ip):
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
    
folder = "lab_output/"
img_f2 = open("lab_output/images2.txt", 'w')
image_count = 0
def write_image2(image, slam, image_count):
    img_fname = "{}pred2_{}.png".format(folder, image_count)
    # image_count += 1
    img_dict = {"pose":slam.robot.state.tolist(),
                "imgfname":img_fname}
    img_line = json.dumps(img_dict)
    img_f2.write(img_line+'\n')
    img_f2.flush()
    cv2.imwrite(img_fname, image)
    return f'pred_{image_count}.png'

yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights.pt')
yolo_model.conf = 0.60

ekf = init_ekf("calibration/param/", None)

for img in sorted(glob.glob("pibot_dataset/*.png")):
    ori_img = cv2.imread(img)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    detector_output, bbox_img = yolo_detect_single_image(ori_img)
    bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Window", bbox_img)
    # cv2.waitKey()
    pred_fname = write_image2(detector_output, ekf, image_count)
    image_count += 1

