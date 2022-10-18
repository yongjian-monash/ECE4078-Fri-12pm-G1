import os 
import time

import cmd_printer
import numpy as np
import torch
from args import args
from res18_skip import Resnet18Skip
from torchvision import transforms
import cv2

import torch

class Detector:
    def __init__(self, ckpt, use_gpu=False):
        self.args = args
        self.model = Resnet18Skip(args)
        if torch.cuda.torch.cuda.device_count() > 0 and use_gpu:
            self.use_gpu = True
            self.model = self.model.cuda()
        else:
            self.use_gpu = False
        # self.load_weights(ckpt)
        self.model = self.model.eval()
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights.pt')
        self.yolo_model.conf = 0.60 # yolo confidence threshold
        cmd_printer.divider(text="warning")
        print('This detector uses "RGB" input convention by default')
        print('If you are using Opencv, the image is likely to be in "BRG"!!!')
        cmd_printer.divider()
        self.colour_code = np.array([(220, 220, 220), (128, 0, 0), (155, 255, 70), (255, 85, 0), (255, 180, 0), (0, 128, 0)])
        # color of background, redapple, greenapple, orange, mango, capsicum

    def detect_single_image(self, np_img):
        torch_img = self.np_img2torch(np_img)
        tick = time.time()
        with torch.no_grad():
            pred = self.model.forward(torch_img)
            if self.use_gpu:
                pred = torch.argmax(pred.squeeze(),
                                    dim=0).detach().cpu().numpy()
            else:
                pred = torch.argmax(pred.squeeze(), dim=0).detach().numpy()
        dt = time.time() - tick
        print(f'Inference Time {dt:.2f}s, approx {1/dt:.2f}fps', end="\r")
        colour_map = self.visualise_output(pred)
        return pred, colour_map
        
    def yolo_detect_single_image(self, img):
        tick = time.time()
        results = self.yolo_model(img, size=640)
        dt = time.time() - tick
        # print(f'Inference Time {dt:.2f}s, approx {1/dt:.2f}fps', end="\r")
        
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
            cv2.rectangle(pred, p1, p2, obj_class+1, -1)
        #########################################################################
        
        # colour_map = self.visualise_output(pred)
        
        return pred, bbox_img # original yolov5 output
        # return pred, colour_map
    
    def visualise_output(self, nn_output):
        r = np.zeros_like(nn_output).astype(np.uint8)
        g = np.zeros_like(nn_output).astype(np.uint8)
        b = np.zeros_like(nn_output).astype(np.uint8)
        for class_idx in range(0, self.args.n_classes + 1):
            idx = nn_output == class_idx
            r[idx] = self.colour_code[class_idx, 0]
            g[idx] = self.colour_code[class_idx, 1]
            b[idx] = self.colour_code[class_idx, 2]
        colour_map = np.stack([r, g, b], axis=2)
        colour_map = cv2.resize(colour_map, (320, 240), cv2.INTER_NEAREST)
        w, h = 10, 10
        pt = (10, 160)
        pad = 5
        labels = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
        font = cv2.FONT_HERSHEY_SIMPLEX 
        for i in range(1, self.args.n_classes + 1):
            c = self.colour_code[i]
            colour_map = cv2.rectangle(colour_map, pt, (pt[0]+w, pt[1]+h),
                            (int(c[0]), int(c[1]), int(c[2])), thickness=-1)
            colour_map  = cv2.putText(colour_map, labels[i-1],
            (pt[0]+w+pad, pt[1]+h-1), font, 0.4, (0, 0, 0))
            pt = (pt[0], pt[1]+h+pad)
        return colour_map

    # def load_weights(self, ckpt_path):
        # ckpt_exists = os.path.exists(ckpt_path)
        # if ckpt_exists:
            # ckpt = torch.load(ckpt_path,
                              # map_location=lambda storage, loc: storage)
            # self.model.load_state_dict(ckpt['weights'])
        # else:
            # print(f'checkpoint not found, weights are randomly initialised')
            
    @staticmethod
    def np_img2torch(np_img, use_gpu=False, _size=(192, 256)):
        preprocess = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=_size),
                                        # transforms.ColorJitter(brightness=0.4, contrast=0.3,
                                        #                         saturation=0.3, hue=0.05),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        img = preprocess(np_img)
        img = img.unsqueeze(0)
        if use_gpu:
            img = img.cuda()
        return img