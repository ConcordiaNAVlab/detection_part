#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: cam.py
# Author: Linhan Qiao <qiaolinhan073@gmail.com>
# Date: 2024-05-30
# Last Modified By: Linhan Qiao <qiaolinhan073@gmail.com>
# -----
# MIT license

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from ultralytics import YOLO
# qiao: need to be modified
from utils import GradCAM, show_cam_on_image

import cv2

def main():
    # Pointing out the model and the last layer name
    model = models.mobilenet_v3_large(pretrained = True)
    # Commonly, the last layer is picked
    # A list can be transfered, but we just transfer one
    target_layers = [model.features[-1]] 

    # model = models.vgg16(pretrained = True)
    # target_layers = [model.features]
    #
    # model = models.resnet34(pretrained = True)
    # target_layers = [model.layer4]
    #
    # model = models.regnet_y_800mf(pretrained = True)
    # target_layers = [model.trunk_output]
    #
    # model = models.efficientnet_b0(pretrained = True)
    # target_layers = [model.features]
    # weights_path = input('[INFO] :: Please input the weight path of the YOLOv8n:\n')
    # model = YOLO(weights_path)
    # target_layers = [model.model.model[-4]]
    # pointing out the class which we are interested
    # Fire Screen 556
    # Lighter 626
    target_category = 556

    # Pre-process of the image, in data_transform
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    #################################
    # load the image
    # qiao: possible to covert this part as openCV version, PIL version used here
    img_path = input("[INFO] :: Please input the image_path:\n")
    assert os.path.exists(img_path), "[ERROR] :: file '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    # converted into numpy format
    img = np.array(img, dtype = np.uint8)
    # pre-process the image data (numpy)
    img_tensor = data_transform(img)
    # adding dimension of 'batch', [C, H, W] ---> [B, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim = 0)

    # img = cv2.imread(img_path)
    # # img = cv2.resize(640, 640)
    # # rgb_img = img.copy()
    # img = np.uint8(img) / 255
    # # pre-process the image data (numpy)
    # img_tensor = data_transform(img)
    # # adding dimension of 'batch', [C, H, W] ---> [B, C, H, W]
    # input_tensor = torch.unsqueeze(img_tensor, dim = 0)

    ################################
    # Transfer the model, target_layer into Grad-CAM
    cam = GradCAM(model = model, target_layers = target_layers, use_cuda = False)

    target_category = 0

    grayscale_cam = cam(input_tensor = input_tensor, target_category = target_category)

    grayscale_cam = grayscale_cam[0, :]
    # Use the show method to draw the heat-map
    # origianl loaded image / 255 to conver the values into 0 to 1
    visualization = show_cam_on_image(img.astype(dtype = np.float32) / 255.,
                                      grayscale_cam, use_rgb = True)
    plt.imshow(visualization)
    plt.show()

if __name__ == "__main__":
    main()
