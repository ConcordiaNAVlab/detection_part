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

# qiao: need to be modified
from utils import GradCAM, show_cam_on_image

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


    # pointing out the class which we are interested
    target_category = 0

    # Pre-process of the image, in data_transform
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transform.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    #################################
    # load the image
    # qiao: possible to covert this part as openCV version, PIL version used here
    image_path = input("Please input the image_path")
    assert os.path.exists(img_path, "[ERROR] :: file '{}' dose not exist.".format(img_path))
    img = Image.open(img_path).convert('RGB')
    # converted into numpy format
    img = np.array(img, dtype = np.uint8)
    # pre-process the image data (numpy)
    img_tensor = data_transform(img)
    # adding dimension of 'batch', [C, H, W] ---> [B, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim = 0)


    ################################
    # Transfer the model, target_layer into Grad-CAM
    cam = GradCAM(model = model, target_layer = target_layer, use_cuda = False)

    target_category = 0

    grayscale_cam = cam(input_tensor = input_tensor, target_category = target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype = np.float32) / 255.,
                                      grayscale_cam, use_rgb = True)
    plt.imshow(visualization)
    plt.show()
