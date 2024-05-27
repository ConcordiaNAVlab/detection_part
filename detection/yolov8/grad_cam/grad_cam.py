#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: grad_cam.py
# Author: Linhan Qiao <qiaolinhan073@gmail.com>
# Date: 2024-05-26
# Last Modified By: Linhan Qiao <qiaolinhan073@gmail.com>
# -----
# MIT license

import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
from matplotlib import pyplot as plt

# qiao20240526: For python script only, I use PIL to input images, maybe there is a
# different format in ROS
from PIL import Image

# To load yolov8
import ultralytics.YOLO 

# define preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# load an image and process it
img = Image.open('').convert('RGB')
# Add batch dimension
input_tensor = preprocess(img).unsqueeze(0)

# load the model
model = YOLO()
# eval mode
model.eval()

# Process the img to get output
output = model(input_tensor)

# Assuming the output is a list of detections, we need to pick one to visualize
# Chhoose the first detection for simplicity
target_detection = output[0]
target_class = target_detection['class']

# qiao: THE KEY: Hook to the gradient of the target layer
gradients = []

def save_gradient(grad):
    gradients.append(grad)

# Assuming the target layer is the last conv layer, get it from the model
target_layer = model.model.layer[-1]
target_layer.register_backward_hook(lambda module, gard_in, grad_out:
                                    save_gradient(grad_out[0]))

# Zero gradeints
model.zero_grad()

# Backward pass for the target class
class_idx = target_class.item()
# Assuming 'score' is the confidence score of the detection
score = output[0]['score']
# qiao20240526: What is retain_graoh?
score.backward(retain_graph = True)

# Get the gradeints
gradients = gradients[0].cpu().data.numpy()

# Get the activation of the target layer
activations = target_layer(input_tensor).detach().cpu().data.numpy()



