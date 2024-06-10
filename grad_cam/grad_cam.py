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

# To load yolov
from ultralytics import YOLO

# define preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# load an image and process it
img = Image.open('/Users/qiaolinhan/dev/datasets/trial1/frame2516.jpg').convert('RGB')

# Add batch dimension
input_tensor = img
input_tensor = preprocess(img).unsqueeze(0)
print('======> Preprocessing the image')

# load the model
model = YOLO('/Users/qiaolinhan/dev/detection_part/detection/yolov8/yolo_weights/yolov8n150_snowwork.pt')
# eval mode
# model.eval()

# Process the img to get output
output = model.predict(input_tensor)
print('======> Input and process the image')

# Assuming the output is a list of detections, we need to pick one to visualize
# Chhoose the first detection for simplicity
target_detection = output
# target_class = target_detection(classes = 0)
#
# # qiao: THE KEY: Hook to the gradient of the target layer
# gradients = []
#
# def save_gradient(grad):
#     gradients.append(grad)
#
# # Assuming the target layer is the last conv layer, get it from the model
# target_layer = model.layer[-1]
# target_layer.register_backward_hook(lambda module, gard_in, grad_out:
#                                     save_gradient(grad_out[0]))
#
# # Zero gradeints
# model.zero_grad()
#
# # Backward pass for the target class
# # class_idx = target_class.item()
# # Assuming 'score' is the confidence score of the detection
# score = output[0]['score']
# # qiao20240526: What is retain_graoh?
# score.backward(retain_graph = True)
#
# # Get the gradeints
# gradients = gradients[0].cpu().data.numpy()
#
# # Get the activation of the target layer
# activations = target_layer(input_tensor).detach().cpu().data.numpy()
#
# # Weight the channels by corresponding gradients
# # A global average pooling
# weights = np.mean(gradients, axis = (2, 3))[0, :]
#
# for i, w in enumerate(weights):
#     # the class activation maps
#     cam += w * activations[0, i, :, :]
#
# # Normalize the heatmap
# cam = np.maximum(cam, 0)
# cam = cam / cam.max()
#
# # Resize heatmap to match the input image size
# heatmap = cv2.resize(cam, (img.size[0], img.size[1]))
# heatmap = np.uint8(255 * heatmap)
#
# # Apply colormap
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#
# # Overlay the heatmap on the orignial image
# superimposed_img = heatmap * 0.2 + np.array(img)
#
# # Save or display the result
# output_path = './cam_applied_img.jpg'
# cv2.imwrite(output_path, superimposed_img)
#
# Display the result
# plt.imshow(superimposed_img)
plt.imshow(img)
plt.imshow(target_detection)
plt.axis('off')
plt.show()
