#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: m2_train_check.py
# Author: Linhan Qiao <qiaolinhan073@gmail.com>
# Date: 2024-03-28
# Last Modified By: Linhan Qiao <qiaolinhan073@gmail.com>
# -----
# MIT license

import os
from ultralytics import YOLO
import torch

print(torch.backends.mps.is_available())

model = YOLO('yolov8n-seg.pt')

# model.train(data = 'coco128-seg.yaml', imgszs=640,
            # epochs=1,)
model.train(data = 'cofig.yaml', imgszs=640,
            epochs=10, device='mps')

