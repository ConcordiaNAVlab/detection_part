#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: yolov8_detail.py
# Author: Linhan Qiao <qiaolinhan073@gmail.com>
# Date: 2024-06-10
# Last Modified By: Linhan Qiao <qiaolinhan073@gmail.com>
# -----

# MIT license

from ultralytics import YOLO

model = YOLO('./test.pt')
# print(model.info(detailed = True))
# print('----------------------')
# print(dir(model))
# print('----------------------')
# print(model)
result = model('./test.jpg')
