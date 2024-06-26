#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: yolov8_seg.py
# Author: Linhan Qiao <qiaolinhan073@gmail.com>
# Date: 2024-03-28
# Last Modified By: Linhan Qiao <qiaolinhan073@gmail.com>
# -----
# MIT license

from ultralytics import YOLO
import cv2

# video_path = 0
# model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')

video_path = "../../datasets/02_iphone6.MOV"
model = YOLO('yolov8n-seg.yaml').load('yolov8n150_snowwork.pt')

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

