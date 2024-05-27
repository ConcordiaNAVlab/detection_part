#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: yolov8_track_overtime.py
# Author: Linhan Qiao <qiaolinhan073@gmail.com>
# Date: 2024-03-27
# Last Modified By: Linhan Qiao <qiaolinhan073@gmail.com>
# -----
# MIT license

from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Loadingthe Nano version weights
model = YOLO('best.pt')
# video_path = "https://youtube.com/shorts/14fewMP8suc?si=Ah1QBJJ7zoWlWPN_"

# model = YOLO('yolov8n.pt')
video_path = 0
conf_level = 0.75
cap = cv2.VideoCapture(video_path)

# To check if the video source is loaded correctly
if (cap.isOpened() == False):  
    print("Error reading video file")

# Acquire the frame size and fps
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
fps = cap.get(cv2.CAP_PROP_FPS)
# VideoWriter to save the video
output = cv2.VideoWriter(
        "tracking_result.mp4",
        cv2.VideoWriter_fourcc(*'MP4V'),
        fps, size)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        print('[INFO]: The image size:', size)
        print('[INFO]: The FPS:', fps)
        # # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # results = model.track(frame, persist=True)
        results = model.track(frame, persist=True, conf=conf_level)

        # # Add the track parameters which ultralytics
        # # supports
        results = model.track(frame, persist=True,
                              conf=conf_level, 
                              tracker = 'botsort.yaml')
        # results = model.track(frame, persist=True,
        #                       conf=conf_level, 
        #                       tracker = 'bytetrack.yaml')

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Printing the FPS on the screen
        annotated_frame = cv2.putText(annotated_frame,
                                      f'FPS: {fps:.2f}',
                                      (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      1, (0, 0, 255), 2, cv2.LINE_AA)
        # annotated_frame = results[0].plot(labels = False)

        # print(results)
        if results[0].boxes.id != None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy().astype(int)

        # original loop, when detected the object
        # else:
            # Get the boxes and track IDs
            # boxes = results[0].boxes.xywh.cpu()
            # track_ids = results[0].boxes.id.int().cpu().tolist()
            # track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                print('[INFO]:', 'x:', x, 'y:', y, 'w:', w,
                      'h:', h)
                track = track_history[track_id]

                # track.append((float(x), float(y)))  # x, y center point

                center_x = float((x + w)/2)
                center_y = float((y + h)/2)
                # Appending the central points
                track.append((center_x, center_y))  # x, y center point

                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                # Printing the tracking points
                # information
                print('[INFO]: The points:', points)
                cv2.polylines(annotated_frame, 
                              [points],
                              isClosed=False, 
                              color=(255, 0, 255),
                              thickness=2)

        # Save and display the annotated frame
        output.write(annotated_frame)
        cv2.imshow("YOLOv8_Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
