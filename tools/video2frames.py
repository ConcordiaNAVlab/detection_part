#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: Tool_VideoCutter.py
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-08-28
#
#   @Email: q_linhan@live.concordia.ca
#
#   @Description: To cut video into frame sequence
#
# ------------------------------------------------------------------------------
import cv2
import numpy as np
import os
from pathlib import Path
import glob
import sys

# For mkdir
import datetime
from pathlib import Path

now = datetime.datetime.now()
today = now.strftime("%Y%m%d")
print('[INFO] Today:', today)
Path(today).mkdir(parents=True, exist_ok=True)
# # Ubuntu path
# path = Path('home/qiao/dev/datasets/20240401/dji_T_W_Z')
# Mac path
while True:
    video_path = input('[QUE] Please input the video path:\n')
    file_path = str(Path(video_path))

    # load the video with cv2
    video = cv2.VideoCapture(file_path)

    # To check whether the video is loaded correctly
    if (video.isOpened() == False):  
        print("[ERROR] Did not read video file")
        continue
    else:
        # get the FPS information of the video, also check if we successfully loaded the video
        fps = video.get(cv2.CAP_PROP_FPS)
        print('[INFO] Fps of the video', fps)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        print('[INFO] Frame_count of the video', frame_count)
        # calculate the duration, how long the video is, through 'duration = frame_count/fps'
        duration = frame_count / fps
        print('[INFO] Frame per seconds (FPS) of this video is %d'%fps)
        print('[INFO] This video remains %d seconds'%duration)

        while True:
            ques = input('[QUE] Video loaded, please press "y" to start, or "n" to quit:\n')

            if ques == "y" or ques == "Y":
                while True:
                    ques_n = int(input('[QUES] How many frames needed? (Please no more than %d):\n'%frame_count))
                    frame_number = ques_n
                    if frame_number <= frame_count:
                        splitting_step = int(frame_count / frame_number)
                        print('[INFO] Working on it')
                        # # cut the video and save into the folder
                        # for i in range(0, frame_id):
                        for i in range(0, frame_count, splitting_step):
                            video.set(cv2.CAP_PROP_POS_FRAMES, i)
                            ret, frame = video.read()
                        #     # # Ubuntu ################
                        #     # cv2.imwrite('/home/qiao/dev/datasets/20240401/6m/infrared/%04d.png'%i, frame)
                        #     # Mac #####################
                            cv2.imwrite('%d/%04d.png'%today%i, frame)
                        break
                    elif frame_number > frame_count: 
                        print('[ERROR] Too much frames required. (Please decrease the frame number)\n')
                        continue
                    else:
                        print('[ERROR] Wrong characters input, please try an int number.\n')
                        continue

            elif ques == "n" or ques == "N":
                print("[INFO] Process stopped")
                sys.exit(0)
            else:
                continue
            break
    ########################################
    ## play the video
    #while video.isOpened():
    #    ret, frame = video.read()
    #    if not ret:
    #        print("[WARNING] Can not receive frame (stream end?).Existing ...")
    #        break
    #    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    #    cv2.imshow('frame', gray)
    #    if cv2.waitKey(1) == ord('q'):
    #        break
    #video.release()
    #cv2.destroyAllWindows()

    #######################################
    # get the frame ID at a particular time top
    # 15sec
    # hours = 00; minutes = 00; seconds = 13
    # frame_id = int(fps * (hours ** 60 + minutes * 60 + seconds))
    # print('[INFO] The specific frame_id at {}:{}:{} is {}'.format(hours, minutes, seconds, frame_id))

    ################################
    ## for single frame ############
    #video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    #ret, frame = video.read()

    ## display and save
    #cv2.imshow('frame', frame)
    #cv2.waitkey(0)
    #cv2.imwrite('saved_frame.png', frame)

    ################################
    print('[INFO] The video is cut into frames and saved in the new folder')
    break
