#!/bin/bash


# ------------------------------------------------------------------------------
#
#   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
#
#   @Filename: runs.sh
#
#   @Author: Linhan Qiao
#
#   @Date: 2023-10-16
#
#   @Email: q_linhan.live@concordia.ca
#
#   @Description: trying to create packages and nodes for deployment
#
# ------------------------------------------------------------------------------
cd src
catkin_create_pkg yolo_pkg rospy std_msgs
cd ..

alias python='python3/home/qiao/dev/giao/works/detecting/segmentation/yolov8/coordinate.py'

