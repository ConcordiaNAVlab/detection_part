import ultralytics
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import torchvision.transforms as transforms
from PIL import Image
import io

from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image

# import matplotlib.pyplot as plt

model = YOLO('./snow.pt')
target_layers =[model.model.model[-2]]
cam = EigenCAM(model, target_layers,task='od')

######################################
# img = cv2.imread('./snow0.png')
# img = cv2.resize(img, (640, 640))
# rgb_img = img.copy()
# img = np.float32(img) / 255
#
# grayscale_cam = cam(rgb_img)[0, :, :]

# cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
#
# plt.imshow(cam_image)
# plt.show()
######################################
video_path = "/Users/qiaolinhan/dev/detection_part/datasets/02_iphone6.MOV"
vs = cv2.VideoCapture(video_path)
_, frame = vs.read()
H, W, _ = frame.shape
vs.release()

fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
out = cv2.VideoWriter("output.avi", fourcc, 10, (W, H), True)

cap = cv2.VideoCapture(video_path)


# idx = 0
while True:
    ret, frame = cap.read()

    if ret == False:
        print("[ERROR] :: Did not load the video")
        cap.release()
        out.release()
        break

    cv2.imshow("frame", frame)

    ori_frame = frame.copy()
    # frame = cv2.resize(frame, (640, 640))
    frame = frame.astype(np.float32)
    # frame = np.expand_dims(frame, axis = 0)
    frame = frame / 255.0

    # frame = np.float32(frame) / 255
    grayscale_cam = cam(ori_frame)[0, :, :]
    cam_img = show_cam_on_image(frame, grayscale_cam, use_rgb = True)
    # cam_img = cam_img.astype(np.float32)
    # cam_img = cv2.resize(cam_img, (W, H))
    cv2.imshow("cam", cam_img)
    # cv2.imwrite(f"video/{idx}.png", cam_img)
    # idx += 1
    out.write(cam_img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
