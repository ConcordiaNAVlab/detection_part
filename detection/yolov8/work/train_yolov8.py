# nvidia-smi
import os
from ultralytics import YOLO

model=YOLO('yolov8l.pt')
results = model.predict(source='https://media.roboflow.com/notebooks/examples/dog.jpeg', conf=0.25')

print(results[0].boxes.xyxy)
