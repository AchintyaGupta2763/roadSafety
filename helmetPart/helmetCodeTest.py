import cv2
from ultralytics import YOLO

model = YOLO("helmetPart/runs/detect/train/weights/best.pt")

results = model.predict(source='helmetPart/video.mp4', show = True)
