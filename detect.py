import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class ObjectDetection:
    def __init__(self):
        # default parameters
        self.email_sent = False
        self.model = YOLO("yolov8n.pt")
        self.annotator = None
        self.start_time = 0
        self.end_time = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self, im):
        results = self.model(im)
        return results

    def display_fps(self, im):
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imshow('Image', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_bboxes(self, results, im):
        class_ids = []
        self.annotator = Annotator(im, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return im, class_ids

    def detect_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened()
        frame_count = 0
        while True:
            self.start_time = time()
            ret, im = cap.read()
            if not ret:
                break
            results = self.predict(im)
            im, class_ids = self.plot_bboxes(results, im)

            self.display_fps(im)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def detect_image(self, image_path):
        im = cv2.imread(image_path)
        results = self.predict(im)
        im, class_ids = self.plot_bboxes(results, im)
        self.display_fps(im)

# Create an instance of ObjectDetection
detector = ObjectDetection()

# Path to the video file
#video_path = 'path/to/your/video.mp4'  # Replace with your video file path

# Perform object detection on a video
#detector.detect_video(video_path)

# Path to the image file
image_path = 'poocha.jpg'  # Replace with your image file path

# Perform object detection on an image
detector.detect_image(image_path)
