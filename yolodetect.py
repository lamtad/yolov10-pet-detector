from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2
import numpy as np
import datetime
from ultralytics import YOLO


class YoloDetect:
    def __init__(self, detect_class="dog", frame_width=1280, frame_height=720):
        self.model_file = "/Users/taduylam/Workspace/python/yolo-pet-detector/weights/yolov10n.pt"
        self.conf_threshold = 0.05
        self.detect_class = detect_class
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.model = YOLO(self.model_file)
        self.classes = self.load_classnames("/Users/taduylam/Workspace/python/yolo-pet-detector/weights/classnames.txt")
        self.last_alert = None
        self.alert_each = 15  # seconds

    def load_classnames(self, classnames_file):
        with open(classnames_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def draw_prediction(self, img, label, box, points):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Bounding box points
        box_points = [
            (x1, y1),
            (x2, y1),
            (x1, y2),
            (x2, y2)
        ]

        # Check if any point is inside the polygon
        if self.is_inside(points, box_points):
            img = self.alert(img)
        return self.is_inside(points, box_points)

    def alert(self, img):
        cv2.putText(img, "DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if (self.last_alert is None) or (
                (datetime.datetime.now() - self.last_alert).total_seconds() > self.alert_each):
            self.last_alert = datetime.datetime.now()
        return img

    def detect_objects(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold)
        detections = results[0].boxes  # Get detections
        return detections

    def detect(self, frame, points):
        detections = self.detect_objects(frame)
        status = False

        for det in detections:
            box = det.xyxy[0].tolist()  # Bounding box (x1, y1, x2, y2)
            conf = det.conf.item()  # Confidence score
            class_id = int(det.cls)  # Class ID
            label = self.classes[class_id]

            if label == self.detect_class:
                status = self.draw_prediction(frame, label, box, points)
        if not status:
            cv2.putText(frame, "NOT DETECT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame, status

    @staticmethod
    def is_inside(points, box_points):
        polygon = Polygon(points)
        for point in box_points:
            p = Point(point)
            if polygon.contains(p):
                return True
        return False
