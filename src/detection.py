from ultralytics import YOLO
import cv2

class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the YOLOv8 model for vehicle detection.
        Vehicle classes in COCO dataset: car(2), motorcycle(3), bus(5), truck(7)
        """
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 3, 5, 7, 0] # car, motorcycle, bus, truck, person (placeholder for ambulance/emergency)
        self.class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 0: 'person'}
        # In a production system, you'd use a model trained specifically for ambulances (class 0 is just for testing)

    def track(self, frame, conf=0.25, iou=0.45):
        """
        Perform object tracking using YOLOv8's built-in tracker.
        Returns the tracking results object.
        """
        results = self.model.track(
            frame, 
            persist=True, 
            verbose=False, 
            classes=self.vehicle_classes,
            conf=conf,
            iou=iou,
            tracker="bytetrack.yaml" # Options: botsort.yaml, bytetrack.yaml
        )
        return results[0] if results else None
