import cv2
import time
from src.detection import VehicleDetector
from src.violations import ViolationDetector
from src.analytics import TrafficAnalytics
from src.utils import Visualizer

class TrafficSystem:
    def __init__(self, video_source=0):
        self.detector = VehicleDetector()
        self.violation_detector = ViolationDetector()
        self.analytics = TrafficAnalytics()
        self.visualizer = Visualizer()
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        
        # Default configuration for ROI and stop line
        self.stop_line_y = 500
        self.signal_state = 'RED' # Can be toggled manually or via a timer
        self.parking_roi = [100, 100, 300, 300] # [x1, y1, x2, y2]
        self.line1_y = 400
        self.line2_y = 600
        self.distance_meters = 10.0 # Distance between line 1 and 2

    def process_video(self, output_path=None):
        if not self.cap.isOpened():
            print("Error: Could not open video source.")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        else:
            out = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 1. Detect and track vehicles
            results = self.detector.track(frame)
            
            # 2. Process tracking results
            if results.boxes is not None and results.boxes.id is not None:
                for box in results.boxes:
                    track_id = int(box.id[0])
                    class_id = int(box.cls[0])
                    class_name = self.detector.model.names[class_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # 3. Violation checks
                    # Red light jumping
                    if self.violation_detector.check_red_light(track_id, (center_x, center_y), self.stop_line_y, self.signal_state):
                        frame = self.visualizer.draw_violation(frame, track_id, "RED LIGHT JUMPING", (x1, y1))
                    
                    # Overspeeding
                    speed = self.violation_detector.check_overspeeding(track_id, (center_x, center_y), self.line1_y, self.line2_y, self.distance_meters, fps)
                    if speed is not None:
                        if speed > 60: # Threshold 60 kph
                            frame = self.visualizer.draw_violation(frame, track_id, f"OVERSPEEDING: {speed:.1f} kph", (x1, y1))
                        else:
                            cv2.putText(frame, f'Speed: {speed:.1f} kph', (int(x1), int(y1) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Illegal parking
                    if self.violation_detector.check_illegal_parking(track_id, (center_x, center_y), self.parking_roi):
                        frame = self.visualizer.draw_violation(frame, track_id, "ILLEGAL PARKING", (x1, y1))
                    
                    # 4. Traffic Analytics
                    # Counting vehicles
                    self.analytics.count_vehicle(track_id, (center_x, center_y), class_name)
                    
                    # Update history
                    self.violation_detector.update_history(track_id, (center_x, center_y))

            # 5. Draw static elements
            frame = self.visualizer.draw_line(frame, ((0, self.stop_line_y), (frame_width, self.stop_line_y)), label="Stop Line", color=(0, 0, 255))
            frame = self.visualizer.draw_line(frame, ((0, self.line1_y), (frame_width, self.line1_y)), label="Speed Line 1", color=(0, 255, 255))
            frame = self.visualizer.draw_line(frame, ((0, self.line2_y), (frame_width, self.line2_y)), label="Speed Line 2", color=(0, 255, 255))
            # Draw parking ROI
            px1, py1, px2, py2 = self.parking_roi
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 255), 2)
            cv2.putText(frame, "Illegal Parking Zone", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Draw tracking
            frame = self.visualizer.draw_tracking(frame, results)
            
            # Display summary on frame
            summary = self.analytics.get_summary()
            cv2.putText(frame, f"Total Vehicles: {summary['total_vehicles']}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Signal State: {self.signal_state}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if self.signal_state == 'RED' else (0, 255, 0), 2)

            if out:
                out.write(frame)
            
            cv2.imshow("Traffic Monitoring System", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'): # Toggle signal state
                self.signal_state = 'GREEN' if self.signal_state == 'RED' else 'RED'

        self.cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # In a real case, you would pass a video file path here
    # Example: system = TrafficSystem("data/sample_traffic.mp4")
    system = TrafficSystem(0) # Use webcam for now
    system.process_video()
