import cv2
import numpy as np

class Visualizer:
    @staticmethod
    def draw_tracking(frame, results, show_boxes=True, show_labels=True, class_names=None, violation_detector=None):
        if results is not None and results.boxes is not None:
            for box in results.boxes:
                if box.id is not None:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    track_id = int(box.id[0])
                    class_id = int(box.cls[0])
                    label = class_names[class_id] if class_names else str(class_id)
                    
                    # Special color for emergency vehicles
                    color = (0, 255, 0) if label in ['ambulance', 'person', 'fire_truck'] else (255, 120, 0)
                    
                    if show_boxes:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    if show_labels:
                        display_label = f'🚑 EMERGENCY #{track_id}' if label in ['ambulance', 'person', 'fire_truck'] else f'{label} #{track_id}'
                        cv2.putText(frame, display_label, (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # 📸 Draw Number Plate if available
                        if violation_detector:
                            plate = violation_detector.extract_number_plate(track_id, frame, (x1, y1, x2, y2))
                            cv2.putText(frame, f'Plate: {plate}', (int(x1), int(y2) + 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return frame

    @staticmethod
    def draw_heatmap(frame, heatmap_data):
        overlay = frame.copy()
        for x, y in heatmap_data:
            cv2.circle(overlay, (x, y), 15, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        return frame

    @staticmethod
    def draw_line(frame, line_coords, label='', color=(0, 255, 255), thickness=2):
        cv2.line(frame, line_coords[0], line_coords[1], color, thickness)
        if label:
            cv2.putText(frame, label, (line_coords[0][0], line_coords[0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    @staticmethod
    def draw_violation(frame, track_id, violation_type, position, color=(0, 0, 255)):
        x, y = position
        cv2.putText(frame, f'ALARM: {violation_type} (ID: {track_id})', (int(x), int(y) - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame
