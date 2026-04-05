import numpy as np
import time
import random
import string

class ViolationDetector:
    def __init__(self):
        # Store tracking history to detect violations over time
        # {track_id: {'positions': [], 'timestamps': [], 'violation': None}}
        self.vehicle_history = {}
        self.accidents = [] # List of {track_id, position, time}
        self.emergency_vehicles = set() # Tracked IDs of emergency vehicles

    def check_emergency(self, track_id, class_name):
        """
        Check if the vehicle is an emergency vehicle.
        In this demo, 'person' is treated as a placeholder for ambulance.
        """
        if class_name in ['ambulance', 'person', 'fire_truck']:
            self.emergency_vehicles.add(track_id)
            return True
        return False

    def check_red_light(self, track_id, position, stop_line_y, signal_state):
        """
        Check if a vehicle crosses the stop line when signal is RED.
        """
        x, y = position
        if signal_state == 'RED' and y > stop_line_y:
            if track_id in self.vehicle_history:
                history = self.vehicle_history[track_id]
                if len(history['positions']) > 0:
                    prev_y = history['positions'][-1][1]
                    if prev_y <= stop_line_y:
                        return True
        return False

    def check_overspeeding(self, track_id, position, line1_y, line2_y, distance_meters, fps):
        """
        Calculate speed between two lines using time-over-distance.
        """
        x, y = position
        if track_id not in self.vehicle_history:
            return None

        history = self.vehicle_history[track_id]
        
        # Cross line 1
        if history.get('line1_time') is None and y >= line1_y:
            history['line1_time'] = time.time()
        
        # Cross line 2
        if history.get('line1_time') is not None and history.get('line2_time') is None and y >= line2_y:
            history['line2_time'] = time.time()
            time_taken = history['line2_time'] - history['line1_time']
            if time_taken > 0:
                speed_mps = distance_meters / time_taken
                speed_kph = speed_mps * 3.6
                return speed_kph
        return None

    def check_wrong_lane(self, track_id, position, lane_roi):
        """
        Check if vehicle is in a restricted lane (e.g., driving in opposite lane).
        lane_roi: (x1, y1, x2, y2)
        """
        lx1, ly1, lx2, ly2 = lane_roi
        vx, vy = position
        if lx1 <= vx <= lx2 and ly1 <= vy <= ly2:
            return True
        return False

    def detect_accident(self, track_id, position, current_detections):
        """
        Simple accident detection based on sudden stop or overlap.
        """
        x, y = position
        if track_id in self.vehicle_history:
            history = self.vehicle_history[track_id]
            if len(history['positions']) > 10:
                # Check for sudden stop: velocity near zero while in traffic area
                recent_pos = history['positions'][-10:]
                dist = np.sqrt((recent_pos[-1][0] - recent_pos[0][0])**2 + (recent_pos[-1][1] - recent_pos[0][1])**2)
                if dist < 5: # Moved less than 5 pixels in 10 frames
                    # Simple check: is it overlapping with another vehicle?
                    for other_id, other_pos, other_cls in current_detections:
                        if other_id != track_id:
                            ox, oy = other_pos
                            overlap_dist = np.sqrt((x-ox)**2 + (y-oy)**2)
                            if overlap_dist < 40: # Threshold for collision
                                return True
        return False

    def extract_number_plate(self, track_id, frame, bbox):
        """
        AI-based Number Plate Detection (ANPR) placeholder.
        In a production system, you'd crop the bbox and use a model like PaddleOCR or Tesseract.
        """
        # Placeholder: Generate a fake number plate based on track_id for demo
        if not hasattr(self, '_plate_cache'): self._plate_cache = {}
        
        if track_id not in self._plate_cache:
            prefix = ''.join(random.choices(string.ascii_uppercase, k=2))
            suffix = ''.join(random.choices(string.digits, k=4))
            self._plate_cache[track_id] = f"{prefix}-{suffix}"
            
        return self._plate_cache[track_id]

    def update_history(self, track_id, position):
        if track_id not in self.vehicle_history:
            self.vehicle_history[track_id] = {'positions': [], 'timestamps': []}
        self.vehicle_history[track_id]['positions'].append(position)
        self.vehicle_history[track_id]['timestamps'].append(time.time())
        if len(self.vehicle_history[track_id]['positions']) > 30:
            self.vehicle_history[track_id]['positions'].pop(0)
            self.vehicle_history[track_id]['timestamps'].pop(0)
