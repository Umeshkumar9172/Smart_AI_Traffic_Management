import numpy as np
import time

class TrafficAnalytics:
    def __init__(self, line_y=500):
        self.vehicle_counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.vehicle_counts_incoming = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.vehicle_counts_outgoing = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
        self.line_y = line_y
        self.tracked_vehicles = {} # {track_id: last_y}
        self.time_history = [] # List of (timestamp, count)
        self.heatmap_data = [] # List of (x, y) for detected objects

    def update_analytics(self, track_id, position, class_name):
        """
        Update vehicle counts, directional logic, and history.
        """
        x, y = position
        self.heatmap_data.append((int(x), int(y)))
        if len(self.heatmap_data) > 1000: self.heatmap_data.pop(0)

        if track_id not in self.tracked_vehicles:
            self.tracked_vehicles[track_id] = {'last_y': y, 'counted': False}
            if class_name in self.vehicle_counts:
                self.vehicle_counts[class_name] += 1
            else:
                self.vehicle_counts['car'] += 1
            self.time_history.append((time.time(), sum(self.vehicle_counts.values())))
            return None

        last_y = self.tracked_vehicles[track_id]['last_y']
        counted = self.tracked_vehicles[track_id]['counted']
        
        direction = None
        if not counted:
            # Crossing from top to bottom (Incoming)
            if last_y < self.line_y and y >= self.line_y:
                if class_name in self.vehicle_counts_incoming:
                    self.vehicle_counts_incoming[class_name] += 1
                self.tracked_vehicles[track_id]['counted'] = True
                direction = "Incoming"
            
            # Crossing from bottom to top (Outgoing)
            elif last_y > self.line_y and y <= self.line_y:
                if class_name in self.vehicle_counts_outgoing:
                    self.vehicle_counts_outgoing[class_name] += 1
                self.tracked_vehicles[track_id]['counted'] = True
                direction = "Outgoing"

        self.tracked_vehicles[track_id]['last_y'] = y
        return direction

    def predict_future_density(self, window_seconds=60):
        """
        AI-based density prediction using recent trends (simple moving average/extrapolation).
        In a production system, this would use a LSTM/Transformer model.
        """
        if len(self.time_history) < 5: return "Insufficient Data"
        
        # Get counts from the last N seconds
        current_time = time.time()
        recent_counts = [count for t, count in self.time_history if current_time - t < window_seconds]
        
        if len(recent_counts) < 2: return "Stable"
        
        # Simple trend calculation
        trend = recent_counts[-1] - recent_counts[0]
        if trend > 5: return "📈 Rising (High Congestion Expected)"
        elif trend < -5: return "📉 Falling (Clearing Soon)"
        else: return "➡️ Stable Traffic"

    def get_summary(self):
        total_in = sum(self.vehicle_counts_incoming.values())
        total_out = sum(self.vehicle_counts_outgoing.values())
        total_all = sum(self.vehicle_counts.values())
        return {
            'total_incoming': total_in,
            'total_outgoing': total_out,
            'total_vehicles': total_all,
            'breakdown': self.vehicle_counts,
            'time_history': self.time_history
        }
