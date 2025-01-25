import cv2
import random

# Define color palette for visualization
COLORS = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
          for _ in range(1000)]

def draw_boxes(frame, tracks):
    """Draw bounding boxes and tracking IDs on frame"""
    for track in tracks:
        x1, y1, x2, y2, track_id, class_id = map(int, track)
        
        # Get color for this track
        color = COLORS[track_id % len(COLORS)]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with track ID and class ID
        label = f"ID: {track_id} | Class: {class_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def display_fps(frame, fps):
    """Display FPS on frame"""
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame
