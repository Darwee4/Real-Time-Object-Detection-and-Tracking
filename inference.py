import cv2
import time
from object_detection import ObjectDetector
from object_tracking import ObjectTracker
from utils import draw_boxes, display_fps

def main(video_source=0):
    # Initialize components
    detector = ObjectDetector()
    tracker = ObjectTracker()
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
        
    # Initialize FPS calculation
    prev_time = 0
    curr_time = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform object detection
        detections = detector.detect(frame)
        
        # Update tracker with detections
        tracks = tracker.update(detections, frame)
        
        # Draw visualization
        frame = draw_boxes(frame, tracks)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        frame = display_fps(frame, fps)
        
        # Display frame
        cv2.imshow("Real-Time Object Tracking", frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
