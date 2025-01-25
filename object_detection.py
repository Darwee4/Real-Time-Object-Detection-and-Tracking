from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize YOLOv8 object detector with pre-trained weights"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        
    def detect(self, frame):
        """Perform object detection on a single frame"""
        results = self.model(frame)
        detections = []
        
        # Convert results to [x1, y1, x2, y2, conf, class_id] format
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confs, class_ids):
                detections.append([*box, conf, class_id])
                
        return detections
