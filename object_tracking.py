import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, max_age=30, n_init=3, nms_max_overlap=1.0):
        """Initialize DeepSORT tracker"""
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap
        )
        
    def update(self, detections, frame):
        """Update tracker with new detections"""
        # Convert detections to [x1, y1, x2, y2, conf, class_id] format
        bboxes = np.array([d[:4] for d in detections])
        confidences = np.array([d[4] for d in detections])
        class_ids = np.array([d[5] for d in detections])
        
        # Update tracker and return tracked objects
        tracks = self.tracker.update_tracks(
            bboxes,
            confidences=confidences,
            class_ids=class_ids,
            frame=frame
        )
        
        # Format tracks as [x1, y1, x2, y2, track_id, class_id]
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            class_id = track.get_class()
            ltrb = track.to_ltrb()
            results.append([*ltrb, track_id, class_id])
            
        return results
