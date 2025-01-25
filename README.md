# Real-Time Object Detection and Tracking System

This project implements a real-time object detection and tracking system using YOLOv8 for detection and DeepSORT for tracking. The system can process video streams from files or webcams, detect objects, and track them across frames.

## Features
- Real-time object detection using YOLOv8
- Object tracking using DeepSORT
- Visualization of bounding boxes and tracking IDs
- FPS display for performance monitoring
- Support for both video files and webcam input

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Darwee4/real-time-object-tracking.git
cd real-time-object-tracking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLOv8 weights (automatically handled by the script)

## Usage

Run the system with default webcam:
```bash
python inference.py
```

Run with a video file:
```bash
python inference.py --source video.mp4
```

## Project Structure
```
.
├── object_detection.py    # YOLOv8 detection implementation
├── object_tracking.py     # DeepSORT tracking implementation
├── inference.py           # Main inference script
├── utils.py               # Visualization utilities
├── requirements.txt       # Dependency list
└── README.md              # This file
```

## Key Bindings
- Press 'q' to quit the application
- Press 'p' to pause/resume video

## Requirements
- Python 3.8+
- CUDA-enabled GPU (recommended for best performance)

## Acknowledgments
- YOLOv8 by Ultralytics
- DeepSORT implementation by nwojke
- OpenCV for video processing
