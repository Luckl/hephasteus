# ESP32 Camera Server

A Python Flask server for managing and streaming ESP32-CAM devices with computer vision capabilities.

## Features

- Real-time video streaming from ESP32-CAM devices
- Automatic device discovery via UDP broadcast
- Computer vision processing with YOLO object detection
- Video recording and playback
- Web-based dashboard for device management
- RESTful API for integration
- WebSocket support for real-time updates

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

- `GET /api/streams` - List all camera streams
- `POST /api/streams` - Add a new camera stream
- `DELETE /api/streams/<name>` - Remove a camera stream
- `POST /api/streams/<name>/start` - Start streaming from a camera
- `POST /api/streams/<name>/stop` - Stop streaming from a camera
- `GET /api/streams/<name>/snapshot` - Get a snapshot from a camera
- `POST /api/streams/<name>/cv/toggle` - Toggle computer vision processing
- `GET /api/streams/<name>/cv/detections` - Get current detections

## Computer Vision

The server supports real-time object detection using YOLO models:

- YOLOv8n (default) - Fast and lightweight
- YOLOv11n - Latest model with improved accuracy
- Florence2 - Advanced vision model for detailed analysis

## Video Recording

- Automatic recording based on motion detection
- Manual recording controls
- Video playback and analysis
- Export capabilities

## Docker Support

Run the server in a Docker container:

```bash
docker-compose up -d
```

## License

This project is licensed under the MIT License. 