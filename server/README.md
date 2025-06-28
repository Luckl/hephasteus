# ESP32 Multi-Stream Camera Server

A Python Flask server for managing and streaming video from ESP32-CAM devices.

## Module Structure

The server has been split into logical modules for better maintainability:

### `app.py` - Main Application
- Flask application setup and configuration
- REST API endpoints for stream management
- WebSocket event handlers
- Application startup and shutdown logic

### `camera_stream.py` - Camera Streaming
- `CameraStream` class for managing individual camera connections
- `stream_camera()` function for background MJPEG streaming
- Camera URL generation and stream processing logic

### `network_discovery.py` - Network Discovery
- Network interface detection and parsing (Windows/Linux/Mac)
- Broadcast address calculation
- Interface selection logic for ESP32 discovery
- `run_discovery()` function for broadcasting discovery requests

### `udp_listener.py` - UDP Communication
- UDP listener thread for receiving ESP32 responses
- Camera discovery event handling
- Socket management and cleanup

## Key Features

- **Automatic Discovery**: Broadcasts UDP discovery messages to find ESP32 cameras
- **Multi-Stream Support**: Manages multiple camera streams simultaneously
- **MJPEG Streaming**: High-performance streaming using MJPEG protocol
- **WebSocket Real-time Updates**: Live video streaming to web clients
- **Cross-platform Network Detection**: Works on Windows, Linux, and Mac

## API Endpoints

- `GET /` - Main dashboard
- `GET /api/streams` - List all camera streams
- `POST /api/streams` - Add new camera stream
- `DELETE /api/streams/<name>` - Remove camera stream
- `POST /api/streams/<name>/start` - Start streaming from camera
- `POST /api/streams/<name>/stop` - Stop streaming from camera
- `GET /api/streams/<name>/snapshot` - Get single snapshot
- `GET /api/streams/<name>/status` - Get camera status

## WebSocket Events

- `connect` - Client connection
- `disconnect` - Client disconnection
- `join_stream` - Join a specific camera stream
- `leave_stream` - Leave a camera stream
- `frame` - Video frame data (emitted to clients)
- `camera_discovered` - New camera discovered (emitted to clients)

## Dependencies

- Flask
- Flask-SocketIO
- Flask-APScheduler
- requests
- threading
- socket
- logging 