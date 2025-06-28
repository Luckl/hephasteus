import os
import json
import time
import threading
import requests
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_apscheduler import APScheduler

# Import our modules
from camera_stream import CameraStream, stream_camera
from network_discovery import run_discovery
from udp_listener import start_udp_listener, stop_udp_listener

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Flask-APScheduler configuration
app.config['SCHEDULER_API_ENABLED'] = True
app.config['SCHEDULER_TIMEZONE'] = 'UTC'

socketio = SocketIO(app, cors_allowed_origins="*")
scheduler = APScheduler()
scheduler.init_app(app)

# Global storage for camera streams
camera_streams = {}
stream_threads = {}
discovery_running = False  # Add this global variable
scheduler_initialized = False  # Add this flag to prevent multiple initializations

def start_scheduler():
    """Start the Flask-APScheduler with discovery job"""
    global scheduler
    
    # Add the discovery job to run every 15 seconds
    scheduler.add_job(
        func=run_discovery,
        trigger='interval',
        seconds=15,
        id='discovery_job',
        name='ESP32 Camera Discovery',
        replace_existing=True,
        max_instances=1,
        coalesce=True
    )

    run_discovery()
    
    # Start the scheduler
    scheduler.start()
    logger.info("Flask-APScheduler started with discovery job (every 15 seconds)")
    next_run = scheduler.get_job('discovery_job').next_run_time
    logger.debug(f"Next discovery run scheduled for: {next_run}")
    
    return scheduler

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', streams=camera_streams)

@app.route('/api/streams', methods=['GET'])
def get_streams():
    """Get all camera streams"""
    streams_data = []
    for name, stream in camera_streams.items():
        streams_data.append({
            'name': stream.name,
            'ip_address': stream.ip_address,
            'port': stream.port,
            'is_active': stream.is_active,
            'fps': stream.fps,
            'frame_count': stream.frame_count,
            'connected_clients': len(stream.connected_clients),
            'last_frame_time': stream.last_frame_time.isoformat() if stream.last_frame_time else None
        })
    return jsonify(streams_data)

@app.route('/api/streams', methods=['POST'])
def add_stream():
    """Add a new camera stream"""
    data = request.get_json()
    name = data.get('name')
    ip_address = data.get('ip_address')
    port = data.get('port', 80)
    
    if not name or not ip_address:
        return jsonify({'error': 'Name and IP address are required'}), 400
    
    if name in camera_streams:
        return jsonify({'error': 'Stream with this name already exists'}), 400
    
    camera_streams[name] = CameraStream(name, ip_address, port)
    return jsonify({'message': 'Stream added successfully', 'name': name}), 201

@app.route('/api/streams/<name>', methods=['DELETE'])
def remove_stream(name):
    """Remove a camera stream"""
    if name not in camera_streams:
        return jsonify({'error': 'Stream not found'}), 404
    
    # Stop the stream if it's running
    if name in stream_threads:
        camera_streams[name].is_active = False
        stream_threads[name].join()
        del stream_threads[name]
    
    del camera_streams[name]
    return jsonify({'message': 'Stream removed successfully'})

@app.route('/api/streams/<name>/start', methods=['POST'])
def start_stream(name):
    """Start streaming from a camera"""
    if name not in camera_streams:
        return jsonify({'error': 'Stream not found'}), 404
    
    stream = camera_streams[name]
    if stream.is_active:
        return jsonify({'error': 'Stream is already active'}), 400
    
    stream.is_active = True
    thread = threading.Thread(
        target=stream_camera, 
        args=(name, camera_streams, stream_threads, socketio), 
        daemon=True
    )
    stream_threads[name] = thread
    thread.start()
    
    return jsonify({'message': 'Stream started successfully'})

@app.route('/api/streams/<name>/stop', methods=['POST'])
def stop_stream(name):
    """Stop streaming from a camera"""
    if name not in camera_streams:
        return jsonify({'error': 'Stream not found'}), 404
    
    stream = camera_streams[name]
    stream.is_active = False
    
    if name in stream_threads:
        stream_threads[name].join()
        del stream_threads[name]
    
    return jsonify({'message': 'Stream stopped successfully'})

@app.route('/api/streams/<name>/snapshot', methods=['GET'])
def get_snapshot(name):
    """Get a single snapshot from a camera"""
    if name not in camera_streams:
        return jsonify({'error': 'Stream not found'}), 404
    
    stream = camera_streams[name]
    try:
        logger.info(f"Attempting to get snapshot from {stream.get_snapshot_url()}")
        response = requests.get(stream.get_snapshot_url(), timeout=10)
        logger.debug(f"Snapshot response status: {response.status_code}")
        
        if response.status_code == 200:
            return Response(response.content, mimetype='image/jpeg')
        else:
            logger.error(f"Failed to capture snapshot: HTTP {response.status_code}")
            return jsonify({'error': f'Failed to capture snapshot: HTTP {response.status_code}'}), 500
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to {stream.ip_address}: {str(e)}")
        return jsonify({'error': f'Cannot connect to camera at {stream.ip_address}. Make sure the ESP32 is powered on and connected to the network.'}), 500
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout error to {stream.ip_address}: {str(e)}")
        return jsonify({'error': f'Timeout connecting to camera at {stream.ip_address}'}), 500
    except Exception as e:
        logger.error(f"Error capturing snapshot from {stream.ip_address}: {str(e)}")
        return jsonify({'error': f'Error capturing snapshot: {str(e)}'}), 500

@app.route('/api/streams/<name>/status', methods=['GET'])
def get_stream_status(name):
    """Get status of a specific camera stream"""
    if name not in camera_streams:
        return jsonify({'error': 'Stream not found'}), 404
    
    stream = camera_streams[name]
    try:
        response = requests.get(stream.get_status_url(), timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Failed to get camera status'}), 500
    except Exception as e:
        return jsonify({'error': f'Error getting status: {str(e)}'}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info(f"Client disconnected: {request.sid}")
    # Remove client from all streams
    for stream in camera_streams.values():
        stream.connected_clients.discard(request.sid)

@socketio.on('join_stream')
def handle_join_stream(data):
    """Handle client joining a specific stream"""
    stream_name = data.get('stream_name')
    if stream_name in camera_streams:
        join_room(stream_name)
        camera_streams[stream_name].connected_clients.add(request.sid)
        logger.info(f"Client {request.sid} joined stream {stream_name}")
        
        # Send current frame if available
        stream = camera_streams[stream_name]
        if stream.last_frame:
            emit('frame', {
                'stream_name': stream_name,
                'frame_data': stream.last_frame,
                'timestamp': stream.last_frame_time.isoformat() if stream.last_frame_time else None,
                'frame_count': stream.frame_count
            })

@socketio.on('leave_stream')
def handle_leave_stream(data):
    """Handle client leaving a specific stream"""
    stream_name = data.get('stream_name')
    if stream_name in camera_streams:
        leave_room(stream_name)
        camera_streams[stream_name].connected_clients.discard(request.sid)
        logger.info(f"Client {request.sid} left stream {stream_name}")

if __name__ == '__main__':
    # Start the continuous UDP listener first
    start_udp_listener(camera_streams, socketio)
    
    # Start the Flask-APScheduler with discovery job
    scheduler_instance = start_scheduler()
    
    logger.info("ESP32 Multi-Stream Camera Server")
    logger.info("Available endpoints:")
    logger.info("  GET  /                    - Main dashboard")
    logger.info("  GET  /api/streams         - List all streams")
    logger.info("  POST /api/streams         - Add new stream")
    logger.info("  DELETE /api/streams/<name> - Remove stream")
    logger.info("  POST /api/streams/<name>/start - Start streaming")
    logger.info("  POST /api/streams/<name>/stop  - Stop streaming")
    logger.info("  GET  /api/streams/<name>/snapshot - Get snapshot")
    logger.info("  GET  /api/streams/<name>/status  - Get camera status")
    logger.info("Example ESP32 endpoints:")
    logger.info("  http://<esp32-ip>/capture  - Get JPEG snapshot")
    logger.info("  http://<esp32-ip>/stream   - Get MJPEG stream")
    logger.info("  http://<esp32-ip>/status   - Get camera status")
    
    try:
        # Disable debug mode to prevent application restarts that cause duplicate jobs
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    finally:
        # Shutdown the scheduler and UDP listener when the app stops
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()
        logger.info("Shutting down UDP listener...")
        stop_udp_listener() 