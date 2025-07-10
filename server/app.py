import os
import json
import time
import threading
import requests
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_apscheduler import APScheduler

# Import our modules
from camera_stream import CameraStream, stream_camera
from network_discovery import run_discovery
from udp_listener import start_udp_listener, stop_udp_listener
from video_recorder import get_video_recorder
from video_analyzer import get_video_analyzer
from florence2_detector import process_video_with_florence2, get_florence2_model

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
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

# Pre-load Florence 2 model on startup
logger.info("Pre-loading Florence 2 model on startup...")
try:
    florence2_model, florence2_processor = get_florence2_model()
    if florence2_model is not None and florence2_processor is not None:
        logger.info("Florence 2 model loaded successfully on startup")
    else:
        logger.warning("Failed to load Florence 2 model on startup - will load on first use")
except Exception as e:
    logger.error(f"Error pre-loading Florence 2 model: {e}")
    logger.info("Florence 2 model will be loaded on first use")



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
        # Get recording status
        recorder = get_video_recorder()
        recording_status = recorder.get_recording_status(name)
        
        streams_data.append({
            'name': stream.name,
            'ip_address': stream.ip_address,
            'port': stream.port,
            'is_active': stream.is_active,
            'fps': stream.fps,
            'frame_count': stream.frame_count,
            'connected_clients': len(stream.connected_clients),
            'last_frame_time': stream.last_frame_time.isoformat() if stream.last_frame_time else None,
            'cv_enabled': stream.cv_enabled,
            'cv_detections': stream.cv_detections,
            'last_detection_time': stream.last_detection_time.isoformat() if stream.last_detection_time else None,
            'recording_enabled': stream.recording_enabled,
            'recording_status': recording_status
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

# Computer Vision API endpoints
@app.route('/api/streams/<name>/cv/toggle', methods=['POST'])
def toggle_computer_vision(name):
    """Toggle computer vision processing for a stream"""
    if name not in camera_streams:
        return jsonify({'error': 'Stream not found'}), 404
    
    stream = camera_streams[name]
    data = request.get_json() or {}
    enabled = data.get('enabled')  # None means toggle
    
    cv_enabled = stream.toggle_cv(enabled)
    
    return jsonify({
        'message': f'Computer vision {"enabled" if cv_enabled else "disabled"}',
        'cv_enabled': cv_enabled
    })

@app.route('/api/streams/<name>/cv/detections', methods=['GET'])
def get_detections(name):
    """Get current detections for a stream"""
    if name not in camera_streams:
        return jsonify({'error': 'Stream not found'}), 404
    
    stream = camera_streams[name]
    
    return jsonify({
        'cv_enabled': stream.cv_enabled,
        'detections': stream.cv_detections,
        'last_detection_time': stream.last_detection_time.isoformat() if stream.last_detection_time else None,
        'detection_count': len(stream.cv_detections)
    })

@app.route('/api/streams/<name>/cv/settings', methods=['GET'])
def get_cv_settings(name):
    """Get computer vision settings for a stream"""
    if name not in camera_streams:
        return jsonify({'error': 'Stream not found'}), 404
    
    stream = camera_streams[name]
    
    return jsonify({
        'cv_enabled': stream.cv_enabled,
        'available_detections': ['face', 'person', 'motion'],
        'detection_types': {
            'face': 'Face detection using Haar cascade',
            'person': 'Person detection using HOG',
            'motion': 'Motion detection using background subtraction'
        }
    })

# Video Recording API endpoints
@app.route('/api/streams/<name>/recording/toggle', methods=['POST'])
def toggle_recording(name):
    """Toggle video recording for a stream"""
    if name not in camera_streams:
        return jsonify({'error': 'Stream not found'}), 404
    
    stream = camera_streams[name]
    data = request.get_json() or {}
    enabled = data.get('enabled')  # None means toggle
    
    recording_enabled = stream.toggle_recording(enabled)
    
    return jsonify({
        'message': f'Video recording {"enabled" if recording_enabled else "disabled"}',
        'recording_enabled': recording_enabled
    })

@app.route('/api/streams/<name>/recording/status', methods=['GET'])
def get_recording_status(name):
    """Get recording status for a stream"""
    if name not in camera_streams:
        return jsonify({'error': 'Stream not found'}), 404
    
    recorder = get_video_recorder()
    status = recorder.get_recording_status(name)
    
    return jsonify(status)

@app.route('/api/recordings', methods=['GET'])
def get_all_recordings():
    """Get all recorded videos"""
    recorder = get_video_recorder()
    recordings = recorder.get_all_recordings()
    
    # Add analysis status to each recording
    analyzer = get_video_analyzer()
    for recording in recordings:
        filepath = recording['filepath']
        analysis_path = filepath.replace('.mp4', '_analysis.json')
        recording['has_analysis'] = os.path.exists(analysis_path)
        if recording['has_analysis']:
            recording['analysis_path'] = analysis_path
    
    return jsonify({
        'recordings': recordings,
        'total_count': len(recordings)
    })

@app.route('/api/recordings/<filename>', methods=['GET'])
def download_recording(filename):
    """Download a recorded video file"""
    recorder = get_video_recorder()
    filepath = os.path.join(recorder.output_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Recording not found'}), 404
    
    return send_file(filepath, as_attachment=True)

# Video Analysis API endpoints
@app.route('/api/recordings/<filename>/analyze', methods=['POST'])
def analyze_recording(filename):
    """Analyze a recorded video using Gemini 2.5 Flash"""
    recorder = get_video_recorder()
    filepath = os.path.join(recorder.output_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Recording not found'}), 404
    
    analyzer = get_video_analyzer()
    result = analyzer.analyze_video(filepath)
    
    if 'error' in result:
        return jsonify(result), 500
    
    return jsonify({
        'message': 'Video analysis completed successfully',
        'analysis': result,
        'analysis_file': filepath.replace('.mp4', '_analysis.json')
    })

@app.route('/api/recordings/<filename>/analysis', methods=['GET'])
def get_recording_analysis(filename):
    """Get analysis for a recorded video"""
    recorder = get_video_recorder()
    filepath = os.path.join(recorder.output_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Recording not found'}), 404
    
    analyzer = get_video_analyzer()
    result = analyzer.get_analysis(filepath)
    
    if 'error' in result:
        return jsonify(result), 404
    
    return jsonify(result)

@app.route('/api/recordings/<filename>/analysis.json', methods=['GET'])
def download_analysis(filename):
    """Download the analysis JSON file for a recording"""
    recorder = get_video_recorder()
    filepath = os.path.join(recorder.output_dir, filename)
    analysis_path = filepath.replace('.mp4', '_analysis.json')
    
    if not os.path.exists(analysis_path):
        return jsonify({'error': 'Analysis not found'}), 404
    
    return send_file(analysis_path, as_attachment=True, mimetype='application/json')

@app.route('/api/recordings/<filename>/ask', methods=['POST'])
def ask_question_about_recording(filename):
    """Ask a question about a recorded video using Gemini"""
    recorder = get_video_recorder()
    filepath = os.path.join(recorder.output_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Recording not found'}), 404
    
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400
    
    question = data['question']
    analyzer = get_video_analyzer()
    result = analyzer.ask_question_about_video(filepath, question)
    
    if 'error' in result:
        return jsonify(result), 500
    
    return jsonify({
        'message': 'Question answered successfully',
        'result': result
    })

@app.route('/api/recordings/<filename>/analyze-florence2', methods=['POST'])
def analyze_recording_with_florence2(filename):
    """Analyze a recorded video with Florence 2 object detection"""
    recorder = get_video_recorder()
    filepath = os.path.join(recorder.output_dir, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Recording not found'}), 404
    
    try:
        # Create output path for Florence 2 processed video
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_florence2.mp4"
        output_path = os.path.join(recorder.output_dir, output_filename)
        
        # Process video with Florence 2
        result = process_video_with_florence2(filepath, output_path)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify({
            'message': 'Florence 2 analysis completed successfully',
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Florence 2 analysis error for {filename}: {e}")
        return jsonify({'error': str(e)}), 500

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
                'frame_count': stream.frame_count,
                'detections': stream.cv_detections if stream.cv_enabled else []
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
    
    logger.info("ESP32 Multi-Stream Camera Server with Computer Vision & Recording")
    logger.info("Available endpoints:")
    logger.info("  GET  /                    - Main dashboard")
    logger.info("  GET  /api/streams         - List all streams")
    logger.info("  POST /api/streams         - Add new stream")
    logger.info("  DELETE /api/streams/<name> - Remove stream")
    logger.info("  POST /api/streams/<name>/start - Start streaming")
    logger.info("  POST /api/streams/<name>/stop  - Stop streaming")
    logger.info("  GET  /api/streams/<name>/snapshot - Get snapshot")
    logger.info("  GET  /api/streams/<name>/status  - Get camera status")
    logger.info("Computer Vision endpoints:")
    logger.info("  POST /api/streams/<name>/cv/toggle - Toggle CV processing")
    logger.info("  GET  /api/streams/<name>/cv/detections - Get current detections")
    logger.info("  GET  /api/streams/<name>/cv/settings - Get CV settings")
    logger.info("Video Recording endpoints:")
    logger.info("  POST /api/streams/<name>/recording/toggle - Toggle recording")
    logger.info("  GET  /api/streams/<name>/recording/status - Get recording status")
    logger.info("  GET  /api/recordings - List all recordings")
    logger.info("  GET  /api/recordings/<filename> - Download recording")
    logger.info("Video Analysis endpoints:")
    logger.info("  POST /api/recordings/<filename>/analyze - Analyze recording")
    logger.info("  GET  /api/recordings/<filename>/analysis - Get recording analysis")
    logger.info("  GET  /api/recordings/<filename>/analysis.json - Download analysis JSON")
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