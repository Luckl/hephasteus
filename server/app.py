import os
import json
import time
import threading
import requests
import socket
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit, join_room, leave_room
import base64
from flask_apscheduler import APScheduler
import subprocess
import platform
import re

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

# Global UDP listener for ESP32 responses
udp_listener_socket = None
udp_listener_thread = None
udp_listener_running = False

class CameraStream:
    def __init__(self, name, ip_address, port=80):
        self.name = name
        self.ip_address = ip_address
        self.port = port
        self.base_url = f"http://{ip_address}:{port}"
        self.is_active = False
        self.last_frame = None
        self.last_frame_time = None
        self.frame_count = 0
        self.fps = 0
        self.connected_clients = set()
        
    def get_stream_url(self):
        return f"{self.base_url}/stream"
    
    def get_snapshot_url(self):
        return f"{self.base_url}/capture"
    
    def get_status_url(self):
        return f"{self.base_url}/status"

def get_network_interfaces():
    """Get all available network interfaces with their broadcast addresses"""
    interfaces = []
    
    try:
        if platform.system() == "Windows":
            # Windows: use ipconfig
            result = subprocess.run(['ipconfig'], capture_output=True, text=True)
            if result.returncode == 0:
                interfaces = parse_windows_ipconfig(result.stdout)
        else:
            # Linux/Mac: use ifconfig or ip
            try:
                result = subprocess.run(['ip', 'addr'], capture_output=True, text=True)
                if result.returncode == 0:
                    interfaces = parse_linux_ip_addr(result.stdout)
                else:
                    # Fallback to ifconfig
                    result = subprocess.run(['ifconfig'], capture_output=True, text=True)
                    if result.returncode == 0:
                        interfaces = parse_ifconfig(result.stdout)
            except FileNotFoundError:
                # Try ifconfig as fallback
                result = subprocess.run(['ifconfig'], capture_output=True, text=True)
                if result.returncode == 0:
                    interfaces = parse_ifconfig(result.stdout)
                    
    except Exception as e:
        logger.error(f"Error getting network interfaces: {e}")
    
    return interfaces

def parse_windows_ipconfig(output):
    """Parse Windows ipconfig output"""
    interfaces = []
    current_interface = {}
    
    for line in output.split('\n'):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # New interface section - starts with adapter name and ends with colon
        if line.endswith(':') and not line.startswith(' '):
            # Save previous interface if it has data
            if current_interface and 'name' in current_interface:
                interfaces.append(current_interface)
            
            # Start new interface
            current_interface = {'name': line[:-1].strip()}  # Remove the colon
        
        # IPv4 Address
        elif 'IPv4 Address' in line and ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                ip = parts[1].strip()
                # Remove "(Preferred)" suffix if present
                if '(Preferred)' in ip:
                    ip = ip.replace('(Preferred)', '').strip()
                if ip and ip != '(Preferred)':
                    current_interface['ip'] = ip
        
        # Subnet Mask
        elif 'Subnet Mask' in line and ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                mask = parts[1].strip()
                if mask:
                    current_interface['subnet'] = mask
        
        # Default Gateway
        elif 'Default Gateway' in line and ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                gateway = parts[1].strip()
                if gateway:
                    current_interface['gateway'] = gateway
    
    # Don't forget the last interface
    if current_interface and 'name' in current_interface:
        interfaces.append(current_interface)
    
    return interfaces

def parse_linux_ip_addr(output):
    """Parse Linux 'ip addr' output"""
    interfaces = []
    current_interface = {}
    
    for line in output.split('\n'):
        line = line.strip()
        
        # Interface number and name
        if re.match(r'^\d+:', line):
            if current_interface:
                interfaces.append(current_interface)
            parts = line.split(':')
            current_interface = {'name': parts[1].strip()}
        
        # IP address
        elif 'inet ' in line:
            match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', line)
            if match:
                current_interface['ip'] = match.group(1)
        
        # Broadcast address
        elif 'brd ' in line:
            match = re.search(r'brd (\d+\.\d+\.\d+\.\d+)', line)
            if match:
                current_interface['broadcast'] = match.group(1)
    
    if current_interface:
        interfaces.append(current_interface)
    
    return interfaces

def parse_ifconfig(output):
    """Parse ifconfig output (Mac/Linux)"""
    interfaces = []
    current_interface = {}
    
    for line in output.split('\n'):
        line = line.strip()
        
        # Interface name
        if line and not line.startswith(' ') and ':' in line:
            if current_interface:
                interfaces.append(current_interface)
            current_interface = {'name': line.split(':')[0]}
        
        # IP address
        elif 'inet ' in line:
            match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', line)
            if match:
                current_interface['ip'] = match.group(1)
        
        # Broadcast address
        elif 'broadcast ' in line:
            match = re.search(r'broadcast (\d+\.\d+\.\d+\.\d+)', line)
            if match:
                current_interface['broadcast'] = match.group(1)
    
    if current_interface:
        interfaces.append(current_interface)
    
    return interfaces

def calculate_broadcast_address(ip, subnet_mask):
    """Calculate broadcast address from IP and subnet mask"""
    try:
        ip_parts = [int(x) for x in ip.split('.')]
        mask_parts = [int(x) for x in subnet_mask.split('.')]
        
        # Calculate network address
        network = [ip_parts[i] & mask_parts[i] for i in range(4)]
        
        # Calculate broadcast address
        broadcast = [network[i] | (255 - mask_parts[i]) for i in range(4)]
        
        return '.'.join(map(str, broadcast))
    except Exception as e:
        logger.error(f"Error calculating broadcast address: {e}")
        return None

def select_best_interface(interfaces):
    """Select the best interface for broadcasting to ESP32 devices"""
    logger.info(f"Found {len(interfaces)} network interfaces to evaluate")
    valid_interfaces = []
    
    for iface in interfaces:
        # Skip interfaces without IP
        if 'ip' not in iface:
            logger.debug(f"Skipping interface {iface.get('name', 'Unknown')} - no IP address")
            continue
            
        ip = iface['ip']
        
        # Skip loopback, link-local, and WSL interfaces
        if (ip.startswith('127.') or 
            ip.startswith('169.254.') or 
            ip.startswith('172.29.') or  # WSL2 default
            ip.startswith('172.30.') or  # WSL2 range
            ip.startswith('172.31.') or  # WSL2 range
            'wsl' in iface['name'].lower() or
            'docker' in iface['name'].lower() or
            'veth' in iface['name'].lower()):
            logger.debug(f"Skipping interface {iface['name']} ({ip}) - unsuitable for broadcasting")
            continue
        
        # Calculate broadcast address if not provided
        if 'broadcast' not in iface and 'subnet' in iface:
            broadcast = calculate_broadcast_address(ip, iface['subnet'])
            if broadcast:
                iface['broadcast'] = broadcast
                logger.debug(f"Calculated broadcast {broadcast} for {iface['name']} ({ip})")
        
        # Only include interfaces with broadcast address
        if 'broadcast' in iface:
            logger.info(f"Valid interface: {iface['name']} - IP: {ip}, Broadcast: {iface['broadcast']}")
            valid_interfaces.append(iface)
    
    # Sort by preference: prefer interfaces with gateways (connected to router)
    valid_interfaces.sort(key=lambda x: 'gateway' not in x)
    
    if valid_interfaces:
        selected = valid_interfaces[0]
        logger.info(f"Selected interface: {selected['name']} ({selected['ip']})")
        return selected
    
    logger.warning("No suitable network interfaces found")
    return None

def run_discovery():
    """Broadcast discovery requests to ESP32 cameras"""
    logger.info("Broadcasting ESP32 camera discovery request...")
    
    # Get all network interfaces
    interfaces = get_network_interfaces()
    logger.info(f"Found {len(interfaces)} network interfaces")
    
    # Select the best interface
    selected_interface = select_best_interface(interfaces)
    
    if not selected_interface:
        logger.error("No suitable network interface found for broadcasting")
        return
    
    network_ip = selected_interface['ip']
    broadcast_ip = selected_interface['broadcast']
    
    logger.info(f"Selected interface: {selected_interface['name']}")
    logger.info(f"Local IP: {network_ip}")
    logger.info(f"Broadcast IP: {broadcast_ip}")
    
    try:
        # Create socket for broadcasting
        broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        # Bind to specific interface to ensure broadcasts go to the right network
        broadcast_socket.bind((network_ip, 0))
        logger.info(f"Bound to interface {network_ip}")
        
        try:
            # Send discovery request to broadcast
            discovery_request = "DISCOVER_CAMERAS"
            broadcast_addr = (broadcast_ip, 8888)  # ESP32s listen on 8888
            broadcast_socket.sendto(discovery_request.encode('utf-8'), broadcast_addr)
            logger.info(f"Sent discovery request: {discovery_request} to {broadcast_addr}")
            
        finally:
            broadcast_socket.close()
            
    except Exception as e:
        logger.error(f"Discovery broadcast error: {e}")

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
    
    # Start the scheduler
    scheduler.start()
    logger.info("Flask-APScheduler started with discovery job (every 15 seconds)")
    next_run = scheduler.get_job('discovery_job').next_run_time
    logger.info(f"Next discovery run scheduled for: {next_run}")
    
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
    thread = threading.Thread(target=stream_camera, args=(name,), daemon=True)
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

def stream_camera(name):
    """Background thread function to continuously stream from a camera"""
    stream = camera_streams[name]
    frame_times = []
    
    while stream.is_active:
        try:
            # Get frame from ESP32 camera
            response = requests.get(stream.get_snapshot_url(), timeout=5)
            if response.status_code == 200:
                # Convert to base64 for WebSocket transmission
                frame_data = base64.b64encode(response.content).decode('utf-8')
                
                # Update stream statistics
                current_time = datetime.now()
                frame_times.append(current_time)
                
                # Keep only last 30 frame times for FPS calculation
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                if len(frame_times) > 1:
                    time_diff = (frame_times[-1] - frame_times[0]).total_seconds()
                    if time_diff > 0:
                        stream.fps = len(frame_times) / time_diff
                
                stream.last_frame = frame_data
                stream.last_frame_time = current_time
                stream.frame_count += 1
                
                # Emit frame to connected clients
                if stream.connected_clients:
                    socketio.emit('frame', {
                        'stream_name': name,
                        'frame_data': frame_data,
                        'timestamp': current_time.isoformat(),
                        'frame_count': stream.frame_count
                    }, to=name)
                
                # Limit frame rate to prevent overwhelming the network
                time.sleep(0.1)  # ~10 FPS
            else:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error streaming from {name}: {str(e)}")
            time.sleep(2)
    
    # Clean up when stream stops
    stream.is_active = False
    stream.last_frame = None
    stream.fps = 0

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

def udp_listener_thread_func():
    """Background thread function to continuously listen for ESP32 responses"""
    global udp_listener_socket, udp_listener_running
    
    try:
        # Create UDP socket for listening
        udp_listener_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_listener_socket.bind(('0.0.0.0', 8888))
        udp_listener_socket.settimeout(1.0)  # 1 second timeout for responsiveness
        
        logger.info("UDP listener started on port 8888")
        
        while udp_listener_running:
            try:
                data, addr = udp_listener_socket.recvfrom(1024)
                message = data.decode('utf-8').strip()
                logger.info(f"UDP listener received from {addr}: {message}")
                
                if message.startswith('ESP32_CAMERA:'):
                    parts = message.split(':')
                    if len(parts) >= 3:
                        ip = parts[1]
                        port = int(parts[2])
                        
                        # Generate camera name from IP
                        camera_name = f"ESP32 Camera ({ip})"
                        
                        # Check if camera already exists
                        if camera_name not in camera_streams:
                            camera_streams[camera_name] = CameraStream(camera_name, ip, port)
                            logger.info(f"Discovered new camera: {camera_name} at {ip}:{port}")
                            
                            # Emit discovery event to connected clients
                            socketio.emit('camera_discovered', {
                                'name': camera_name,
                                'ip': ip,
                                'port': port,
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            # Update last seen time
                            camera_streams[camera_name].discovered_time = datetime.now()
                            logger.debug(f"Camera {camera_name} still alive")
                            
            except socket.timeout:
                continue  # Normal timeout, continue listening
            except Exception as e:
                logger.error(f"UDP listener error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Failed to start UDP listener: {e}")
    finally:
        if udp_listener_socket:
            udp_listener_socket.close()
            logger.info("UDP listener socket closed")

def start_udp_listener():
    """Start the continuous UDP listener thread"""
    global udp_listener_thread, udp_listener_running
    
    if udp_listener_thread and udp_listener_thread.is_alive():
        logger.info("UDP listener already running")
        return
    
    udp_listener_running = True
    udp_listener_thread = threading.Thread(target=udp_listener_thread_func, daemon=True)
    udp_listener_thread.start()
    logger.info("UDP listener thread started")

def stop_udp_listener():
    """Stop the continuous UDP listener thread"""
    global udp_listener_running, udp_listener_socket
    
    udp_listener_running = False
    if udp_listener_socket:
        udp_listener_socket.close()
    logger.info("UDP listener stopped")

if __name__ == '__main__':
    # Start the continuous UDP listener first
    start_udp_listener()
    
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