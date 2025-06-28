import base64
import logging
import requests
import threading
import time
from datetime import datetime
from flask_socketio import SocketIO

logger = logging.getLogger(__name__)

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
        return f"http://{self.ip_address}:{self.port + 1}/stream"
    
    def get_snapshot_url(self):
        return f"{self.base_url}/capture"
    
    def get_status_url(self):
        return f"{self.base_url}/status"

def stream_camera(name, camera_streams, stream_threads, socketio):
    """Background thread function to continuously stream from a camera using MJPEG"""
    stream = camera_streams[name]
    frame_times = []
    
    # Use MJPEG stream instead of individual snapshots for better performance
    stream_url = stream.get_stream_url()
    
    logger.info(f"Starting MJPEG stream from {stream_url}")
    
    while stream.is_active:
        try:
            # Use streaming endpoint with browser-compatible headers
            headers = {
                'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Cache-Control': 'no-cache',
                'Origin': f'http://{stream.ip_address}',
                'Pragma': 'no-cache',
                'Referer': f'http://{stream.ip_address}/',
            }
            
            response = requests.get(stream_url, stream=True, timeout=30, headers=headers)
            if response.status_code == 200:
                logger.info(f"Successfully connected to MJPEG stream at {stream_url}")
                
                # Process MJPEG stream - simplified approach
                buffer = b''
                in_image = False
                
                for chunk in response.iter_content(chunk_size=1024):
                    if not stream.is_active:
                        break
                    
                    buffer += chunk
                    
                    # Look for JPEG start marker
                    if b'\xff\xd8' in buffer and not in_image:
                        # Start of JPEG image
                        jpeg_start = buffer.find(b'\xff\xd8')
                        buffer = buffer[jpeg_start:]
                        in_image = True
                        continue
                    
                    # Look for JPEG end marker
                    if in_image and b'\xff\xd9' in buffer:
                        # End of JPEG image
                        jpeg_end = buffer.find(b'\xff\xd9') + 2
                        image_data = buffer[:jpeg_end]
                        buffer = buffer[jpeg_end:]
                        in_image = False
                        
                        if len(image_data) > 100:  # Ensure we have a valid image
                            # Convert to base64 for WebSocket transmission
                            frame_data = base64.b64encode(image_data).decode('utf-8')
                            
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
                            
                            # No artificial delay - let it run as fast as possible
                            
            else:
                logger.error(f"Failed to connect to stream: HTTP {response.status_code}")
                time.sleep(2)
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to {stream.ip_address}: {str(e)}")
            time.sleep(2)
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error to {stream.ip_address}: {str(e)}")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error streaming from {name}: {str(e)}")
            time.sleep(2)
    
    # Clean up when stream stops
    stream.is_active = False
    stream.last_frame = None
    stream.fps = 0
    logger.info(f"Stream stopped for {name}") 