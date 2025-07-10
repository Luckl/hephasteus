import base64
import logging
import requests
import threading
import time
from datetime import datetime
from flask_socketio import SocketIO

# Import our modules
from computer_vision import process_frame_for_display, process_frame_for_recording
from video_recorder import process_frame_for_recording as record_frame

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
        
        # Computer vision settings
        self.cv_enabled = True
        self.cv_detections = []
        self.last_detection_time = None
        
        # Video recording settings
        self.recording_enabled = True
        
        # Background detection settings
        self.background_detection_active = False
        self.background_thread = None
        
    def get_stream_url(self):
        return f"http://{self.ip_address}:{self.port + 1}/stream"
    
    def get_snapshot_url(self):
        return f"{self.base_url}/capture"
    
    def get_status_url(self):
        return f"{self.base_url}/status"
    
    def toggle_cv(self, enabled=None):
        """Toggle computer vision processing"""
        if enabled is not None:
            self.cv_enabled = enabled
        else:
            self.cv_enabled = not self.cv_enabled
        logger.info(f"Computer vision {'enabled' if self.cv_enabled else 'disabled'} for {self.name}")
        return self.cv_enabled
    
    def toggle_recording(self, enabled=None):
        """Toggle video recording"""
        if enabled is not None:
            self.recording_enabled = enabled
        else:
            self.recording_enabled = not self.recording_enabled
        logger.info(f"Video recording {'enabled' if self.recording_enabled else 'disabled'} for {self.name}")
        return self.recording_enabled
    
    def start_background_detection(self):
        """Start background detection thread"""
        if self.background_detection_active:
            return
        
        self.background_detection_active = True
        self.background_thread = threading.Thread(
            target=self._background_detection_loop,
            daemon=True
        )
        self.background_thread.start()
        logger.info(f"Started background detection for {self.name}")
    
    def stop_background_detection(self):
        """Stop background detection thread"""
        self.background_detection_active = False
        if self.background_thread:
            self.background_thread.join(timeout=2)
        logger.info(f"Stopped background detection for {self.name}")
    
    def _background_detection_loop(self):
        """Background thread for continuous detection (monitoring only)"""
        detection_interval = 0.2  # Check every 200ms (increased frequency)
        
        while self.background_detection_active and self.is_active:
            try:
                # Get a snapshot for detection
                response = requests.get(self.get_snapshot_url(), timeout=5)
                if response.status_code == 200:
                    # Convert to base64
                    frame_data = base64.b64encode(response.content).decode('utf-8')
                    
                    # Process with computer vision (for monitoring only)
                    if self.cv_enabled:
                        try:
                            # Use recording processing for background detection (clean frame, person detections only)
                            clean_frame, detections = process_frame_for_recording(frame_data)
                            self.cv_detections = detections
                            self.last_detection_time = datetime.now()
                            
                            # Log significant detections
                            if detections:
                                detection_types = [d['type'] for d in detections]
                                logger.info(f"Background detection for {self.name}: {detection_types}")
                                
                        except Exception as e:
                            logger.error(f"Background detection error for {self.name}: {e}")
                
                # Wait before next detection
                time.sleep(detection_interval)
                
            except requests.exceptions.RequestException as e:
                logger.debug(f"Background detection request error for {self.name}: {e}")
                time.sleep(detection_interval)
            except Exception as e:
                logger.error(f"Background detection loop error for {self.name}: {e}")
                time.sleep(detection_interval)

def stream_camera(name, camera_streams, stream_threads, socketio):
    """Background thread function to continuously stream from a camera using MJPEG"""
    stream = camera_streams[name]
    frame_times = []
    
    # Start background detection
    stream.start_background_detection()
    
    # Use MJPEG stream instead of individual snapshots for better performance
    stream_url = stream.get_stream_url()
    
    logger.info(f"Starting MJPEG stream from {stream_url}")
    logger.info(f"Camera base URL: {stream.base_url}")
    logger.info(f"Camera IP: {stream.ip_address}, Port: {stream.port}")
    
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
            
            logger.info(f"Attempting to connect to stream at {stream_url}")
            response = requests.get(stream_url, stream=True, timeout=30, headers=headers)
            logger.info(f"Stream connection response status: {response.status_code}")
            
            if response.status_code == 200:
                logger.info(f"Successfully connected to MJPEG stream at {stream_url}")
                
                # Process MJPEG stream - simplified approach
                buffer = b''
                in_image = False
                frame_count = 0
                
                for chunk in response.iter_content(chunk_size=1024):
                    if not stream.is_active:
                        logger.info(f"Stream stopped for {name}")
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
                            frame_count += 1
                            logger.debug(f"Received frame {frame_count} from {name} ({len(image_data)} bytes)")
                            
                            # Convert to base64 for processing
                            frame_data = base64.b64encode(image_data).decode('utf-8')
                            
                            # Apply computer vision processing if enabled (for display)
                            detections = []
                            display_frame_data = frame_data  # Default to original frame
                            if stream.cv_enabled:
                                try:
                                    # Process frame for display (with bounding boxes)
                                    processed_frame, detections = process_frame_for_display(frame_data)
                                    # Use detections from background thread if available and more recent
                                    if stream.cv_detections and stream.last_detection_time:
                                        # Use background detections if they're recent (within last 5 seconds)
                                        time_diff = (datetime.now() - stream.last_detection_time).total_seconds()
                                        if time_diff < 5.0:
                                            detections = stream.cv_detections
                                    display_frame_data = processed_frame
                                except Exception as e:
                                    logger.error(f"Computer vision display processing error for {name}: {e}")
                                    # Continue with original frame if CV fails
                            
                            # Process video recording if enabled (using clean frames without bounding boxes)
                            if stream.recording_enabled:
                                try:
                                    current_time = datetime.now()
                                    # Use detections from background thread for recording decisions
                                    recording_detections = stream.cv_detections if stream.cv_detections else detections
                                    # Use clean frame data (original frame without bounding boxes) for recording
                                    record_frame(name, frame_data, recording_detections, current_time)
                                except Exception as e:
                                    logger.error(f"Video recording error for {name}: {e}")
                            
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
                            
                            stream.last_frame = display_frame_data  # Store display frame (with boxes) for UI
                            stream.last_frame_time = current_time
                            stream.frame_count += 1
                            
                            # Emit frame to connected clients (display frame with bounding boxes)
                            if stream.connected_clients:
                                socketio.emit('frame', {
                                    'stream_name': name,
                                    'frame_data': display_frame_data,  # Display frame with boxes
                                    'timestamp': current_time.isoformat(),
                                    'frame_count': stream.frame_count,
                                    'detections': stream.cv_detections if stream.cv_enabled else []
                                }, to=name)
                            
                            # No artificial delay - let it run as fast as possible
                            
            else:
                logger.error(f"Failed to connect to stream: HTTP {response.status_code}")
                logger.error(f"Response headers: {dict(response.headers)}")
                time.sleep(2)
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error to {stream.ip_address}: {str(e)}")
            logger.error(f"Tried to connect to: {stream_url}")
            time.sleep(2)
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error to {stream.ip_address}: {str(e)}")
            logger.error(f"Tried to connect to: {stream_url}")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error streaming from {name}: {str(e)}")
            logger.error(f"Tried to connect to: {stream_url}")
            time.sleep(2)
    
    # Clean up when stream stops
    stream.stop_background_detection()
    stream.is_active = False
    stream.last_frame = None
    stream.fps = 0
    logger.info(f"Stream stopped for {name}") 