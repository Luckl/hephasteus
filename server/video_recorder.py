import cv2
import numpy as np
import base64
import logging
import threading
import time
import os
from datetime import datetime
from collections import deque
import json

logger = logging.getLogger(__name__)

class VideoRecorder:
    def __init__(self, output_dir="recordings", buffer_seconds=1, max_buffer_size=30, min_recording_duration=1.0):
        self.output_dir = output_dir
        self.buffer_seconds = buffer_seconds
        self.max_buffer_size = max_buffer_size
        self.min_recording_duration = min_recording_duration  # Minimum recording duration in seconds
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Frame buffers and recording states
        self.frame_buffers = {}
        self.recording_states = {}
        self.video_writers = {}
        self.lock = threading.Lock()
        
        # Add cooldown tracking to prevent premature stops
        self.last_detection_times = {}  # Track when we last saw a detection for each stream
        
        logger.info(f"Video recorder initialized: {output_dir}, min duration: {min_recording_duration}s")
    
    def add_frame_to_buffer(self, stream_name, frame_data, timestamp):
        """Add frame to buffer"""
        with self.lock:
            if stream_name not in self.frame_buffers:
                self.frame_buffers[stream_name] = deque(maxlen=self.max_buffer_size)
            
            self.frame_buffers[stream_name].append({
                'frame_data': frame_data,
                'timestamp': timestamp
            })
    
    def should_start_recording(self, detections):
        """Check if recording should start"""
        if not detections:
            return False
        
        for detection in detections:
            if detection['type'] in ['face', 'person'] and detection.get('confidence', 0) >= 0.8:
                return True
        return False
    
    def should_stop_recording(self, stream_name, detections):
        """Check if recording should stop"""
        if stream_name not in self.recording_states:
            return False
        
        # Check if we have current detections
        has_current_detections = False
        if detections:
            for detection in detections:
                if detection['type'] in ['face', 'person'] and detection.get('confidence', 0) >= 0.8:
                    has_current_detections = True
                    # Update last detection time
                    self.last_detection_times[stream_name] = datetime.now()
                    break
        
        # If we have current detections, don't stop
        if has_current_detections:
            return False
        
        # If no current detections, check cooldown period
        if stream_name in self.last_detection_times:
            time_since_last_detection = (datetime.now() - self.last_detection_times[stream_name]).total_seconds()
            # Keep recording for 3 seconds after last detection to avoid premature stops
            if time_since_last_detection < 3.0:
                return False
        
        # Stop recording if no detections and cooldown period has passed
        return True
    
    def start_recording(self, stream_name):
        """Start recording"""
        with self.lock:
            if stream_name in self.recording_states:
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{stream_name}_{timestamp}.mp4"
            filepath = os.path.join(self.output_dir, filename)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(filepath, fourcc, 10, (640, 480))
            
            if not video_writer.isOpened():
                logger.error(f"Failed to create video writer for {stream_name}")
                return
            
            self.recording_states[stream_name] = {
                'start_time': datetime.now(),
                'filepath': filepath,
                'frame_count': 0,
                'detection_types': set(),
                'temp_filepath': filepath  # Store original path for potential deletion
            }
            
            # Initialize last detection time when starting recording
            self.last_detection_times[stream_name] = datetime.now()
            
            self.video_writers[stream_name] = video_writer
            logger.info(f"Started recording: {filepath}")
    
    def stop_recording(self, stream_name):
        """Stop recording"""
        with self.lock:
            if stream_name not in self.recording_states:
                return
            
            recording_info = self.recording_states[stream_name]
            video_writer = self.video_writers.get(stream_name)
            
            if video_writer:
                video_writer.release()
                del self.video_writers[stream_name]
            
            duration = (datetime.now() - recording_info['start_time']).total_seconds()
            
            # Check if recording duration meets minimum threshold
            if duration < self.min_recording_duration:
                # Delete the short recording file
                try:
                    if os.path.exists(recording_info['filepath']):
                        os.remove(recording_info['filepath'])
                        logger.info(f"Deleted short recording ({duration:.1f}s < {self.min_recording_duration}s): {recording_info['filepath']}")
                except Exception as e:
                    logger.error(f"Error deleting short recording: {e}")
                
                # Don't save metadata for short recordings
                del self.recording_states[stream_name]
                return
            
            # Save metadata for valid recordings
            metadata = {
                'stream_name': stream_name,
                'start_time': recording_info['start_time'].isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': duration,
                'frame_count': recording_info['frame_count'],
                'detection_types': list(recording_info['detection_types'])
            }
            
            metadata_path = recording_info['filepath'].replace('.mp4', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Stopped recording: {duration:.1f}s, {recording_info['frame_count']} frames")
            del self.recording_states[stream_name]
            
            # Clean up last detection time
            if stream_name in self.last_detection_times:
                del self.last_detection_times[stream_name]
    
    def write_buffered_frames(self, stream_name):
        """Write buffered frames to recording"""
        with self.lock:
            if stream_name not in self.recording_states:
                return
            
            video_writer = self.video_writers[stream_name]
            buffer = self.frame_buffers.get(stream_name, deque())
            
            for frame_info in buffer:
                try:
                    frame_bytes = base64.b64decode(frame_info['frame_data'])
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        frame = cv2.resize(frame, (640, 480))
                        video_writer.write(frame)
                        self.recording_states[stream_name]['frame_count'] += 1
                        
                except Exception as e:
                    logger.error(f"Error writing buffered frame: {e}")
    
    def process_frame(self, stream_name, frame_data, detections, timestamp):
        """Process frame for recording"""
        # Add to buffer
        self.add_frame_to_buffer(stream_name, frame_data, timestamp)
        
        # Check if should start recording
        if self.should_start_recording(detections):
            if stream_name not in self.recording_states:
                logger.info(f"Starting recording for {stream_name}")
                self.start_recording(stream_name)
                self.write_buffered_frames(stream_name)
        
        # Write current frame if recording
        if stream_name in self.recording_states:
            try:
                frame_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    frame = cv2.resize(frame, (640, 480))
                    self.video_writers[stream_name].write(frame)
                    self.recording_states[stream_name]['frame_count'] += 1
                    
                    # Update detection types
                    for detection in detections:
                        if detection['type'] in ['face', 'person']:
                            self.recording_states[stream_name]['detection_types'].add(detection['type'])
                            
            except Exception as e:
                logger.error(f"Error writing frame: {e}")
        
        # Check if should stop recording
        if self.should_stop_recording(stream_name, detections):
            if stream_name in self.recording_states:
                logger.info(f"Stopping recording for {stream_name}")
                self.stop_recording(stream_name)
    
    def get_recording_status(self, stream_name):
        """Get recording status"""
        with self.lock:
            if stream_name in self.recording_states:
                recording_info = self.recording_states[stream_name]
                duration = (datetime.now() - recording_info['start_time']).total_seconds()
                return {
                    'is_recording': True,
                    'start_time': recording_info['start_time'].isoformat(),
                    'duration_seconds': duration,
                    'frame_count': recording_info['frame_count'],
                    'detection_types': list(recording_info['detection_types']),
                    'min_duration': self.min_recording_duration
                }
            else:
                return {
                    'is_recording': False,
                    'buffer_frames': len(self.frame_buffers.get(stream_name, deque())),
                    'min_duration': self.min_recording_duration
                }
    
    def get_all_recordings(self):
        """Get list of recordings"""
        recordings = []
        
        try:
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.mp4'):
                    filepath = os.path.join(self.output_dir, filename)
                    stat = os.stat(filepath)
                    
                    metadata = {}
                    metadata_path = filepath.replace('.mp4', '_metadata.json')
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                        except:
                            pass
                    
                    recordings.append({
                        'filename': filename,
                        'filepath': filepath,
                        'file_size_mb': stat.st_size / (1024 * 1024),
                        'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'metadata': metadata
                    })
            
            recordings.sort(key=lambda x: x['created_time'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting recordings: {e}")
        
        return recordings

# Global instance
video_recorder = None

def get_video_recorder():
    """Get video recorder instance"""
    global video_recorder
    if video_recorder is None:
        video_recorder = VideoRecorder()
    return video_recorder

def process_frame_for_recording(stream_name, frame_data, detections, timestamp):
    """Process frame for recording"""
    recorder = get_video_recorder()
    recorder.process_frame(stream_name, frame_data, detections, timestamp) 