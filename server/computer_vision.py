import cv2
import numpy as np
import base64
import logging
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)

class ComputerVisionProcessor:
    def __init__(self):
        self.face_cascade = None
        self.person_cascade = None
        self.motion_detector = None
        self.previous_frame = None
        self.motion_threshold = 25
        self.min_motion_area = 500
        self.confidence_threshold = 0.8  # 80% confidence threshold
        
        # Initialize OpenCV models
        self._load_models()
        
    def _load_models(self):
        """Load OpenCV cascade classifiers and models"""
        try:
            # Face detection - using Haar cascade (built into OpenCV)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Person detection - using HOG (Histogram of Oriented Gradients)
            self.person_detector = cv2.HOGDescriptor()
            self.person_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Motion detection - using background subtraction
            self.motion_detector = cv2.createBackgroundSubtractorMOG2(
                history=100, 
                varThreshold=40, 
                detectShadows=False
            )
            
            logger.info("Computer vision models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading computer vision models: {e}")
    
    def detect_faces(self, image):
        """Detect faces in the image"""
        if self.face_cascade is None:
            return []
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Convert to list of dictionaries
            face_detections = []
            for (x, y, w, h) in faces:
                face_detections.append({
                    'type': 'face',
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': 0.8  # Haar cascade doesn't provide confidence
                })
            
            return face_detections
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def detect_persons(self, image):
        """Detect persons in the image"""
        if self.person_detector is None:
            return []
        
        try:
            # Detect persons
            boxes, weights = self.person_detector.detectMultiScale(
                image,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )
            
            # Convert to list of dictionaries
            person_detections = []
            for (x, y, w, h), weight in zip(boxes, weights):
                person_detections.append({
                    'type': 'person',
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': float(weight)
                })
            
            return person_detections
            
        except Exception as e:
            logger.error(f"Error detecting persons: {e}")
            return []
    
    def detect_motion(self, image):
        """Detect motion in the image"""
        if self.motion_detector is None:
            return []
        
        try:
            # Apply background subtraction
            fgmask = self.motion_detector.apply(image)
            
            # Find contours of moving objects
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detections = []
            for contour in contours:
                # Filter by area to avoid noise
                area = cv2.contourArea(contour)
                if area > self.min_motion_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = min(area / 1000, 1.0)  # Normalize confidence
                    
                    # Only include if confidence meets threshold
                    if confidence >= self.confidence_threshold:
                        motion_detections.append({
                            'type': 'motion',
                            'bbox': [int(x), int(y), int(w), int(h)],
                            'confidence': confidence,
                            'area': area
                        })
            
            return motion_detections
            
        except Exception as e:
            logger.error(f"Error detecting motion: {e}")
            return []
    
    def filter_detections_by_confidence(self, detections):
        """Filter detections to only include those above confidence threshold"""
        return [detection for detection in detections if detection.get('confidence', 0) >= self.confidence_threshold]
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on the image"""
        try:
            # Filter detections by confidence threshold
            filtered_detections = self.filter_detections_by_confidence(detections)
            
            # Color map for different detection types
            colors = {
                'face': (0, 255, 0),      # Green
                'person': (255, 0, 0),    # Blue
                'animal': (0, 0, 255),    # Red
                'motion': (255, 255, 0)   # Cyan
            }
            
            for detection in filtered_detections:
                detection_type = detection['type']
                bbox = detection['bbox']
                confidence = detection.get('confidence', 0.0)
                
                x, y, w, h = bbox
                color = colors.get(detection_type, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # Draw label with confidence
                label = f"{detection_type.capitalize()}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(image, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), color, -1)
                
                # Draw label text
                cv2.putText(image, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return image
            
        except Exception as e:
            logger.error(f"Error drawing detections: {e}")
            return image
    
    def process_frame(self, image_data):
        """Process a frame and return detections with bounding boxes"""
        try:
            # Decode base64 image
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            
            # Convert to OpenCV format
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image")
                return image_data, []
            
            # Run all detections
            all_detections = []
            
            # Face detection
            faces = self.detect_faces(image)
            all_detections.extend(faces)
            
            # Person detection
            persons = self.detect_persons(image)
            all_detections.extend(persons)
            
            # Motion detection
            motion = self.detect_motion(image)
            all_detections.extend(motion)
            
            # Filter detections by confidence threshold
            filtered_detections = self.filter_detections_by_confidence(all_detections)
            
            # Draw detections on image (only high-confidence ones)
            if filtered_detections:
                image = self.draw_detections(image, all_detections)  # Pass all detections for drawing
            
            # Convert back to base64
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            processed_image_data = base64.b64encode(buffer).decode('utf-8')
            
            return processed_image_data, filtered_detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return image_data, []

# Global instance
cv_processor = None

def get_cv_processor():
    """Get or create the computer vision processor instance"""
    global cv_processor
    if cv_processor is None:
        cv_processor = ComputerVisionProcessor()
    return cv_processor

def process_frame_with_cv(image_data):
    """Process a frame with computer vision and return processed image with detections"""
    processor = get_cv_processor()
    return processor.process_frame(image_data) 