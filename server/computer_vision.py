import cv2
import numpy as np
import base64
import logging
import os
from ultralytics import YOLO
import torch
import ultralytics

logger = logging.getLogger(__name__)

# Initialize YOLOv8 model for person detection
yolo_model = None

def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        try:
            torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel, torch.nn.modules.container.Sequential, ultralytics.nn.modules.Conv, torch.nn.modules.linear.Linear])

            
            logger.info("Loading YOLOv11 model...")
            # Use YOLOv8n (nano) model for person detection - lightweight and fast
            yolo_model = YOLO('yolo11n.pt')
            logger.info("YOLOv11 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            yolo_model = None
    return yolo_model

def detect_persons(frame):
    """Detect persons using YOLOv11"""
    try:
        model = get_yolo_model()
        if model is None:
            logger.warning("YOLOv8 model not available, skipping person detection")
            return []
        
        # Run YOLOv8 inference
        results = model(frame, verbose=False)
        
        results_list = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # YOLOv8 COCO classes - class 0 is 'person'
                    if class_id == 0:  # Person class
                        # Apply confidence threshold
                        if confidence >= 0.5:  # Lower threshold for YOLOv8
                            # Convert to integer coordinates
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Calculate width and height
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Validate bounding box
                            if width > 0 and height > 0:
                                results_list.append({
                                    'type': 'person',
                                    'bbox': [x1, y1, width, height],
                                    'confidence': confidence
                                })
                                
                                logger.debug(f"YOLOv8 person detected: confidence={confidence:.3f}, bbox=[{x1}, {y1}, {width}, {height}]")
        
        if results_list:
            logger.info(f"YOLOv8 detections found: {len(results_list)}")
        
        return results_list
        
    except Exception as e:
        logger.error(f"YOLOv8 detection error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return []

def detect_motion(frame, prev_frame=None):
    """Detect motion using frame differencing"""
    if prev_frame is None:
        return []
    
    # Convert frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate frame difference
    frame_diff = cv2.absdiff(prev_gray, gray)
    
    # Apply threshold to get motion mask
    _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in motion areas
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum motion area
            x, y, w, h = cv2.boundingRect(contour)
            confidence = min(0.8, area / (frame.shape[0] * frame.shape[1]) * 20)
            detections.append({
                'type': 'motion',
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': float(confidence)
            })
    
    return detections

def draw_detections(frame, detections):
    """Draw detection bounding boxes on frame"""
    for detection in detections:
        bbox = detection['bbox']
        x, y, w, h = bbox
        confidence = detection['confidence']
        detection_type = detection['type']
        
        # Draw bounding box around the detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Get class label
        label = f"{detection_type}: {confidence:.2f}"
        
        # Get width and text of the label string
        (w_text, h_text), t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y, h_text)
        
        # Draw bounding box around the text
        cv2.rectangle(frame, (x, y - h_text), (x + w_text, y + t), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame

def process_frame_with_cv(frame_data):
    """Process frame with computer vision"""
    try:
        # Decode base64 frame
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return frame_data, []
        
        # Run all detection methods
        all_detections = []
        
        # Person detection using YOLOv8
        person_detections = detect_persons(frame)
        if person_detections:
            logger.info(f"Person detections found: {len(person_detections)}")
            for det in person_detections:
                logger.info(f"Detection: {det}")
        all_detections.extend(person_detections)
        
        # Draw detections on frame
        frame_with_boxes = draw_detections(frame.copy(), all_detections)
        
        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', frame_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 85])
        processed_frame_data = base64.b64encode(buffer).decode('utf-8')
        
        return processed_frame_data, all_detections
        
    except Exception as e:
        logger.error(f"Computer vision processing error: {e}")
        return frame_data, [] 