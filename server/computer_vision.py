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
    """Detect persons using YOLOv11 - returns only person detections for recording triggers"""
    try:
        model = get_yolo_model()
        if model is None:
            logger.warning("YOLOv8 model not available, skipping person detection")
            return []
        
        # Run YOLOv8 inference
        results = model(frame, verbose=False)
        
        person_detections = []
        
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
                                person_detections.append({
                                    'type': 'person',
                                    'bbox': [x1, y1, width, height],
                                    'confidence': confidence
                                })
                                
                                logger.debug(f"YOLOv8 person detected: confidence={confidence:.3f}, bbox=[{x1}, {y1}, {width}, {height}]")
        
        if person_detections:
            logger.info(f"YOLOv8 person detections found: {len(person_detections)}")
        
        return person_detections
        
    except Exception as e:
        logger.error(f"YOLOv8 detection error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return []

def detect_all_objects(frame):
    """Detect all objects using YOLOv11 - returns all detections for visualization"""
    try:
        model = get_yolo_model()
        if model is None:
            logger.warning("YOLOv8 model not available, skipping object detection")
            return []
        
        # COCO class names for YOLOv8
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
        
        # Run YOLOv8 inference
        results = model(frame, verbose=False)
        
        all_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Apply confidence threshold for all objects
                    if confidence >= 0.3:  # Lower threshold to see more objects
                        # Convert to integer coordinates
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Calculate width and height
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Validate bounding box
                        if width > 0 and height > 0:
                            # Get class name
                            class_name = coco_classes[class_id] if class_id < len(coco_classes) else f"class_{class_id}"
                            
                            all_detections.append({
                                'type': class_name,
                                'bbox': [x1, y1, width, height],
                                'confidence': confidence,
                                'class_id': class_id
                            })
                            
                            logger.debug(f"YOLOv8 {class_name} detected: confidence={confidence:.3f}, bbox=[{x1}, {y1}, {width}, {height}]")
        
        if all_detections:
            logger.info(f"YOLOv8 total detections found: {len(all_detections)}")
            # Log unique object types detected
            unique_types = set(det['type'] for det in all_detections)
            logger.info(f"Object types detected: {', '.join(unique_types)}")
        
        return all_detections
        
    except Exception as e:
        logger.error(f"YOLOv8 object detection error: {e}")
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
    """Draw detection bounding boxes on frame with different colors for different object types"""
    for detection in detections:
        bbox = detection['bbox']
        x, y, w, h = bbox
        confidence = detection['confidence']
        detection_type = detection['type']
        
        # Choose color based on object type
        if detection_type == 'person':
            color = (0, 255, 0)  # Green for persons
        elif detection_type in ['cup', 'bowl', 'fork', 'knife', 'spoon', 'bottle', 'wine glass']:
            color = (255, 0, 0)  # Blue for kitchen utensils
        elif detection_type in ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake']:
            color = (0, 255, 255)  # Yellow for food items
        elif detection_type in ['chair', 'couch', 'bed', 'dining table', 'toilet']:
            color = (255, 0, 255)  # Magenta for furniture
        elif detection_type in ['tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone']:
            color = (255, 255, 0)  # Cyan for electronics
        elif detection_type in ['book', 'clock', 'vase', 'scissors']:
            color = (128, 128, 128)  # Gray for miscellaneous objects
        else:
            color = (0, 165, 255)  # Orange for other objects
        
        # Draw bounding box around the detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Get class label
        label = f"{detection_type}: {confidence:.2f}"
        
        # Get width and text of the label string
        (w_text, h_text), t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y, h_text)
        
        # Draw bounding box around the text
        cv2.rectangle(frame, (x, y - h_text), (x + w_text, y + t), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

def process_frame_for_display(frame_data):
    """Process frame with computer vision for display (with bounding boxes)"""
    try:
        # Decode base64 frame
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return frame_data, []
        
        # Get person detections for recording triggers (only persons)
        person_detections = detect_persons(frame)
        if person_detections:
            logger.info(f"Person detections found: {len(person_detections)}")
            for det in person_detections:
                logger.info(f"Person detection: {det}")
        
        # Get all object detections for visualization (including persons)
        all_object_detections = detect_all_objects(frame)
        
        # Draw all object detections on frame
        frame_with_boxes = draw_detections(frame.copy(), all_object_detections)
        
        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', frame_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 85])
        processed_frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # Return processed frame with boxes for display, and person detections for recording triggers
        return processed_frame_data, person_detections
        
    except Exception as e:
        logger.error(f"Computer vision display processing error: {e}")
        return frame_data, []

def process_frame_for_recording(frame_data):
    """Process frame for recording (clean frame without bounding boxes)"""
    try:
        # Decode base64 frame
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return frame_data, []
        
        # Get person detections for recording triggers (only persons)
        person_detections = detect_persons(frame)
        if person_detections:
            logger.info(f"Person detections found for recording: {len(person_detections)}")
            for det in person_detections:
                logger.info(f"Person detection for recording: {det}")
        
        # Return clean frame (no bounding boxes) for recording, and person detections for triggers
        # Encode the original frame without any modifications
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        clean_frame_data = base64.b64encode(buffer).decode('utf-8')
        
        return clean_frame_data, person_detections
        
    except Exception as e:
        logger.error(f"Computer vision recording processing error: {e}")
        return frame_data, [] 