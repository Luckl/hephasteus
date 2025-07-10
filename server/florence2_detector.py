import cv2
import numpy as np
import base64
import logging
import os
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont
import torch
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Initialize Florence 2 model
florence2_model = None
florence2_processor = None

def get_florence2_model():
    """Get or initialize Florence 2 model"""
    global florence2_model, florence2_processor
    
    if florence2_model is None:
        try:
            logger.info("Loading Florence 2 model...")
            logger.info("This may take several minutes on first load (model size: ~3.2GB)")
            
            # Set device and dtype
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            logger.info(f"Using device: {device}, dtype: {torch_dtype}")
            
            # Load Florence 2 model and processor based on official example
            model_name = "microsoft/Florence-2-large"
            logger.info(f"Downloading/loading model: {model_name}")
            
            logger.info("Loading model weights...")
            florence2_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch_dtype, 
                trust_remote_code=True
            ).to(device)
            logger.info("Model weights loaded successfully")
            
            logger.info("Loading processor...")
            florence2_processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            logger.info("Processor loaded successfully")
            
            logger.info(f"Florence 2 model loaded successfully on {device}")
                
        except Exception as e:
            logger.error(f"Failed to load Florence 2 model: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            florence2_model = None
            florence2_processor = None
    
    return florence2_model, florence2_processor

def detect_objects_with_florence2(frame):
    """Detect objects using Florence 2 with prompt-based approach"""
    try:
        model, processor = get_florence2_model()
        if model is None or processor is None:
            logger.warning("Florence 2 model not available")
            return []
        
        # Convert OpenCV frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Use dense region caption prompt
        prompt = "<DENSE_REGION_CAPTION>"
        
        # Prepare inputs
        inputs = processor(text=prompt, images=pil_image, return_tensors="pt")
        
        # Move inputs to device with appropriate dtypes
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Convert inputs to device with correct dtypes
        # input_ids should remain as integers, pixel_values as float
        processed_inputs = {}
        for k, v in inputs.items():
            if k == "input_ids":
                processed_inputs[k] = v.to(device, dtype=torch.long)  # Keep as integers
            else:
                processed_inputs[k] = v.to(device, dtype=torch_dtype)  # Convert to float
        
        # Run inference
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=processed_inputs["input_ids"],
                pixel_values=processed_inputs["pixel_values"],
                max_new_tokens=512,  # Reduced for faster inference
                num_beams=1,  # Reduced for faster inference
                do_sample=False,
                early_stopping=True
            )
        
        # Decode and post-process
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        logger.debug(f"Florence 2 raw output: {generated_text[:200]}...")
        
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task="<OD>", 
            image_size=(pil_image.width, pil_image.height)
        )
        logger.debug(f"Florence 2 parsed answer: {parsed_answer}")
        
        detections = []
        
        # Parse the object detection results
        if '<OD>' in parsed_answer:
            od_results = parsed_answer['<OD>']
            bboxes = od_results.get('bboxes', [])
            labels = od_results.get('labels', [])
            
            logger.debug(f"Florence 2 found {len(bboxes)} bboxes and {len(labels)} labels")
            
            for i, (bbox, label) in enumerate(zip(bboxes, labels)):
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Calculate width and height
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Validate bounding box
                    if width > 0 and height > 0:
                        detections.append({
                            'type': label,
                            'bbox': [x1, y1, width, height],
                            'confidence': 0.8,  # Florence 2 doesn't provide confidence scores in this format
                            'class_id': i
                        })
                        
                        logger.debug(f"Florence 2 {label} detected: bbox=[{x1}, {y1}, {width}, {height}]")
        else:
            logger.debug("No '<OD>' found in Florence 2 response")
        
        if detections:
            logger.info(f"Florence 2 detections found: {len(detections)}")
            # Log unique object types detected
            unique_types = set(det['type'] for det in detections)
            logger.info(f"Florence 2 object types: {', '.join(unique_types)}")
        else:
            logger.debug("No Florence 2 detections found in this frame")
        
        return detections
        
    except Exception as e:
        logger.error(f"Florence 2 detection error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []

def draw_florence2_detections(frame, detections):
    """Draw Florence 2 detection bounding boxes on frame"""
    for detection in detections:
        bbox = detection['bbox']
        x, y, w, h = bbox
        confidence = detection['confidence']
        detection_type = detection['type']
        
        # Choose color based on object type
        if detection_type.lower() in ['person', 'people', 'human']:
            color = (0, 255, 0)  # Green for persons
        elif detection_type.lower() in ['cup', 'bowl', 'fork', 'knife', 'spoon', 'bottle', 'wine glass', 'plate']:
            color = (255, 0, 0)  # Blue for kitchen items
        elif detection_type.lower() in ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'food']:
            color = (0, 255, 255)  # Yellow for food
        elif detection_type.lower() in ['chair', 'couch', 'bed', 'table', 'furniture']:
            color = (255, 0, 255)  # Magenta for furniture
        elif detection_type.lower() in ['tv', 'laptop', 'computer', 'phone', 'electronic']:
            color = (255, 255, 0)  # Cyan for electronics
        elif detection_type.lower() in ['book', 'clock', 'vase', 'scissors']:
            color = (128, 128, 128)  # Gray for miscellaneous
        else:
            color = (0, 165, 255)  # Orange for other objects
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Create label
        label = f"{detection_type}: {confidence:.2f}"
        
        # Get text size
        (w_text, h_text), t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y, h_text)
        
        # Draw text background
        cv2.rectangle(frame, (x, y - h_text), (x + w_text, y + t), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

def process_video_with_florence2(video_path, output_path, sample_interval=3):
    """Process a video file with Florence 2 detection and save with bounding boxes
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        sample_interval: Process every Nth frame (default: 3 for efficiency)
    """
    try:
        # Load Florence 2 model
        model, processor = get_florence2_model()
        if model is None or processor is None:
            return {"error": "Florence 2 model not available"}
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Could not open video: {video_path}"}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            return {"error": f"Could not create output video: {output_path}"}
        
        # Process frames with sampling for efficiency
        frame_count = 0
        detection_results = []
        total_detections = 0
        
        logger.info(f"Processing video with Florence 2: {video_path}")
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        logger.info(f"Sampling every {sample_interval} frames for efficiency")
        
        # Store all frames for output
        all_frames = []
        frame_detections = {}  # Map frame number to detections
        
        # First pass: collect all frames and detect on sampled frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            all_frames.append(frame.copy())
            
            # Log progress every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Florence 2 processing progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Detect objects on sampled frames only
            if frame_count % sample_interval == 0:
                detections = detect_objects_with_florence2(frame)
                frame_detections[frame_count] = detections
                
                # Store detection results
                if detections:
                    detection_results.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'detections': detections
                    })
                    total_detections += len(detections)
                    logger.debug(f"Frame {frame_count}: Found {len(detections)} detections")
            else:
                frame_detections[frame_count] = []
        
        # Second pass: write frames with detections
        logger.info("Writing output video with detections...")
        for i, frame in enumerate(all_frames, 1):
            # Get detections for this frame (use nearest sampled frame if not sampled)
            if i in frame_detections:
                detections = frame_detections[i]
            else:
                # Find nearest sampled frame
                nearest_sampled = ((i - 1) // sample_interval) * sample_interval + sample_interval
                if nearest_sampled <= len(all_frames):
                    detections = frame_detections.get(nearest_sampled, [])
                else:
                    detections = []
            
            # Draw detections on frame
            frame_with_boxes = draw_florence2_detections(frame.copy(), detections)
            
            # Write frame
            out.write(frame_with_boxes)
        
        # Clean up
        cap.release()
        out.release()
        
        logger.info(f"Florence 2 processing completed: {total_detections} total detections across {len(detection_results)} frames")
        
        # Save detection results
        results_path = output_path.replace('.mp4', '_florence2_detections.json')
        with open(results_path, 'w') as f:
            json.dump({
                'video_path': video_path,
                'output_path': output_path,
                'processing_time': datetime.now().isoformat(),
                'model': 'microsoft/Florence-2-large',
                'total_frames': frame_count,
                'frames_with_detections': len(detection_results),
                'total_detections': total_detections,
                'detection_results': detection_results
            }, f, indent=2)
        
        logger.info(f"Florence 2 video processing completed: {output_path}")
        logger.info(f"Detection results saved: {results_path}")
        
        return {
            'success': True,
            'output_path': output_path,
            'results_path': results_path,
            'total_frames': frame_count,
            'frames_with_detections': len(detection_results),
            'total_detections': total_detections,
            'model': 'microsoft/Florence-2-large'
        }
        
    except Exception as e:
        logger.error(f"Florence 2 video processing error: {e}")
        return {"error": str(e)} 