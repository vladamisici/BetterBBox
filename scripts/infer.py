"""
Simplified Inference Script - Works immediately with YOLOv8
This is a working version that can be used right away without training
"""

import os
import argparse
from ultralytics import YOLO
import cv2
import glob
from pathlib import Path
import numpy as np


def draw_boxes(image, results):
    """Draw bounding boxes on image"""
    # Get the first result
    if len(results) > 0:
        result = results[0]
        
        # Check if there are any detections
        if result.boxes is not None and len(result.boxes) > 0:
            # Get boxes, classes, and confidences
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            # Define class names (using COCO classes for default YOLO)
            # You can modify these to match your document classes
            class_names = result.names
            
            # Draw each box
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create label
                label = f"{class_names[int(cls)]}: {conf:.2f}"
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 10
                
                cv2.rectangle(image,
                            (x1, label_y - label_size[1] - 5),
                            (x1 + label_size[0] + 5, label_y + 5),
                            (0, 255, 0), -1)
                
                # Draw label text
                cv2.putText(image, label,
                          (x1 + 2, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (0, 0, 0), 1)
    
    return image


def main():
    parser = argparse.ArgumentParser(description='Run inference with YOLO for document detection')
    parser.add_argument('--weights', default='models/yolov8m.pt',
                        help='Path to model weights (.pt)')
    parser.add_argument('--source', required=True,
                        help='Image or directory to run inference on')
    parser.add_argument('--output', default='results',
                        help='Directory to save results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--device', default='0',
                        help='Device to run on (0 for GPU, cpu for CPU)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.weights}...")
    try:
        model = YOLO(args.weights)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have run setup.py first to download the model!")
        return
    
    # Set device
    if args.device == 'cpu':
        model.to('cpu')
    else:
        model.to(f'cuda:{args.device}')
    
    # Collect image paths
    if os.path.isdir(args.source):
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.bmp']:
            image_paths.extend(glob.glob(os.path.join(args.source, ext)))
        print(f"Found {len(image_paths)} images in directory")
    else:
        image_paths = [args.source]
    
    # Process each image
    for img_path in image_paths:
        print(f"\nProcessing: {img_path}")
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            continue
        
        # Run inference
        results = model(img, conf=args.conf)
        
        # Draw boxes
        img_with_boxes = draw_boxes(img.copy(), results)
        
        # Save result
        output_path = os.path.join(args.output, f"detected_{Path(img_path).name}")
        cv2.imwrite(output_path, img_with_boxes)
        print(f"Saved result to: {output_path}")
        
        # Print detection summary
        if len(results) > 0 and results[0].boxes is not None:
            num_detections = len(results[0].boxes)
            print(f"Detected {num_detections} objects")
            
            # Print each detection
            for i, (cls, conf) in enumerate(zip(results[0].boxes.cls, results[0].boxes.conf)):
                class_name = results[0].names[int(cls)]
                print(f"  {i+1}: {class_name} (confidence: {conf:.3f})")
        else:
            print("No objects detected")
    
    print(f"\nAll results saved to: {args.output}/")


if __name__ == '__main__':
    main()