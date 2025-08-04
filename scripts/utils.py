"""
Simple utility functions for document detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def draw_boxes(image, boxes, labels=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image (numpy array)
        boxes: List of boxes in format [[x1, y1, x2, y2], ...]
        labels: Optional list of labels for each box
        color: Box color (B, G, R)
        thickness: Line thickness
    
    Returns:
        Image with drawn boxes
    """
    img = image.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if labels and i < len(labels):
            label = labels[i]
            
            # Get text size
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
            cv2.rectangle(img,
                         (x1, label_y - label_size[1] - 4),
                         (x1 + label_size[0] + 4, label_y + 4),
                         color, -1)
            
            # Draw label text
            cv2.putText(img, label,
                       (x1 + 2, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
    
    return img


def plot_results(image, boxes, labels=None, title="Detection Results", save_path=None):
    """
    Plot detection results using matplotlib
    
    Args:
        image: Input image
        boxes: List of bounding boxes
        labels: Optional list of labels
        title: Plot title
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    
    # Display image
    if image.shape[2] == 3:
        # Convert BGR to RGB if needed
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image_rgb)
    else:
        ax.imshow(image)
    
    # Define colors for different classes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(boxes)))
    
    # Draw boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle
        rect = patches.Rectangle((x1, y1), width, height,
                               linewidth=2, edgecolor=colors[i],
                               facecolor='none')
        ax.add_patch(rect)
        
        # Add label if provided
        if labels and i < len(labels):
            ax.text(x1, y1 - 5, labels[i],
                   color='white', fontsize=10,
                   bbox=dict(facecolor=colors[i], alpha=0.7, pad=2))
    
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def create_yolo_labels(boxes, class_ids, img_width, img_height, save_path):
    """
    Create YOLO format label file
    
    Args:
        boxes: List of boxes in [x1, y1, x2, y2] format
        class_ids: List of class IDs for each box
        img_width: Image width
        img_height: Image height
        save_path: Path to save the label file
    """
    yolo_labels = []
    
    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = box
        
        # Convert to YOLO format (normalized center x, y, width, height)
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(yolo_labels))
    
    print(f"Labels saved to: {save_path}")


def load_yolo_labels(label_path, img_width, img_height):
    """
    Load YOLO format labels and convert to boxes
    
    Args:
        label_path: Path to label file
        img_width: Image width
        img_height: Image height
    
    Returns:
        boxes: List of boxes in [x1, y1, x2, y2] format
        class_ids: List of class IDs
    """
    boxes = []
    class_ids = []
    
    if not Path(label_path).exists():
        return boxes, class_ids
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # Convert to corner coordinates
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                boxes.append([x1, y1, x2, y2])
                class_ids.append(class_id)
    
    return boxes, class_ids


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
    
    Returns:
        IoU value (0-1)
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Calculate IoU
    if union == 0:
        return 0.0
    
    return intersection / union


def non_max_suppression(boxes, scores, threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to boxes
    
    Args:
        boxes: List of boxes [[x1, y1, x2, y2], ...]
        scores: List of confidence scores
        threshold: IoU threshold for suppression
    
    Returns:
        List of indices to keep after NMS
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort by score
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Take the first (highest score) box
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        ious = []
        for other_box in other_boxes:
            iou = calculate_iou(current_box, other_box)
            ious.append(iou)
        
        # Remove boxes with high IoU
        ious = np.array(ious)
        indices = indices[1:][ious <= threshold]
    
    return keep


# Test functions
if __name__ == "__main__":
    print("Utils module loaded successfully!")
    print("\nAvailable functions:")
    print("- draw_boxes: Draw bounding boxes on image")
    print("- plot_results: Plot results with matplotlib")
    print("- create_yolo_labels: Create YOLO format labels")
    print("- load_yolo_labels: Load YOLO format labels")
    print("- calculate_iou: Calculate IoU between boxes")
    print("- non_max_suppression: Apply NMS to boxes")