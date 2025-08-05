"""
Utility Functions
Common utilities for document processing system
"""

import os
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import cv2
from PIL import Image
import json

logger = logging.getLogger(__name__)


def serialize_detection_results(detections: List['BoundingBox']) -> List[Dict]:
    """
    Serialize BoundingBox objects to JSON-compatible format
    
    Args:
        detections: List of BoundingBox objects
        
    Returns:
        List of serialized detection dictionaries
    """
    serialized = []
    for det in detections:
        serialized.append({
            "x1": float(det.x1),
            "y1": float(det.y1),
            "x2": float(det.x2),
            "y2": float(det.y2),
            "confidence": float(det.confidence),
            "class_id": int(det.class_id),
            "class_name": str(det.class_name),
            "area": float((det.x2 - det.x1) * (det.y2 - det.y1)),
            "center_x": float((det.x1 + det.x2) / 2),
            "center_y": float((det.y1 + det.y2) / 2),
            "metadata": det.metadata if hasattr(det, 'metadata') else {}
        })
    return serialized


def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (300, 300)) -> np.ndarray:
    """
    Create thumbnail from image array
    
    Args:
        image: Input image as numpy array
        size: Target thumbnail size
        
    Returns:
        Thumbnail as numpy array
    """
    height, width = image.shape[:2]
    
    # Calculate aspect ratio
    aspect = width / height
    
    # Calculate new dimensions
    if aspect > 1:
        # Width is greater
        new_width = size[0]
        new_height = int(new_width / aspect)
    else:
        # Height is greater or equal
        new_height = size[1]
        new_width = int(new_height * aspect)
    
    # Resize image
    thumbnail = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create blank image with target size
    if len(image.shape) == 3:
        blank = np.zeros((size[1], size[0], image.shape[2]), dtype=np.uint8)
    else:
        blank = np.zeros((size[1], size[0]), dtype=np.uint8)
    
    # Center thumbnail in blank image
    y_offset = (size[1] - new_height) // 2
    x_offset = (size[0] - new_width) // 2
    
    blank[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = thumbnail
    
    return blank


def extract_document_metadata(image: np.ndarray) -> Dict[str, Any]:
    """
    Extract metadata from document image
    
    Args:
        image: Document image as numpy array
        
    Returns:
        Dictionary containing document metadata
    """
    metadata = {}
    
    # Basic image properties
    if len(image.shape) == 3:
        height, width, channels = image.shape
        metadata['channels'] = int(channels)
        metadata['color_mode'] = 'RGB' if channels == 3 else 'RGBA' if channels == 4 else 'Unknown'
    else:
        height, width = image.shape
        metadata['channels'] = 1
        metadata['color_mode'] = 'Grayscale'
    
    metadata['height'] = int(height)
    metadata['width'] = int(width)
    metadata['aspect_ratio'] = float(width / height)
    metadata['pixel_count'] = int(height * width)
    
    # Image statistics
    metadata['mean_brightness'] = float(np.mean(image))
    metadata['std_brightness'] = float(np.std(image))
    
    # Estimate DPI (assuming standard paper sizes)
    estimated_dpi = estimate_dpi(width, height)
    metadata['estimated_dpi'] = estimated_dpi
    
    # Color analysis for color images
    if len(image.shape) == 3 and image.shape[2] >= 3:
        # Dominant colors
        metadata['dominant_colors'] = get_dominant_colors(image, n_colors=5)
        
        # Check if mostly grayscale
        is_grayscale = check_if_grayscale(image)
        metadata['is_grayscale'] = is_grayscale
    
    # Edge density (indicates text/content density)
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image, 50, 150)
    metadata['edge_density'] = float(np.sum(edges > 0) / (height * width))
    
    # Blur detection
    metadata['blur_score'] = calculate_blur_score(image)
    metadata['is_blurry'] = metadata['blur_score'] < 100
    
    return metadata


def estimate_dpi(width: int, height: int) -> int:
    """Estimate DPI based on common paper sizes"""
    # Common paper sizes in inches
    paper_sizes = {
        'letter': (8.5, 11),
        'a4': (8.27, 11.69),
        'legal': (8.5, 14),
        'a3': (11.69, 16.54)
    }
    
    aspect_ratio = width / height
    best_dpi = 72  # Default
    
    for size_name, (w_inch, h_inch) in paper_sizes.items():
        # Check both orientations
        for paper_w, paper_h in [(w_inch, h_inch), (h_inch, w_inch)]:
            paper_aspect = paper_w / paper_h
            
            # If aspect ratio is close (within 5%)
            if abs(aspect_ratio - paper_aspect) / paper_aspect < 0.05:
                estimated_dpi_w = width / paper_w
                estimated_dpi_h = height / paper_h
                best_dpi = int((estimated_dpi_w + estimated_dpi_h) / 2)
                break
    
    # Round to common DPI values
    common_dpis = [72, 96, 150, 200, 300, 600]
    closest_dpi = min(common_dpis, key=lambda x: abs(x - best_dpi))
    
    return closest_dpi


def get_dominant_colors(image: np.ndarray, n_colors: int = 5) -> List[Dict]:
    """Extract dominant colors from image"""
    from sklearn.cluster import KMeans
    
    # Reshape image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Sample pixels for faster processing
    sample_size = min(10000, len(pixels))
    sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
    sample_pixels = pixels[sample_indices]
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(sample_pixels)
    
    # Get color centers and percentages
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.predict(sample_pixels)
    
    color_info = []
    for i, color in enumerate(colors):
        percentage = float(np.sum(labels == i) / len(labels))
        color_info.append({
            'rgb': [int(c) for c in color],
            'hex': '#{:02x}{:02x}{:02x}'.format(*color),
            'percentage': round(percentage * 100, 2)
        })
    
    # Sort by percentage
    color_info.sort(key=lambda x: x['percentage'], reverse=True)
    
    return color_info


def check_if_grayscale(image: np.ndarray, threshold: float = 10) -> bool:
    """Check if a color image is actually grayscale"""
    if len(image.shape) != 3 or image.shape[2] < 3:
        return True
    
    # Calculate the standard deviation between color channels
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    # Sample pixels for efficiency
    sample_size = min(10000, image.shape[0] * image.shape[1])
    sample_indices = np.random.choice(image.shape[0] * image.shape[1], sample_size, replace=False)
    
    b_flat = b.flatten()[sample_indices]
    g_flat = g.flatten()[sample_indices]
    r_flat = r.flatten()[sample_indices]
    
    # Check if R, G, B values are similar
    rg_diff = np.mean(np.abs(r_flat - g_flat))
    rb_diff = np.mean(np.abs(r_flat - b_flat))
    gb_diff = np.mean(np.abs(g_flat - b_flat))
    
    avg_diff = (rg_diff + rb_diff + gb_diff) / 3
    
    return avg_diff < threshold


def calculate_blur_score(image: np.ndarray) -> float:
    """Calculate blur score using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def generate_presigned_url(client, bucket: str, key: str, expires_in: int = 3600) -> str:
    """
    Generate presigned URL for S3/MinIO object
    
    Args:
        client: S3 client instance
        bucket: Bucket name
        key: Object key
        expires_in: URL expiration time in seconds
        
    Returns:
        Presigned URL string
    """
    try:
        url = client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expires_in
        )
        return url
    except Exception as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        raise


def calculate_file_hash(file_data: bytes, algorithm: str = 'sha256') -> str:
    """Calculate hash of file data"""
    if algorithm == 'sha256':
        return hashlib.sha256(file_data).hexdigest()
    elif algorithm == 'md5':
        return hashlib.md5(file_data).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def validate_image_file(file_path: str) -> bool:
    """Validate if file is a valid image"""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False


def convert_image_format(image: np.ndarray, target_format: str = 'RGB') -> np.ndarray:
    """Convert image to target format"""
    if len(image.shape) == 2:  # Grayscale
        if target_format == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif target_format == 'BGR':
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:  # Already 3-channel
            if target_format == 'RGB':
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif target_format == 'BGR':
                return image
        elif image.shape[2] == 4:  # RGBA
            if target_format == 'RGB':
                return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif target_format == 'BGR':
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    return image


def split_image_into_patches(image: np.ndarray, patch_size: int = 512, 
                           overlap: int = 64) -> List[Dict]:
    """Split large image into overlapping patches"""
    height, width = image.shape[:2]
    patches = []
    
    stride = patch_size - overlap
    
    for y in range(0, height - overlap, stride):
        for x in range(0, width - overlap, stride):
            # Calculate patch boundaries
            y_end = min(y + patch_size, height)
            x_end = min(x + patch_size, width)
            
            # Extract patch
            patch = image[y:y_end, x:x_end]
            
            # Pad if necessary
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                if len(image.shape) == 3:
                    padded = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
                else:
                    padded = np.zeros((patch_size, patch_size), dtype=image.dtype)
                
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            
            patches.append({
                'patch': patch,
                'position': (x, y),
                'original_size': (x_end - x, y_end - y)
            })
    
    return patches


def merge_detection_patches(patches_results: List[Dict], image_size: Tuple[int, int],
                          overlap: int = 64) -> List[Dict]:
    """Merge detection results from overlapping patches"""
    merged_detections = []
    
    for patch_result in patches_results:
        position = patch_result['position']
        detections = patch_result['detections']
        
        for det in detections:
            # Adjust coordinates to global image space
            global_det = {
                'x1': det['x1'] + position[0],
                'y1': det['y1'] + position[1],
                'x2': det['x2'] + position[0],
                'y2': det['y2'] + position[1],
                'confidence': det['confidence'],
                'class_id': det['class_id'],
                'class_name': det['class_name']
            }
            merged_detections.append(global_det)
    
    # Remove duplicates from overlapping regions
    # This is a simplified version - in production, use proper NMS
    filtered_detections = []
    used = set()
    
    for i, det1 in enumerate(merged_detections):
        if i in used:
            continue
            
        # Check for overlapping detections
        overlapping = [i]
        for j, det2 in enumerate(merged_detections[i+1:], i+1):
            if j in used:
                continue
                
            iou = calculate_iou(det1, det2)
            if iou > 0.5 and det1['class_id'] == det2['class_id']:
                overlapping.append(j)
                used.add(j)
        
        # Keep detection with highest confidence
        best_idx = max(overlapping, key=lambda idx: merged_detections[idx]['confidence'])
        filtered_detections.append(merged_detections[best_idx])
    
    return filtered_detections


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """Calculate Intersection over Union for two boxes"""
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


# Time utilities
def get_time_ago(timestamp: datetime) -> str:
    """Get human-readable time ago string"""
    now = datetime.utcnow()
    diff = now - timestamp
    
    if diff.days > 365:
        return f"{diff.days // 365} year{'s' if diff.days // 365 > 1 else ''} ago"
    elif diff.days > 30:
        return f"{diff.days // 30} month{'s' if diff.days // 30 > 1 else ''} ago"
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "just now"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


# Validation utilities
def validate_detection_results(detections: List[Dict]) -> Tuple[bool, List[str]]:
    """Validate detection results format"""
    errors = []
    
    required_fields = ['x1', 'y1', 'x2', 'y2', 'confidence', 'class_id', 'class_name']
    
    for i, det in enumerate(detections):
        # Check required fields
        for field in required_fields:
            if field not in det:
                errors.append(f"Detection {i}: Missing required field '{field}'")
        
        # Validate coordinates
        if 'x1' in det and 'x2' in det:
            if det['x2'] <= det['x1']:
                errors.append(f"Detection {i}: Invalid x coordinates (x2 <= x1)")
        
        if 'y1' in det and 'y2' in det:
            if det['y2'] <= det['y1']:
                errors.append(f"Detection {i}: Invalid y coordinates (y2 <= y1)")
        
        # Validate confidence
        if 'confidence' in det:
            if not 0 <= det['confidence'] <= 1:
                errors.append(f"Detection {i}: Invalid confidence value {det['confidence']}")
    
    return len(errors) == 0, errors


# Export all utilities
__all__ = [
    'serialize_detection_results',
    'create_thumbnail',
    'extract_document_metadata',
    'estimate_dpi',
    'get_dominant_colors',
    'check_if_grayscale',
    'calculate_blur_score',
    'generate_presigned_url',
    'calculate_file_hash',
    'validate_image_file',
    'convert_image_format',
    'split_image_into_patches',
    'merge_detection_patches',
    'calculate_iou',
    'get_time_ago',
    'format_file_size',
    'validate_detection_results'
]