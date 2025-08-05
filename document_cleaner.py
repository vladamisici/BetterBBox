"""
Advanced Document Cleaning Module
State-of-the-art document cleaning algorithms for poorly scanned documents
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage
from scipy.ndimage import morphology
from skimage import filters, morphology as sk_morphology
from skimage.transform import hough_line, hough_line_peaks, rotate
from PIL import Image, ImageEnhance
import pytesseract
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for document cleaning pipeline"""
    enable_deskew: bool = True
    enable_denoise: bool = True
    enable_artifact_removal: bool = True
    enable_contrast: bool = True
    enable_border_removal: bool = True
    enable_orientation: bool = True
    
    # Deskewing parameters
    deskew_angle_threshold: float = 0.5  # Minimum angle to correct
    deskew_method: str = "hough"  # "hough" or "deep_learning"
    
    # Denoising parameters
    denoise_kernel_size: int = 3
    noise_threshold: float = 30
    
    # Contrast parameters
    contrast_clipLimit: float = 2.0
    contrast_tileGridSize: Tuple[int, int] = (8, 8)
    
    # Border detection parameters
    border_threshold: float = 0.1
    min_border_size: int = 10
    
    # Performance
    batch_size: int = 4
    use_gpu: bool = torch.cuda.is_available()


class DeskewNet(nn.Module):
    """Deep learning model for document deskewing"""
    
    def __init__(self):
        super().__init__()
        # Lightweight CNN for angle prediction
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Output: rotation angle
        )
        
    def forward(self, x):
        features = self.features(x).view(x.size(0), -1)
        angle = self.regressor(features)
        return angle * 45  # Scale to [-45, 45] degrees


class DocumentCleaner:
    """Advanced document cleaning pipeline"""
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize deep learning models if needed
        if self.config.deskew_method == "deep_learning":
            self.deskew_model = DeskewNet()
            if self.config.use_gpu:
                self.deskew_model = self.deskew_model.cuda()
            self.deskew_model.eval()
    
    def clean_document(self, image: np.ndarray, 
                      return_intermediate: bool = False) -> Dict:
        """
        Apply full cleaning pipeline to document image
        
        Args:
            image: Input document image
            return_intermediate: Return intermediate processing results
            
        Returns:
            Dictionary containing cleaned image and metadata
        """
        results = {
            'original': image.copy(),
            'steps': {}
        }
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Orientation correction
        if self.config.enable_orientation:
            gray, orientation_angle = self._correct_orientation(gray)
            results['steps']['orientation'] = {
                'image': gray.copy() if return_intermediate else None,
                'angle': orientation_angle
            }
        
        # Step 2: Deskewing
        if self.config.enable_deskew:
            gray, skew_angle = self._deskew_image(gray)
            results['steps']['deskew'] = {
                'image': gray.copy() if return_intermediate else None,
                'angle': skew_angle
            }
        
        # Step 3: Border removal
        if self.config.enable_border_removal:
            gray, borders = self._remove_borders(gray)
            results['steps']['border_removal'] = {
                'image': gray.copy() if return_intermediate else None,
                'borders': borders
            }
        
        # Step 4: Noise reduction
        if self.config.enable_denoise:
            gray = self._denoise_image(gray)
            results['steps']['denoise'] = {
                'image': gray.copy() if return_intermediate else None
            }
        
        # Step 5: Artifact removal
        if self.config.enable_artifact_removal:
            gray = self._remove_artifacts(gray)
            results['steps']['artifact_removal'] = {
                'image': gray.copy() if return_intermediate else None
            }
        
        # Step 6: Contrast enhancement
        if self.config.enable_contrast:
            gray = self._enhance_contrast(gray)
            results['steps']['contrast'] = {
                'image': gray.copy() if return_intermediate else None
            }
        
        results['cleaned'] = gray
        results['processing_complete'] = True
        
        return results
    
    def clean_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """Process multiple images in batch for efficiency"""
        futures = []
        
        for img in images:
            future = self.executor.submit(self.clean_document, img)
            futures.append(future)
        
        results = []
        for future in futures:
            results.append(future.result())
        
        return results
    
    def _correct_orientation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct page orientation (0, 90, 180, 270 degrees)"""
        angles = [0, 90, 180, 270]
        best_angle = 0
        best_score = -float('inf')
        
        for angle in angles:
            rotated = ndimage.rotate(image, angle, reshape=True)
            
            # Use text detection confidence as orientation score
            score = self._calculate_orientation_score(rotated)
            
            if score > best_score:
                best_score = score
                best_angle = angle
        
        if best_angle != 0:
            image = ndimage.rotate(image, best_angle, reshape=True)
            logger.info(f"Corrected orientation by {best_angle} degrees")
        
        return image, best_angle
    
    def _calculate_orientation_score(self, image: np.ndarray) -> float:
        """Calculate orientation score based on text-like features"""
        # Simplified version - in production, use OCR confidence or ML model
        edges = cv2.Canny(image, 50, 150)
        
        # Horizontal projection profile
        h_proj = np.sum(edges, axis=1)
        h_variance = np.var(h_proj)
        
        # Vertical projection profile  
        v_proj = np.sum(edges, axis=0)
        v_variance = np.var(v_proj)
        
        # Text typically has higher horizontal variance
        return h_variance / (v_variance + 1e-6)
    
    def _deskew_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskew document using Hough transform or deep learning"""
        if self.config.deskew_method == "hough":
            return self._deskew_hough(image)
        else:
            return self._deskew_deep_learning(image)
    
    def _deskew_hough(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskew using Hough line transform"""
        # Edge detection
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Hough transform
        tested_angles = np.linspace(-np.pi/6, np.pi/6, 120)  # ±30 degrees
        h, theta, d = hough_line(edges, theta=tested_angles)
        
        # Find most prominent lines
        hough_peaks_result = hough_line_peaks(h, theta, d, num_peaks=20)
        
        if len(hough_peaks_result[1]) == 0:
            return image, 0.0
        
        # Calculate median angle
        angles = hough_peaks_result[1]
        angle_degrees = np.rad2deg(np.median(angles)) - 90
        
        # Only correct if angle is significant
        if abs(angle_degrees) > self.config.deskew_angle_threshold:
            # Rotate image
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
            
            # Calculate new dimensions
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))
            
            # Adjust rotation matrix
            M[0, 2] += (new_width / 2) - center[0]
            M[1, 2] += (new_height / 2) - center[1]
            
            rotated = cv2.warpAffine(image, M, (new_width, new_height), 
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
            
            logger.info(f"Deskewed by {angle_degrees:.2f} degrees")
            return rotated, angle_degrees
        
        return image, 0.0
    
    def _deskew_deep_learning(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskew using deep learning model"""
        # Preprocess for model
        h, w = image.shape[:2]
        resized = cv2.resize(image, (256, 256))
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        if self.config.use_gpu:
            tensor = tensor.cuda()
        
        # Predict angle
        with torch.no_grad():
            angle = self.deskew_model(tensor).item()
        
        # Apply rotation if significant
        if abs(angle) > self.config.deskew_angle_threshold:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new dimensions
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_width = int((h * sin) + (w * cos))
            new_height = int((h * cos) + (w * sin))
            
            # Adjust rotation matrix
            M[0, 2] += (new_width / 2) - center[0]
            M[1, 2] += (new_height / 2) - center[1]
            
            rotated = cv2.warpAffine(image, M, (new_width, new_height),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
            
            logger.info(f"Deskewed by {angle:.2f} degrees (DL)")
            return rotated, angle
        
        return image, 0.0
    
    def _remove_borders(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detect and remove document borders"""
        h, w = image.shape[:2]
        
        # Use morphological operations to find borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (document)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_c, h_c = cv2.boundingRect(largest_contour)
            
            # Check if border is significant
            border_ratio_x = x / w
            border_ratio_y = y / h
            border_ratio_w = (w - (x + w_c)) / w
            border_ratio_h = (h - (y + h_c)) / h
            
            if (max(border_ratio_x, border_ratio_y, border_ratio_w, border_ratio_h) > 
                self.config.border_threshold):
                
                # Add small margin
                margin = 5
                x = max(0, x - margin)
                y = max(0, y - margin)
                w_c = min(w - x, w_c + 2 * margin)
                h_c = min(h - y, h_c + 2 * margin)
                
                cropped = image[y:y+h_c, x:x+w_c]
                
                borders = {
                    'top': y,
                    'bottom': h - (y + h_c),
                    'left': x,
                    'right': w - (x + w_c)
                }
                
                logger.info(f"Removed borders: {borders}")
                return cropped, borders
        
        return image, {'top': 0, 'bottom': 0, 'left': 0, 'right': 0}
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using adaptive techniques"""
        # Apply bilateral filter for edge-preserving smoothing
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Morphological opening to remove small noise
        kernel = np.ones((self.config.denoise_kernel_size, 
                         self.config.denoise_kernel_size), np.uint8)
        opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        # Adaptive thresholding for better text preservation
        adaptive = cv2.adaptiveThreshold(opened, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Combine original and processed for best results
        result = cv2.addWeighted(opened, 0.7, adaptive, 0.3, 0)
        
        return result
    
    def _remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Remove speckles, dots, and other artifacts"""
        # Connected component analysis
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = 255 - binary  # Invert for component analysis
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 
                                                                       connectivity=8)
        
        # Calculate average component size
        sizes = stats[1:, cv2.CC_STAT_AREA]
        if len(sizes) > 0:
            avg_size = np.median(sizes)
            
            # Create mask for small components (artifacts)
            mask = np.zeros_like(binary)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < avg_size * 0.1:  # Small artifacts
                    mask[labels == i] = 255
            
            # Remove artifacts
            cleaned = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        else:
            cleaned = image
        
        # Additional median filter for salt-and-pepper noise
        cleaned = cv2.medianBlur(cleaned, 3)
        
        return cleaned
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast and normalize brightness"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=self.config.contrast_clipLimit,
                               tileGridSize=self.config.contrast_tileGridSize)
        enhanced = clahe.apply(image)
        
        # Normalize brightness
        # Calculate current brightness
        mean_brightness = np.mean(enhanced)
        target_brightness = 200  # Target mean brightness for documents
        
        if mean_brightness > 0:
            brightness_factor = target_brightness / mean_brightness
            brightness_factor = np.clip(brightness_factor, 0.5, 2.0)
            
            # Apply brightness adjustment
            enhanced = cv2.convertScaleAbs(enhanced, alpha=brightness_factor, beta=0)
        
        # Gamma correction for better text visibility
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image specifically for object detection models"""
        # Clean the document
        cleaned_result = self.clean_document(image)
        cleaned = cleaned_result['cleaned']
        
        # Convert back to 3-channel for detection models
        if len(cleaned.shape) == 2:
            cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        # Additional preprocessing for detection
        # Enhance edges for better boundary detection
        edges = cv2.Canny(cleaned, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine original and edges
        result = cv2.addWeighted(cleaned, 0.8, edges_colored, 0.2, 0)
        
        return result


class BatchDocumentProcessor:
    """High-performance batch document processing"""
    
    def __init__(self, cleaner: DocumentCleaner, num_workers: int = 4):
        self.cleaner = cleaner
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
    
    def process_batch(self, images: List[np.ndarray], 
                     chunk_size: int = 10) -> List[Dict]:
        """Process large batches efficiently"""
        results = []
        
        # Process in chunks for memory efficiency
        for i in range(0, len(images), chunk_size):
            chunk = images[i:i + chunk_size]
            chunk_results = self.cleaner.clean_batch(chunk)
            results.extend(chunk_results)
        
        return results
    
    async def process_batch_async(self, images: List[np.ndarray]) -> List[Dict]:
        """Async batch processing for integration with FastAPI"""
        import asyncio
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self.executor,
            self.cleaner.clean_batch,
            images
        )
        
        return results


# Utility functions
def estimate_document_quality(image: np.ndarray) -> Dict[str, float]:
    """Estimate document scan quality metrics"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate various quality metrics
    metrics = {}
    
    # Contrast
    metrics['contrast'] = gray.std()
    
    # Noise level (using Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    metrics['sharpness'] = laplacian.var()
    
    # Skew detection
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        # Median angle indicates skew
        metrics['skew'] = abs(np.median(angles) % 90)
    else:
        metrics['skew'] = 0.0
    
    # Brightness
    metrics['brightness'] = np.mean(gray)
    
    # Overall quality score (0-100)
    quality_score = 100
    
    # Penalize for poor contrast
    if metrics['contrast'] < 30:
        quality_score -= 20
    
    # Penalize for blur
    if metrics['sharpness'] < 100:
        quality_score -= 15
    
    # Penalize for skew
    if metrics['skew'] > 2:
        quality_score -= 10
    
    # Penalize for poor brightness
    if metrics['brightness'] < 100 or metrics['brightness'] > 200:
        quality_score -= 15
    
    metrics['overall_quality'] = max(0, quality_score)
    
    return metrics


# Example usage
if __name__ == "__main__":
    # Initialize cleaner
    config = CleaningConfig(
        enable_deskew=True,
        enable_denoise=True,
        enable_artifact_removal=True,
        enable_contrast=True,
        enable_border_removal=True,
        enable_orientation=True
    )
    
    cleaner = DocumentCleaner(config)
    
    # Load test image
    image_path = "test_document.jpg"
    image = cv2.imread(image_path)
    
    # Estimate quality
    quality = estimate_document_quality(image)
    print(f"Document quality metrics: {quality}")
    
    # Clean document
    result = cleaner.clean_document(image, return_intermediate=True)
    
    # Save results
    cv2.imwrite("cleaned_document.jpg", result['cleaned'])
    
    # Save intermediate steps
    for step_name, step_data in result['steps'].items():
        if step_data.get('image') is not None:
            cv2.imwrite(f"step_{step_name}.jpg", step_data['image'])