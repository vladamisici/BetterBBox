"""
Enhanced Inference Script for Document Detection
Supports single images, batch processing, and various output formats
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import cv2
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enhanced_content_detector import EnhancedContentDetector, BoundingBox, visualize_results


class InferenceEngine:
    """Enhanced inference engine with multiple output formats and optimizations"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: Optional[str] = None,
                 device: str = 'cuda',
                 use_tta: bool = False):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                import yaml
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # Initialize detector
        self.detector = EnhancedContentDetector(self.config)
        
        # Load model weights
        if model_path:
            self._load_model_weights(model_path)
        
        # Test-time augmentation settings
        self.use_tta = use_tta
        self.tta_transforms = self._get_tta_transforms()
        
        # Performance tracking
        self.inference_times = []
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'thresholds': {
                'confidence': 0.3,
                'nms': 0.5
            },
            'inference': {
                'max_detections': 500,
                'use_tta': False
            }
        }
    
    def _load_model_weights(self, model_path: str):
        """Load model weights"""
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.detector.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.detector.model.load_state_dict(checkpoint)
        
        self.detector.model.eval()
        print("Model loaded successfully!")
    
    def _get_tta_transforms(self) -> List:
        """Get test-time augmentation transforms"""
        transforms = []
        
        if self.config.get('inference', {}).get('tta_scales'):
            for scale in self.config['inference']['tta_scales']:
                transforms.append(lambda img: cv2.resize(
                    img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
                ))
        
        if 'horizontal' in self.config.get('inference', {}).get('tta_flips', []):
            transforms.append(lambda img: cv2.flip(img, 1))
        
        return transforms
    
    def process_image(self, 
                     image: Union[str, np.ndarray, Image.Image],
                     return_visualization: bool = False) -> Dict:
        """
        Process a single image
        
        Args:
            image: Image path, numpy array, or PIL Image
            return_visualization: Whether to return visualization
            
        Returns:
            Dictionary with detection results
        """
        # Load image if path
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Time inference
        start_time = time.time()
        
        # Run detection
        if self.use_tta:
            boxes = self._detect_with_tta(image)
        else:
            boxes = self.detector.detect(image, use_ensemble=True)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Prepare results
        results = {
            'boxes': self._boxes_to_dict(boxes),
            'num_detections': len(boxes),
            'inference_time': inference_time,
            'image_shape': image.shape[:2]
        }
        
        # Add visualization if requested
        if return_visualization:
            viz_image = visualize_results(image, boxes)
            results['visualization'] = viz_image
        
        return results
    
    def _detect_with_tta(self, image: np.ndarray) -> List[BoundingBox]:
        """Apply test-time augmentation"""
        all_boxes = []
        
        # Original image
        boxes = self.detector.detect(image)
        all_boxes.extend(boxes)
        
        # Apply TTA transforms
        for transform in self.tta_transforms:
            aug_image = transform(image)
            aug_boxes = self.detector.detect(aug_image)
            
            # Transform boxes back to original coordinates
            # (Implementation depends on specific transform)
            all_boxes.extend(aug_boxes)
        
        # Merge predictions
        from enhanced_content_detector import FusionModule
        fusion = FusionModule()
        merged_boxes = fusion.fuse(all_boxes)
        
        return merged_boxes
    
    def _boxes_to_dict(self, boxes: List[BoundingBox]) -> List[Dict]:
        """Convert BoundingBox objects to dictionaries"""
        return [
            {
                'x1': box.x1,
                'y1': box.y1,
                'x2': box.x2,
                'y2': box.y2,
                'confidence': box.confidence,
                'class_id': box.class_id,
                'class_name': box.class_name,
                'metadata': box.metadata
            }
            for box in boxes
        ]
    
    def process_batch(self, 
                     image_paths: List[str],
                     batch_size: int = 8,
                     show_progress: bool = True) -> List[Dict]:
        """Process multiple images in batches"""
        results = []
        
        # Create batches
        batches = [image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]
        
        # Process batches
        iterator = tqdm(batches, desc='Processing batches') if show_progress else batches
        
        for batch in iterator:
            batch_results = []
            
            for image_path in batch:
                try:
                    result = self.process_image(image_path)
                    result['image_path'] = image_path
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    batch_results.append({
                        'image_path': image_path,
                        'error': str(e)
                    })
            
            results.extend(batch_results)
        
        return results
    
    def process_directory(self, 
                         directory: str,
                         extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tif'],
                         recursive: bool = True) -> List[Dict]:
        """Process all images in a directory"""
        directory = Path(directory)
        
        # Find all images
        image_paths = []
        for ext in extensions:
            if recursive:
                image_paths.extend(directory.rglob(f'*{ext}'))
            else:
                image_paths.extend(directory.glob(f'*{ext}'))
        
        print(f"Found {len(image_paths)} images to process")
        
        # Process images
        results = self.process_batch([str(p) for p in image_paths])
        
        return results
    
    def export_results(self, 
                      results: List[Dict],
                      output_path: str,
                      format: str = 'json'):
        """Export results in various formats"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            self._export_json(results, output_path)
        elif format == 'csv':
            self._export_csv(results, output_path)
        elif format == 'coco':
            self._export_coco(results, output_path)
        elif format == 'yolo':
            self._export_yolo(results, output_path)
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def _export_json(self, results: List[Dict], output_path: Path):
        """Export results as JSON"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    def _export_csv(self, results: List[Dict], output_path: Path):
        """Export results as CSV"""
        rows = []
        
        for result in results:
            if 'error' in result:
                continue
                
            image_path = result.get('image_path', '')
            
            for box in result['boxes']:
                rows.append({
                    'image_path': image_path,
                    'x1': box['x1'],
                    'y1': box['y1'],
                    'x2': box['x2'],
                    'y2': box['y2'],
                    'width': box['x2'] - box['x1'],
                    'height': box['y2'] - box['y1'],
                    'confidence': box['confidence'],
                    'class_id': box['class_id'],
                    'class_name': box['class_name']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    def _export_coco(self, results: List[Dict], output_path: Path):
        """Export results in COCO format"""
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        from enhanced_content_detector import EnhancedClasses
        for class_name, class_id in EnhancedClasses.CLASSES.items():
            coco_data['categories'].append({
                'id': class_id,
                'name': class_name
            })
        
        # Process results
        ann_id = 1
        for img_id, result in enumerate(results):
            if 'error' in result:
                continue
            
            # Add image
            coco_data['images'].append({
                'id': img_id,
                'file_name': result.get('image_path', ''),
                'width': result['image_shape'][1],
                'height': result['image_shape'][0]
            })
            
            # Add annotations
            for box in result['boxes']:
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                coco_data['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': box['class_id'],
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # COCO format
                    'area': (x2 - x1) * (y2 - y1),
                    'score': box['confidence'],
                    'iscrowd': 0
                })
                ann_id += 1
        
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        print(f"Results saved in COCO format to {output_path}")
    
    def _export_yolo(self, results: List[Dict], output_dir: Path):
        """Export results in YOLO format"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            if 'error' in result:
                continue
            
            # Get image filename
            image_path = Path(result.get('image_path', ''))
            label_path = output_dir / f"{image_path.stem}.txt"
            
            # Write YOLO format annotations
            with open(label_path, 'w') as f:
                for box in result['boxes']:
                    # Convert to YOLO format (normalized center x, y, width, height)
                    img_h, img_w = result['image_shape']
                    x_center = (box['x1'] + box['x2']) / 2 / img_w
                    y_center = (box['y1'] + box['y2']) / 2 / img_h
                    width = (box['x2'] - box['x1']) / img_w
                    height = (box['y2'] - box['y1']) / img_h
                    
                    f.write(f"{box['class_id']} {x_center:.6f} {y_center:.6f} "
                           f"{width:.6f} {height:.6f}\n")
        
        print(f"Results saved in YOLO format to {output_dir}")
    
    def visualize_results(self, 
                         image_path: str,
                         results: Dict,
                         save_path: Optional[str] = None,
                         show: bool = True):
        """Visualize detection results with matplotlib"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Define colors for classes
        from enhanced_content_detector import EnhancedClasses
        colors = plt.cm.rainbow(np.linspace(0, 1, len(EnhancedClasses.CLASSES)))
        
        # Draw boxes
        for box in results['boxes']:
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            width = x2 - x1
            height = y2 - y1
            
            # Get color for class
            color = colors[box['class_id'] % len(colors)]
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{box['class_name']}: {box['confidence']:.2f}"
            ax.text(
                x1, y1 - 5, label,
                color='white', fontsize=10,
                bbox=dict(facecolor=color, alpha=0.7, pad=2)
            )
        
        ax.set_title(f"Detected {len(results['boxes'])} objects")
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'mean_inference_time': np.mean(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'total_images_processed': len(self.inference_times),
            'fps': 1.0 / np.mean(self.inference_times) if self.inference_times else 0
        }


def main():
    parser = argparse.ArgumentParser(description='Enhanced document detection inference')
    
    # Input/Output arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Input image, directory, or list file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--format', type=str, default='json',
                       choices=['json', 'csv', 'coco', 'yolo'],
                       help='Output format')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    # Inference arguments
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.5,
                       help='NMS threshold')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--use-tta', action='store_true',
                       help='Use test-time augmentation')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualizations')
    parser.add_argument('--show', action='store_true',
                       help='Show visualizations')
    
    # Other arguments
    parser.add_argument('--recursive', action='store_true',
                       help='Process directories recursively')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.tif'],
                       help='Image extensions to process')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = InferenceEngine(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        use_tta=args.use_tta
    )
    
    # Update thresholds if provided
    if args.confidence:
        engine.config['thresholds']['confidence'] = args.confidence
    if args.nms:
        engine.config['thresholds']['nms'] = args.nms
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image or list file
        if input_path.suffix.lower() in args.extensions:
            # Single image
            print(f"Processing single image: {input_path}")
            results = [engine.process_image(str(input_path), return_visualization=args.visualize)]
            results[0]['image_path'] = str(input_path)
        else:
            # List file
            print(f"Processing list file: {input_path}")
            with open(input_path, 'r') as f:
                image_paths = [line.strip() for line in f if line.strip()]
            results = engine.process_batch(image_paths, batch_size=args.batch_size)
    
    elif input_path.is_dir():
        # Directory
        print(f"Processing directory: {input_path}")
        results = engine.process_directory(
            str(input_path),
            extensions=args.extensions,
            recursive=args.recursive
        )
    
    else:
        raise ValueError(f"Invalid input: {input_path}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export results
    output_file = output_dir / f"detections.{args.format}"
    if args.format == 'yolo':
        engine.export_results(results, output_dir / 'labels', format=args.format)
    else:
        engine.export_results(results, output_file, format=args.format)
    
    # Save visualizations if requested
    if args.visualize:
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        for result in results:
            if 'error' in result or 'image_path' not in result:
                continue
            
            image_path = Path(result['image_path'])
            save_path = viz_dir / f"{image_path.stem}_detected{image_path.suffix}"
            
            engine.visualize_results(
                str(image_path),
                result,
                save_path=str(save_path),
                show=args.show
            )
    
    # Print performance statistics
    stats = engine.get_performance_stats()
    if stats:
        print("\nPerformance Statistics:")
        print(f"  Mean inference time: {stats['mean_inference_time']:.3f}s")
        print(f"  FPS: {stats['fps']:.1f}")
        print(f"  Total images processed: {stats['total_images_processed']}")
    
    # Summary
    successful = sum(1 for r in results if 'error' not in r)
    failed = len(results) - successful
    total_detections = sum(r['num_detections'] for r in results if 'error' not in r)
    
    print("\nSummary:")
    print(f"  Successfully processed: {successful} images")
    print(f"  Failed: {failed} images")
    print(f"  Total detections: {total_detections}")
    print(f"  Results saved to: {output_dir}")


if __name__ == '__main__':
    main()