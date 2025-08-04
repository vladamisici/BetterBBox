"""
Comprehensive Evaluation Script for Document Detection
Calculates mAP, per-class metrics, and generates detailed reports
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import cv2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import precision_recall_curve, average_precision_score
import wandb

from enhanced_content_detector import EnhancedContentDetector, BoundingBox, EnhancedClasses


class DocumentDetectionEvaluator:
    """Comprehensive evaluator for document detection models"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: Optional[str] = None,
                 device: str = 'cuda'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize detector
        self.detector = EnhancedContentDetector()
        self._load_model(model_path)
        
        # Evaluation settings
        self.iou_thresholds = np.arange(0.5, 1.0, 0.05)
        self.conf_thresholds = np.arange(0.1, 1.0, 0.1)
        
        # Results storage
        self.results = {
            'predictions': [],
            'ground_truth': [],
            'metrics': {},
            'per_class_metrics': {},
            'confusion_matrix': None
        }
    
    def _load_model(self, model_path: str):
        """Load model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.detector.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.detector.model.load_state_dict(checkpoint)
        
        self.detector.model.eval()
        print(f"Model loaded from {model_path}")
    
    def evaluate_dataset(self, 
                        annotations_path: str,
                        image_root: Optional[str] = None,
                        max_images: Optional[int] = None) -> Dict:
        """
        Evaluate model on a dataset
        
        Args:
            annotations_path: Path to annotations JSON file
            image_root: Root directory for images (if different from annotations)
            max_images: Maximum number of images to evaluate (for debugging)
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Loading annotations from {annotations_path}")
        
        # Load annotations
        with open(annotations_path, 'r') as f:
            dataset = json.load(f)
        
        # Limit dataset size if requested
        if max_images:
            dataset = dataset[:max_images]
        
        print(f"Evaluating on {len(dataset)} images...")
        
        # Process each image
        for idx, img_data in enumerate(tqdm(dataset, desc='Evaluating')):
            # Get image path
            img_path = img_data['image_path']
            if image_root:
                img_path = os.path.join(image_root, os.path.basename(img_path))
            
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get predictions
            predictions = self.detector.detect(image, use_ensemble=True)
            
            # Convert ground truth to BoundingBox format
            ground_truth = []
            for ann in img_data['annotations']:
                gt_box = BoundingBox(
                    x1=ann['bbox'][0],
                    y1=ann['bbox'][1],
                    x2=ann['bbox'][2],
                    y2=ann['bbox'][3],
                    confidence=1.0,
                    class_id=EnhancedClasses.CLASSES.get(ann['category'], -1),
                    class_name=ann['category']
                )
                ground_truth.append(gt_box)
            
            # Store results
            self.results['predictions'].append({
                'image_id': idx,
                'image_path': img_path,
                'predictions': predictions,
                'width': img_data['width'],
                'height': img_data['height']
            })
            
            self.results['ground_truth'].append({
                'image_id': idx,
                'image_path': img_path,
                'annotations': ground_truth,
                'width': img_data['width'],
                'height': img_data['height']
            })
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        self.results['metrics'] = metrics
        
        return metrics
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        print("Calculating evaluation metrics...")
        
        # Calculate mAP at different IoU thresholds
        map_results = self.calculate_map()
        
        # Calculate per-class metrics
        per_class_metrics = self.calculate_per_class_metrics()
        
        # Calculate confusion matrix
        confusion_matrix = self.calculate_confusion_matrix()
        
        # Calculate additional metrics
        additional_metrics = self.calculate_additional_metrics()
        
        # Combine all metrics
        metrics = {
            'mAP': map_results,
            'per_class': per_class_metrics,
            'confusion_matrix': confusion_matrix,
            'additional': additional_metrics
        }
        
        return metrics
    
    def calculate_map(self) -> Dict:
        """Calculate mean Average Precision at different IoU thresholds"""
        # Convert to COCO format for evaluation
        coco_gt = self._convert_to_coco_format(self.results['ground_truth'], is_gt=True)
        coco_dt = self._convert_to_coco_format(self.results['predictions'], is_gt=False)
        
        # Save temporary files for COCO evaluation
        gt_file = 'temp_gt.json'
        dt_file = 'temp_dt.json'
        
        with open(gt_file, 'w') as f:
            json.dump(coco_gt, f)
        
        with open(dt_file, 'w') as f:
            json.dump(coco_dt['annotations'], f)
        
        # Run COCO evaluation
        coco = COCO(gt_file)
        coco_dt_loaded = coco.loadRes(dt_file)
        coco_eval = COCOeval(coco, coco_dt_loaded, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Clean up temp files
        os.remove(gt_file)
        os.remove(dt_file)
        
        # Extract metrics
        map_results = {
            'mAP@0.5:0.95': coco_eval.stats[0],
            'mAP@0.5': coco_eval.stats[1],
            'mAP@0.75': coco_eval.stats[2],
            'mAP_small': coco_eval.stats[3],
            'mAP_medium': coco_eval.stats[4],
            'mAP_large': coco_eval.stats[5],
            'AR@1': coco_eval.stats[6],
            'AR@10': coco_eval.stats[7],
            'AR@100': coco_eval.stats[8],
            'AR_small': coco_eval.stats[9],
            'AR_medium': coco_eval.stats[10],
            'AR_large': coco_eval.stats[11]
        }
        
        return map_results
    
    def _convert_to_coco_format(self, data: List[Dict], is_gt: bool) -> Dict:
        """Convert evaluation data to COCO format"""
        coco_format = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for class_name, class_id in EnhancedClasses.CLASSES.items():
            coco_format['categories'].append({
                'id': class_id,
                'name': class_name
            })
        
        # Process data
        ann_id = 1
        for item in data:
            # Add image
            coco_format['images'].append({
                'id': item['image_id'],
                'file_name': item['image_path'],
                'width': item['width'],
                'height': item['height']
            })
            
            # Add annotations
            boxes = item['annotations'] if is_gt else item['predictions']
            
            for box in boxes:
                if box.class_id < 0:  # Skip invalid classes
                    continue
                
                x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
                
                ann = {
                    'id': ann_id,
                    'image_id': item['image_id'],
                    'category_id': box.class_id,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'area': (x2 - x1) * (y2 - y1),
                    'iscrowd': 0
                }
                
                if not is_gt:
                    ann['score'] = box.confidence
                
                coco_format['annotations'].append(ann)
                ann_id += 1
        
        if is_gt:
            return coco_format
        else:
            return {'annotations': coco_format['annotations']}
    
    def calculate_per_class_metrics(self) -> Dict:
        """Calculate precision, recall, and F1 for each class"""
        per_class_metrics = defaultdict(lambda: {
            'precision': [],
            'recall': [],
            'f1': [],
            'ap': 0,
            'support': 0
        })
        
        # Calculate metrics for each class
        for class_name, class_id in EnhancedClasses.CLASSES.items():
            # Collect predictions and ground truth for this class
            y_true = []
            y_scores = []
            
            for pred_data, gt_data in zip(self.results['predictions'], 
                                         self.results['ground_truth']):
                # Ground truth boxes for this class
                gt_boxes = [box for box in gt_data['annotations'] 
                           if box.class_id == class_id]
                
                # Predicted boxes for this class
                pred_boxes = [box for box in pred_data['predictions'] 
                             if box.class_id == class_id]
                
                # Match predictions to ground truth
                matched_gt = set()
                
                for pred_box in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx in matched_gt:
                            continue
                        
                        iou = self._calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= 0.5:  # Match found
                        y_true.append(1)
                        y_scores.append(pred_box.confidence)
                        matched_gt.add(best_gt_idx)
                    else:  # False positive
                        y_true.append(0)
                        y_scores.append(pred_box.confidence)
                
                # Add false negatives
                num_fn = len(gt_boxes) - len(matched_gt)
                y_true.extend([1] * num_fn)
                y_scores.extend([0] * num_fn)
            
            # Calculate metrics
            if len(y_true) > 0 and sum(y_true) > 0:
                # Precision-recall curve
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                
                # Average precision
                ap = average_precision_score(y_true, y_scores)
                
                # F1 scores at different thresholds
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                per_class_metrics[class_name] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'f1': f1_scores.tolist(),
                    'ap': ap,
                    'support': sum(y_true),
                    'max_f1': np.max(f1_scores)
                }
        
        return dict(per_class_metrics)
    
    def calculate_confusion_matrix(self) -> np.ndarray:
        """Calculate confusion matrix for all classes"""
        num_classes = len(EnhancedClasses.CLASSES)
        confusion_matrix = np.zeros((num_classes + 1, num_classes + 1))  # +1 for background
        
        for pred_data, gt_data in zip(self.results['predictions'], 
                                     self.results['ground_truth']):
            # Match predictions to ground truth
            matched_gt = set()
            
            for pred_box in pred_data['predictions']:
                best_iou = 0
                best_gt = None
                
                for gt_idx, gt_box in enumerate(gt_data['annotations']):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self._calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = (gt_idx, gt_box)
                
                if best_iou >= 0.5 and best_gt:  # True positive
                    confusion_matrix[best_gt[1].class_id, pred_box.class_id] += 1
                    matched_gt.add(best_gt[0])
                else:  # False positive (background)
                    confusion_matrix[num_classes, pred_box.class_id] += 1
            
            # False negatives
            for gt_idx, gt_box in enumerate(gt_data['annotations']):
                if gt_idx not in matched_gt:
                    confusion_matrix[gt_box.class_id, num_classes] += 1
        
        return confusion_matrix
    
    def calculate_additional_metrics(self) -> Dict:
        """Calculate additional evaluation metrics"""
        metrics = {
            'total_predictions': 0,
            'total_ground_truth': 0,
            'avg_confidence': 0,
            'inference_speed': [],
            'size_analysis': defaultdict(int),
            'aspect_ratio_analysis': defaultdict(int)
        }
        
        all_confidences = []
        
        for pred_data, gt_data in zip(self.results['predictions'], 
                                     self.results['ground_truth']):
            metrics['total_predictions'] += len(pred_data['predictions'])
            metrics['total_ground_truth'] += len(gt_data['annotations'])
            
            # Confidence analysis
            for pred_box in pred_data['predictions']:
                all_confidences.append(pred_box.confidence)
                
                # Size analysis
                width = pred_box.x2 - pred_box.x1
                height = pred_box.y2 - pred_box.y1
                area = width * height
                
                if area < 32 * 32:
                    metrics['size_analysis']['small'] += 1
                elif area < 96 * 96:
                    metrics['size_analysis']['medium'] += 1
                else:
                    metrics['size_analysis']['large'] += 1
                
                # Aspect ratio analysis
                aspect_ratio = width / (height + 1e-8)
                if aspect_ratio < 0.5:
                    metrics['aspect_ratio_analysis']['tall'] += 1
                elif aspect_ratio > 2.0:
                    metrics['aspect_ratio_analysis']['wide'] += 1
                else:
                    metrics['aspect_ratio_analysis']['square'] += 1
        
        if all_confidences:
            metrics['avg_confidence'] = np.mean(all_confidences)
            metrics['confidence_std'] = np.std(all_confidences)
        
        return metrics
    
    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def generate_report(self, output_dir: str):
        """Generate comprehensive evaluation report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw metrics
        metrics_file = output_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.results['metrics'], f, indent=2)
        
        # Generate visualizations
        self._plot_precision_recall_curves(output_dir)
        self._plot_confusion_matrix(output_dir)
        self._plot_class_performance(output_dir)
        self._plot_size_analysis(output_dir)
        
        # Generate text report
        self._generate_text_report(output_dir)
        
        # Generate HTML report
        self._generate_html_report(output_dir)
        
        print(f"Evaluation report saved to {output_dir}")
    
    def _plot_precision_recall_curves(self, output_dir: Path):
        """Plot precision-recall curves for each class"""
        fig, axes = plt.subplots(5, 6, figsize=(20, 15))
        axes = axes.flatten()
        
        for idx, (class_name, metrics) in enumerate(self.results['metrics']['per_class'].items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            if metrics['precision'] and metrics['recall']:
                ax.plot(metrics['recall'], metrics['precision'], 'b-')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'{class_name} (AP={metrics["ap"]:.3f})')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(class_name)
        
        # Hide unused subplots
        for idx in range(len(self.results['metrics']['per_class']), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_recall_curves.png', dpi=150)
        plt.close()
    
    def _plot_confusion_matrix(self, output_dir: Path):
        """Plot confusion matrix"""
        cm = self.results['metrics']['confusion_matrix']
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Create labels
        labels = list(EnhancedClasses.CLASSES.keys()) + ['background']
        
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title('Confusion Matrix (Normalized)', fontsize=16)
        plt.xlabel('Predicted Class', fontsize=14)
        plt.ylabel('True Class', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
    
    def _plot_class_performance(self, output_dir: Path):
        """Plot per-class performance metrics"""
        class_names = []
        ap_scores = []
        f1_scores = []
        support = []
        
        for class_name, metrics in self.results['metrics']['per_class'].items():
            class_names.append(class_name)
            ap_scores.append(metrics['ap'])
            f1_scores.append(metrics.get('max_f1', 0))
            support.append(metrics['support'])
        
        # Sort by AP score
        sorted_indices = np.argsort(ap_scores)[::-1]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # AP scores
        x = np.arange(len(class_names))
        ax1.bar(x, [ap_scores[i] for i in sorted_indices])
        ax1.set_xticks(x)
        ax1.set_xticklabels([class_names[i] for i in sorted_indices], rotation=45, ha='right')
        ax1.set_ylabel('Average Precision')
        ax1.set_title('Per-Class Average Precision')
        ax1.grid(True, alpha=0.3)
        
        # Support (number of instances)
        ax2.bar(x, [support[i] for i in sorted_indices])
        ax2.set_xticks(x)
        ax2.set_xticklabels([class_names[i] for i in sorted_indices], rotation=45, ha='right')
        ax2.set_ylabel('Number of Instances')
        ax2.set_title('Class Distribution in Test Set')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'class_performance.png', dpi=150)
        plt.close()
    
    def _plot_size_analysis(self, output_dir: Path):
        """Plot object size distribution analysis"""
        size_data = self.results['metrics']['additional']['size_analysis']
        aspect_data = self.results['metrics']['additional']['aspect_ratio_analysis']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Size distribution
        sizes = list(size_data.keys())
        counts = list(size_data.values())
        ax1.pie(counts, labels=sizes, autopct='%1.1f%%')
        ax1.set_title('Object Size Distribution')
        
        # Aspect ratio distribution
        aspects = list(aspect_data.keys())
        counts = list(aspect_data.values())
        ax2.pie(counts, labels=aspects, autopct='%1.1f%%')
        ax2.set_title('Aspect Ratio Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'size_analysis.png', dpi=150)
        plt.close()
    
    def _generate_text_report(self, output_dir: Path):
        """Generate detailed text report"""
        report_lines = [
            "=" * 80,
            "DOCUMENT DETECTION EVALUATION REPORT",
            "=" * 80,
            "",
            "Overall Metrics:",
            "-" * 40,
            f"mAP@0.5:0.95: {self.results['metrics']['mAP']['mAP@0.5:0.95']:.4f}",
            f"mAP@0.5: {self.results['metrics']['mAP']['mAP@0.5']:.4f}",
            f"mAP@0.75: {self.results['metrics']['mAP']['mAP@0.75']:.4f}",
            "",
            "Size-based Performance:",
            "-" * 40,
            f"mAP (small): {self.results['metrics']['mAP']['mAP_small']:.4f}",
            f"mAP (medium): {self.results['metrics']['mAP']['mAP_medium']:.4f}",
            f"mAP (large): {self.results['metrics']['mAP']['mAP_large']:.4f}",
            "",
            "Per-Class Performance:",
            "-" * 40,
            f"{'Class':<20} {'AP':<10} {'Max F1':<10} {'Support':<10}",
            "-" * 50
        ]
        
        # Sort classes by AP
        sorted_classes = sorted(
            self.results['metrics']['per_class'].items(),
            key=lambda x: x[1]['ap'],
            reverse=True
        )
        
        for class_name, metrics in sorted_classes:
            report_lines.append(
                f"{class_name:<20} {metrics['ap']:<10.4f} "
                f"{metrics.get('max_f1', 0):<10.4f} {metrics['support']:<10d}"
            )
        
        report_lines.extend([
            "",
            "Dataset Statistics:",
            "-" * 40,
            f"Total predictions: {self.results['metrics']['additional']['total_predictions']}",
            f"Total ground truth: {self.results['metrics']['additional']['total_ground_truth']}",
            f"Average confidence: {self.results['metrics']['additional']['avg_confidence']:.4f}",
            "",
            "=" * 80
        ])
        
        # Save report
        report_file = output_dir / 'evaluation_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
    
    def _generate_html_report(self, output_dir: Path):
        """Generate interactive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Detection Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-card {{ 
                    background-color: #f0f0f0; 
                    padding: 15px; 
                    margin: 10px; 
                    border-radius: 5px; 
                    display: inline-block;
                    min-width: 200px;
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Document Detection Evaluation Report</h1>
            
            <h2>Overall Performance</h2>
            <div>
                <div class="metric-card">
                    <div>mAP@0.5:0.95</div>
                    <div class="metric-value">{self.results['metrics']['mAP']['mAP@0.5:0.95']:.3f}</div>
                </div>
                <div class="metric-card">
                    <div>mAP@0.5</div>
                    <div class="metric-value">{self.results['metrics']['mAP']['mAP@0.5']:.3f}</div>
                </div>
                <div class="metric-card">
                    <div>mAP@0.75</div>
                    <div class="metric-value">{self.results['metrics']['mAP']['mAP@0.75']:.3f}</div>
                </div>
            </div>
            
            <h2>Visualizations</h2>
            <img src="precision_recall_curves.png" alt="Precision-Recall Curves">
            <img src="class_performance.png" alt="Class Performance">
            <img src="confusion_matrix.png" alt="Confusion Matrix">
            <img src="size_analysis.png" alt="Size Analysis">
            
            <h2>Per-Class Metrics</h2>
            <table>
                <tr>
                    <th>Class</th>
                    <th>Average Precision</th>
                    <th>Max F1-Score</th>
                    <th>Support</th>
                </tr>
        """
        
        # Add per-class metrics
        sorted_classes = sorted(
            self.results['metrics']['per_class'].items(),
            key=lambda x: x[1]['ap'],
            reverse=True
        )
        
        for class_name, metrics in sorted_classes:
            html_content += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{metrics['ap']:.4f}</td>
                    <td>{metrics.get('max_f1', 0):.4f}</td>
                    <td>{metrics['support']}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Additional Statistics</h2>
            <ul>
                <li>Total Predictions: {}</li>
                <li>Total Ground Truth: {}</li>
                <li>Average Confidence: {:.4f}</li>
            </ul>
        </body>
        </html>
        """.format(
            self.results['metrics']['additional']['total_predictions'],
            self.results['metrics']['additional']['total_ground_truth'],
            self.results['metrics']['additional']['avg_confidence']
        )
        
        # Save HTML report
        html_file = output_dir / 'evaluation_report.html'
        with open(html_file, 'w') as f:
            f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description='Evaluate document detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to evaluation dataset (JSON)')
    parser.add_argument('--image-root', type=str, default=None,
                       help='Root directory for images')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--wandb', action='store_true',
                       help='Log results to Weights & Biases')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = DocumentDetectionEvaluator(
        model_path=args.model,
        device=args.device
    )
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        annotations_path=args.dataset,
        image_root=args.image_root,
        max_images=args.max_images
    )
    
    # Generate report
    evaluator.generate_report(args.output)
    
    # Log to wandb if requested
    if args.wandb:
        wandb.init(project='document-detection-evaluation')
        wandb.log(metrics)
        
        # Log plots
        output_dir = Path(args.output)
        wandb.log({
            'precision_recall_curves': wandb.Image(str(output_dir / 'precision_recall_curves.png')),
            'confusion_matrix': wandb.Image(str(output_dir / 'confusion_matrix.png')),
            'class_performance': wandb.Image(str(output_dir / 'class_performance.png')),
            'size_analysis': wandb.Image(str(output_dir / 'size_analysis.png'))
        })
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"mAP@0.5:0.95: {metrics['mAP']['mAP@0.5:0.95']:.4f}")
    print(f"mAP@0.5: {metrics['mAP']['mAP@0.5']:.4f}")
    print(f"mAP@0.75: {metrics['mAP']['mAP@0.75']:.4f}")
    print(f"\nDetailed report saved to: {args.output}")


if __name__ == '__main__':
    main()  