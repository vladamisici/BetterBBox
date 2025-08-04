"""
Enhanced Content Detection System
State-of-the-art document layout analysis with multi-model ensemble
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum
import cv2
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForTokenClassification,
    DetrImageProcessor, 
    DetrForObjectDetection,
    DonutProcessor,
    VisionEncoderDecoderModel
)
import timm
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DocumentType(Enum):
    ACADEMIC = "academic"
    MUSIC_SCORE = "music_score"
    FORM = "form"
    DIAGRAM = "diagram"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    metadata: Optional[Dict] = None


class EnhancedClasses:
    """Extended class definitions for various document types"""
    
    CLASSES = {
        # Document elements
        'text': 0,
        'title': 1,
        'list': 2,
        'table': 3,
        'figure': 4,
        'caption': 5,
        'header': 6,
        'footer': 7,
        'page_number': 8,
        
        # Musical elements
        'staff': 9,
        'measure': 10,
        'note': 11,
        'clef': 12,
        'time_signature': 13,
        'lyrics': 14,
        
        # Form elements
        'checkbox': 15,
        'input_field': 16,
        'signature_field': 17,
        'dropdown': 18,
        
        # Diagrams
        'flowchart': 19,
        'graph': 20,
        'equation': 21,
        
        # Special
        'barcode': 22,
        'qr_code': 23,
        'logo': 24,
        'stamp': 25
    }
    
    INVERSE_CLASSES = {v: k for k, v in CLASSES.items()}
    
    @classmethod
    def get_class_name(cls, class_id: int) -> str:
        return cls.INVERSE_CLASSES.get(class_id, "unknown")


class DocumentTypeClassifier(nn.Module):
    """Classifies document type for optimal model selection"""
    
    def __init__(self, pretrained_model='efficientnet_b4'):
        super().__init__()
        self.backbone = timm.create_model(pretrained_model, pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(DocumentType))
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def predict(self, image: np.ndarray) -> DocumentType:
        """Predict document type from image"""
        # Preprocess
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        img_tensor = transform(image=image)['image'].unsqueeze(0)
        
        with torch.no_grad():
            logits = self(img_tensor)
            pred = torch.argmax(logits, dim=1).item()
            
        return list(DocumentType)[pred]


class HybridDetectionHead(nn.Module):
    """Hybrid detection head combining YOLO and Transformer approaches"""
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        # YOLO-style head
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Classification and regression heads
        self.cls_head = nn.Conv2d(256, num_classes, 1)
        self.reg_head = nn.Conv2d(256, 4, 1)
        self.obj_head = nn.Conv2d(256, 1, 1)
        
        # Transformer decoder for complex relationships
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Object queries for DETR-style detection
        self.object_queries = nn.Parameter(torch.randn(100, 256))
        
    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}
        
        # Process through conv branch
        for i, feat in enumerate(features):
            conv_feat = self.conv_branch(feat)
            
            # Get predictions
            cls_pred = self.cls_head(conv_feat)
            reg_pred = self.reg_head(conv_feat)
            obj_pred = self.obj_head(conv_feat)
            
            outputs[f'level_{i}'] = {
                'class': cls_pred,
                'bbox': reg_pred,
                'objectness': obj_pred
            }
        
        # Transformer decoding for global context
        # Flatten features and create positional encoding
        flattened_features = []
        for feat in features:
            b, c, h, w = feat.shape
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # B, HW, C
            flattened_features.append(feat_flat)
        
        # Concatenate all feature levels
        all_features = torch.cat(flattened_features, dim=1)  # B, Total_HW, C
        
        # Decoder with object queries
        queries = self.object_queries.unsqueeze(0).expand(b, -1, -1)
        decoded = self.transformer_decoder(queries, all_features)
        
        outputs['transformer_output'] = decoded
        
        return outputs


class MusicScoreDetector:
    """Specialized detector for musical scores"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = self._load_model(model_path)
        self.music_classes = ['staff', 'measure', 'note', 'clef', 'time_signature', 'lyrics']
        
    def _load_model(self, model_path: str) -> nn.Module:
        # Load specialized music detection model
        # This could be a fine-tuned YOLO or custom architecture
        if model_path:
            return YOLO(model_path)
        else:
            # Default to a pre-configured model
            return YOLO('yolov8m.pt')
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect musical elements in the image"""
        results = self.model(image)
        
        boxes = []
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                if conf > 0.3:  # Music-specific threshold
                    boxes.append(BoundingBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        confidence=float(conf),
                        class_id=int(cls),
                        class_name=EnhancedClasses.get_class_name(int(cls))
                    ))
        
        # Post-process for music-specific rules
        boxes = self._apply_music_constraints(boxes)
        
        return boxes
    
    def _apply_music_constraints(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Apply domain-specific constraints for music notation"""
        # Example: Ensure notes are within staves
        # Group staves and validate note positions
        staves = [b for b in boxes if b.class_name == 'staff']
        notes = [b for b in boxes if b.class_name == 'note']
        
        valid_notes = []
        for note in notes:
            # Check if note is within any staff
            for staff in staves:
                if (staff.y1 <= note.y1 <= staff.y2 and 
                    staff.x1 <= note.x1 <= staff.x2):
                    valid_notes.append(note)
                    break
        
        # Return all non-note boxes plus valid notes
        other_boxes = [b for b in boxes if b.class_name != 'note']
        return other_boxes + valid_notes


class FormDetector:
    """Specialized detector for forms and structured documents"""
    
    def __init__(self):
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=len(EnhancedClasses.CLASSES)
        )
        
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect form elements using LayoutLMv3"""
        # Process image
        encoding = self.processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
        
        # Extract bounding boxes from predictions
        boxes = self._extract_boxes_from_layout(encoding, predictions)
        
        return boxes
    
    def _extract_boxes_from_layout(self, encoding, predictions):
        """Convert LayoutLM predictions to bounding boxes"""
        boxes = []
        # Implementation depends on specific LayoutLM output format
        # This is a simplified version
        return boxes


class EnhancedContentDetector:
    """Main detector class orchestrating all components"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.doc_classifier = DocumentTypeClassifier()
        self.detectors = {
            DocumentType.ACADEMIC: YOLO(self.config['models']['academic']),
            DocumentType.MUSIC_SCORE: MusicScoreDetector(self.config['models']['music']),
            DocumentType.FORM: FormDetector(),
            DocumentType.DIAGRAM: YOLO(self.config['models']['diagram'])
        }
        
        # DETR for complex layouts
        self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
        # Fusion module
        self.fusion = FusionModule()
        
    def _get_default_config(self) -> Dict:
        return {
            'models': {
                'academic': 'yolov8m.pt',
                'music': 'models/music_detector.pt',
                'diagram': 'yolov8m.pt'
            },
            'thresholds': {
                'confidence': 0.25,
                'nms': 0.45
            },
            'ensemble': {
                'use_voting': True,
                'voting_threshold': 0.3
            }
        }
    
    def detect(self, image: np.ndarray, 
               use_ensemble: bool = False,
               return_metadata: bool = True) -> List[BoundingBox]:
        """
        Main detection method
        
        Args:
            image: Input image as numpy array
            use_ensemble: Whether to use ensemble of all models
            return_metadata: Include additional metadata in results
            
        Returns:
            List of detected bounding boxes
        """
        # Step 1: Classify document type
        doc_type = self.doc_classifier.predict(image)
        
        # Step 2: Apply appropriate detector(s)
        if use_ensemble or doc_type == DocumentType.UNKNOWN:
            results = self._ensemble_detect(image)
        else:
            results = self._single_model_detect(image, doc_type)
        
        # Step 3: Apply NMS and post-processing
        results = self._post_process(results)
        
        # Step 4: Add metadata if requested
        if return_metadata:
            for box in results:
                box.metadata = {
                    'document_type': doc_type.value,
                    'detection_method': 'ensemble' if use_ensemble else 'single',
                    'timestamp': torch.cuda.Event().elapsed_time()
                }
        
        return results
    
    def _single_model_detect(self, image: np.ndarray, 
                           doc_type: DocumentType) -> List[BoundingBox]:
        """Run detection with a single model based on document type"""
        if doc_type in self.detectors:
            return self.detectors[doc_type].detect(image)
        else:
            # Fallback to general YOLO
            return self._yolo_detect(image)
    
    def _ensemble_detect(self, image: np.ndarray) -> List[BoundingBox]:
        """Run ensemble detection using multiple models"""
        all_detections = []
        
        # 1. YOLO detection
        yolo_boxes = self._yolo_detect(image)
        all_detections.extend(yolo_boxes)
        
        # 2. DETR detection
        detr_boxes = self._detr_detect(image)
        all_detections.extend(detr_boxes)
        
        # 3. Specialized detectors if confidence is low
        if self._should_use_specialized(all_detections):
            for detector in self.detectors.values():
                if hasattr(detector, 'detect'):
                    specialized_boxes = detector.detect(image)
                    all_detections.extend(specialized_boxes)
        
        # 4. Fusion
        fused_results = self.fusion.fuse(all_detections)
        
        return fused_results
    
    def _yolo_detect(self, image: np.ndarray) -> List[BoundingBox]:
        """Standard YOLO detection"""
        model = YOLO(self.config['models']['academic'])
        results = model(image)
        
        boxes = []
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                if conf > self.config['thresholds']['confidence']:
                    boxes.append(BoundingBox(
                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),
                        confidence=float(conf),
                        class_id=int(cls),
                        class_name=EnhancedClasses.get_class_name(int(cls))
                    ))
        
        return boxes
    
    def _detr_detect(self, image: np.ndarray) -> List[BoundingBox]:
        """DETR detection for complex layouts"""
        # Prepare image
        inputs = self.detr_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.detr_model(**inputs)
        
        # Process results
        target_sizes = torch.tensor([image.shape[:2]])
        results = self.detr_processor.post_process_object_detection(
            outputs, threshold=self.config['thresholds']['confidence'], 
            target_sizes=target_sizes
        )[0]
        
        boxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            boxes.append(BoundingBox(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                confidence=float(score),
                class_id=int(label),
                class_name=EnhancedClasses.get_class_name(int(label))
            ))
        
        return boxes
    
    def _should_use_specialized(self, detections: List[BoundingBox]) -> bool:
        """Determine if specialized detectors should be used"""
        if not detections:
            return True
        
        avg_confidence = np.mean([d.confidence for d in detections])
        return avg_confidence < 0.5
    
    def _post_process(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Apply NMS and other post-processing"""
        if not boxes:
            return boxes
        
        # Group by class
        class_groups = {}
        for box in boxes:
            if box.class_id not in class_groups:
                class_groups[box.class_id] = []
            class_groups[box.class_id].append(box)
        
        # Apply NMS per class
        final_boxes = []
        for class_id, class_boxes in class_groups.items():
            nms_boxes = self._nms(class_boxes, self.config['thresholds']['nms'])
            final_boxes.extend(nms_boxes)
        
        return final_boxes
    
    def _nms(self, boxes: List[BoundingBox], threshold: float) -> List[BoundingBox]:
        """Non-maximum suppression"""
        if not boxes:
            return boxes
        
        # Convert to numpy arrays
        boxes_np = np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes])
        scores = np.array([b.confidence for b in boxes])
        
        # NMS
        indices = self._nms_numpy(boxes_np, scores, threshold)
        
        return [boxes[i] for i in indices]
    
    def _nms_numpy(self, boxes: np.ndarray, scores: np.ndarray, 
                   threshold: float) -> List[int]:
        """NumPy implementation of NMS"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= threshold)[0]
            order = order[inds + 1]
        
        return keep


class FusionModule:
    """Fuses predictions from multiple models"""
    
    def __init__(self, strategy: str = 'weighted_voting'):
        self.strategy = strategy
        self.iou_threshold = 0.5
        
    def fuse(self, all_boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Fuse boxes from multiple detectors"""
        if self.strategy == 'weighted_voting':
            return self._weighted_voting_fusion(all_boxes)
        elif self.strategy == 'nms':
            return self._nms_fusion(all_boxes)
        else:
            return all_boxes
    
    def _weighted_voting_fusion(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Weighted voting based on confidence and overlap"""
        if not boxes:
            return boxes
        
        # Group overlapping boxes
        groups = self._group_overlapping_boxes(boxes)
        
        # Fuse each group
        fused_boxes = []
        for group in groups:
            if len(group) == 1:
                fused_boxes.append(group[0])
            else:
                fused_box = self._fuse_group(group)
                fused_boxes.append(fused_box)
        
        return fused_boxes
    
    def _group_overlapping_boxes(self, boxes: List[BoundingBox]) -> List[List[BoundingBox]]:
        """Group boxes that overlap significantly"""
        groups = []
        used = set()
        
        for i, box1 in enumerate(boxes):
            if i in used:
                continue
                
            group = [box1]
            used.add(i)
            
            for j, box2 in enumerate(boxes[i+1:], i+1):
                if j in used:
                    continue
                    
                if self._iou(box1, box2) > self.iou_threshold:
                    group.append(box2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _fuse_group(self, group: List[BoundingBox]) -> BoundingBox:
        """Fuse a group of overlapping boxes"""
        # Weighted average based on confidence
        total_conf = sum(b.confidence for b in group)
        
        x1 = sum(b.x1 * b.confidence for b in group) / total_conf
        y1 = sum(b.y1 * b.confidence for b in group) / total_conf
        x2 = sum(b.x2 * b.confidence for b in group) / total_conf
        y2 = sum(b.y2 * b.confidence for b in group) / total_conf
        
        # Most common class
        class_votes = {}
        for b in group:
            class_votes[b.class_id] = class_votes.get(b.class_id, 0) + b.confidence
        
        best_class = max(class_votes, key=class_votes.get)
        
        return BoundingBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=total_conf / len(group),
            class_id=best_class,
            class_name=EnhancedClasses.get_class_name(best_class)
        )
    
    def _iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate IoU between two boxes"""
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
    
    def _nms_fusion(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Simple NMS-based fusion"""
        # Implementation similar to _post_process NMS
        pass


# Utility functions
def visualize_results(image: np.ndarray, boxes: List[BoundingBox], 
                     save_path: Optional[str] = None) -> np.ndarray:
    """Visualize detection results"""
    img = image.copy()
    
    # Define colors for different classes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(EnhancedClasses.CLASSES)))
    colors = (colors[:, :3] * 255).astype(int)
    
    for box in boxes:
        color = colors[box.class_id % len(colors)]
        color = tuple(map(int, color))
        
        # Draw box
        cv2.rectangle(img, 
                     (int(box.x1), int(box.y1)), 
                     (int(box.x2), int(box.y2)), 
                     color, 2)
        
        # Draw label
        label = f"{box.class_name}: {box.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Background for label
        cv2.rectangle(img,
                     (int(box.x1), int(box.y1) - label_size[1] - 4),
                     (int(box.x1) + label_size[0], int(box.y1)),
                     color, -1)
        
        # Text
        cv2.putText(img, label,
                   (int(box.x1), int(box.y1) - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if save_path:
        cv2.imwrite(save_path, img)
    
    return img


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Initialize detector
    detector = EnhancedContentDetector()
    
    # Load image
    image_path = "path/to/your/document.jpg"
    image = cv2.imread(image_path)
    
    # Detect
    boxes = detector.detect(image, use_ensemble=True)
    
    # Visualize
    result_img = visualize_results(image, boxes)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Detected {len(boxes)} elements")
    plt.show()
    
    # Print results
    for box in boxes:
        print(f"{box.class_name}: ({box.x1:.1f}, {box.y1:.1f}, {box.x2:.1f}, {box.y2:.1f}) - {box.confidence:.3f}")