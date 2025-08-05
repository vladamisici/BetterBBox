"""
Detection Tasks
Tasks for running document detection models
"""

import os
import logging
import time
from typing import Dict, List, Optional, Any
import numpy as np
import cv2
import torch
import json
from datetime import datetime

from celery import group
from celery_app import app, DetectionTask, get_db_session, CircuitBreaker
from enhanced_content_detector import (
    EnhancedContentDetector,
    DocumentTypeClassifier,
    BoundingBox,
    visualize_results
)
from models import Document, DetectionResult, ModelVersion
from utils import serialize_detection_results

logger = logging.getLogger(__name__)


# Model instances cache
_model_cache = {}


def get_model_instance(model_type: str = "default") -> EnhancedContentDetector:
    """Get or create model instance with caching"""
    if model_type not in _model_cache:
        with get_db_session() as session:
            model_version = session.query(ModelVersion).filter_by(
                model_type=model_type,
                is_active=True,
                is_default=True
            ).first()
            
            if not model_version:
                raise ValueError(f"No active model found for type: {model_type}")
            
            config = {
                'models': {
                    'academic': model_version.file_path,
                    'music': '/models/music_detector.pt',
                    'diagram': '/models/diagram_detector.pt'
                }
            }
            
            _model_cache[model_type] = EnhancedContentDetector(config)
    
    return _model_cache[model_type]


@app.task(bind=True, base=DetectionTask, name="tasks.detection_tasks.detect_objects")
def detect_objects(self, clean_result: Dict, detection_options: Dict = None) -> Dict:
    """
    Run object detection on cleaned document
    
    Args:
        clean_result: Result from document cleaning task
        detection_options: Detection configuration options
        
    Returns:
        Detection results dictionary
    """
    detection_options = detection_options or {}
    document_id = clean_result["document_id"]
    
    try:
        # Get cleaned image from cache
        cache_key = f"document:cleaned:{document_id}"
        cached_data = self.redis_client.get(cache_key)
        
        if not cached_data:
            raise ValueError("Cleaned document image not found in cache")
        
        # Decode image
        file_data = bytes.fromhex(cached_data)
        nparr = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get model instance
        detector = get_model_instance(detection_options.get("model_type", "default"))
        
        # Run detection
        start_time = time.time()
        detections = detector.detect(
            image,
            use_ensemble=detection_options.get("use_ensemble", True),
            return_metadata=True
        )
        processing_time = int((time.time() - start_time) * 1000)
        
        # Calculate statistics
        confidence_scores = [d.confidence for d in detections]
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Serialize detections
        serialized_detections = serialize_detection_results(detections)
        
        # Store results in database
        with get_db_session() as session:
            detection_result = DetectionResult(
                document_id=document_id,
                detection_count=len(detections),
                confidence_avg=float(avg_confidence),
                processing_time_ms=processing_time,
                model_version=detector.config.get("version", "v1.0.0"),
                detections=serialized_detections,
                metadata={
                    "detection_options": detection_options,
                    "quality_metrics": clean_result.get("quality_after", {})
                }
            )
            session.add(detection_result)
            session.commit()
            result_id = detection_result.id
        
        # Cache detection results
        detection_cache_key = f"detection:results:{document_id}"
        self.redis_client.setex(
            detection_cache_key,
            3600,
            json.dumps(serialized_detections)
        )
        
        # Generate visualization if requested
        if detection_options.get("generate_visualization", False):
            viz_task = create_detection_visualization.delay(
                document_id,
                str(result_id)
            )
            visualization_task_id = viz_task.id
        else:
            visualization_task_id = None
        
        return {
            "document_id": document_id,
            "detection_result_id": str(result_id),
            "detection_count": len(detections),
            "average_confidence": float(avg_confidence),
            "processing_time_ms": processing_time,
            "detections": serialized_detections,
            "visualization_task_id": visualization_task_id
        }
        
    except Exception as e:
        logger.error(f"Detection failed for document {document_id}: {e}")
        raise


@app.task(bind=True, base=DetectionTask, name="tasks.detection_tasks.ensemble_detect")
def ensemble_detect(self, document_id: str, models: List[str] = None) -> Dict:
    """
    Run ensemble detection using multiple models
    
    Args:
        document_id: Document UUID
        models: List of model types to use
        
    Returns:
        Merged detection results
    """
    models = models or ["yolo", "detr", "layoutlm"]
    
    try:
        # Create detection tasks for each model
        detection_tasks = []
        for model_type in models:
            task = detect_with_model.s(document_id, model_type)
            detection_tasks.append(task)
        
        # Run all detections in parallel
        job = group(detection_tasks)
        results = job.apply_async().get()
        
        # Merge results
        from enhanced_content_detector import FusionModule
        fusion = FusionModule(strategy="weighted_voting")
        
        all_detections = []
        for result in results:
            if "detections" in result:
                all_detections.extend(result["detections"])
        
        # Convert back to BoundingBox objects for fusion
        bbox_objects = []
        for det in all_detections:
            bbox = BoundingBox(
                x1=det["x1"],
                y1=det["y1"],
                x2=det["x2"],
                y2=det["y2"],
                confidence=det["confidence"],
                class_id=det["class_id"],
                class_name=det["class_name"]
            )
            bbox_objects.append(bbox)
        
        # Apply fusion
        fused_detections = fusion.fuse(bbox_objects)
        
        # Serialize fused results
        serialized_fused = serialize_detection_results(fused_detections)
        
        # Store ensemble results
        with get_db_session() as session:
            ensemble_result = DetectionResult(
                document_id=document_id,
                detection_count=len(fused_detections),
                confidence_avg=float(np.mean([d.confidence for d in fused_detections])),
                model_version="ensemble",
                detections=serialized_fused,
                metadata={
                    "models_used": models,
                    "fusion_strategy": "weighted_voting"
                }
            )
            session.add(ensemble_result)
            session.commit()
            result_id = ensemble_result.id
        
        return {
            "document_id": document_id,
            "ensemble_result_id": str(result_id),
            "models_used": models,
            "detection_count": len(fused_detections),
            "detections": serialized_fused
        }
        
    except Exception as e:
        logger.error(f"Ensemble detection failed: {e}")
        raise


@app.task(bind=True, base=DetectionTask, name="tasks.detection_tasks.detect_with_model")
def detect_with_model(self, document_id: str, model_type: str) -> Dict:
    """Run detection with a specific model type"""
    try:
        # Get image from cache
        cache_key = f"document:cleaned:{document_id}"
        cached_data = self.redis_client.get(cache_key)
        
        if not cached_data:
            cache_key = f"document:image:{document_id}"
            cached_data = self.redis_client.get(cache_key)
        
        if not cached_data:
            raise ValueError("Document image not found in cache")
        
        # Decode image
        file_data = bytes.fromhex(cached_data)
        nparr = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Model-specific detection
        if model_type == "yolo":
            detections = _detect_with_yolo(image)
        elif model_type == "detr":
            detections = _detect_with_detr(image)
        elif model_type == "layoutlm":
            detections = _detect_with_layoutlm(image)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return {
            "document_id": document_id,
            "model_type": model_type,
            "detections": serialize_detection_results(detections)
        }
        
    except Exception as e:
        logger.error(f"Detection with {model_type} failed: {e}")
        raise


def _detect_with_yolo(image: np.ndarray) -> List[BoundingBox]:
    """YOLO-specific detection"""
    from ultralytics import YOLO
    
    model = YOLO('/models/yolov8m.pt')
    results = model(image)
    
    detections = []
    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if conf > 0.3:
                detections.append(BoundingBox(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    confidence=float(conf),
                    class_id=int(cls),
                    class_name=model.names[int(cls)]
                ))
    
    return detections


def _detect_with_detr(image: np.ndarray) -> List[BoundingBox]:
    """DETR-specific detection"""
    from transformers import DetrImageProcessor, DetrForObjectDetection
    
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # Process results
    target_sizes = torch.tensor([image.shape[:2]])
    results = processor.post_process_object_detection(
        outputs, threshold=0.3, target_sizes=target_sizes
    )[0]
    
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append(BoundingBox(
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
            confidence=float(score),
            class_id=int(label),
            class_name=model.config.id2label[int(label)]
        ))
    
    return detections


def _detect_with_layoutlm(image: np.ndarray) -> List[BoundingBox]:
    """LayoutLM-specific detection for document understanding"""
    # Placeholder - would implement actual LayoutLM detection
    return []


@app.task(bind=True, base=DetectionTask, name="tasks.detection_tasks.create_detection_visualization")
def create_detection_visualization(self, document_id: str, detection_result_id: str) -> Dict:
    """Create visualization of detection results"""
    try:
        # Get image and detections
        image_cache_key = f"document:cleaned:{document_id}"
        cached_image = self.redis_client.get(image_cache_key)
        
        if not cached_image:
            image_cache_key = f"document:image:{document_id}"
            cached_image = self.redis_client.get(image_cache_key)
        
        detection_cache_key = f"detection:results:{document_id}"
        cached_detections = self.redis_client.get(detection_cache_key)
        
        if not cached_image or not cached_detections:
            raise ValueError("Required data not found in cache")
        
        # Decode image
        file_data = bytes.fromhex(cached_image)
        nparr = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Parse detections
        detections_data = json.loads(cached_detections)
        
        # Convert to BoundingBox objects
        detections = []
        for det in detections_data:
            bbox = BoundingBox(
                x1=det["x1"],
                y1=det["y1"],
                x2=det["x2"],
                y2=det["y2"],
                confidence=det["confidence"],
                class_id=det["class_id"],
                class_name=det["class_name"]
            )
            detections.append(bbox)
        
        # Create visualization
        viz_image = visualize_results(image, detections)
        
        # Save visualization
        viz_path = f"visualizations/{document_id}_{detection_result_id}.jpg"
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', viz_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        viz_data = buffer.tobytes()
        
        # Store in MinIO
        from storage import MinIOStorage
        storage = MinIOStorage()
        import io
        storage.put_object(
            "documents",
            viz_path,
            io.BytesIO(viz_data),
            len(viz_data),
            content_type="image/jpeg"
        )
        
        # Update database
        with get_db_session() as session:
            detection_result = session.query(DetectionResult).filter_by(
                id=detection_result_id
            ).first()
            if detection_result:
                metadata = detection_result.metadata or {}
                metadata["visualization_path"] = viz_path
                detection_result.metadata = metadata
                session.commit()
        
        return {
            "document_id": document_id,
            "detection_result_id": detection_result_id,
            "visualization_path": viz_path,
            "visualization_size": len(viz_data)
        }
        
    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")
        raise


@app.task(bind=True, base=DetectionTask, name="tasks.detection_tasks.classify_document_type")
def classify_document_type(self, document_id: str) -> Dict:
    """Classify document type using specialized classifier"""
    try:
        # Get image from cache
        cache_key = f"document:image:{document_id}"
        cached_data = self.redis_client.get(cache_key)
        
        if not cached_data:
            raise ValueError("Document image not found in cache")
        
        # Decode image
        file_data = bytes.fromhex(cached_data)
        nparr = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Classify document type
        classifier = DocumentTypeClassifier()
        doc_type = classifier.predict(image)
        
        # Update database
        with get_db_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if document:
                document.document_type = doc_type.value
                metadata = document.metadata or {}
                metadata["classification_confidence"] = 0.95  # Placeholder
                document.metadata = metadata
                session.commit()
        
        return {
            "document_id": document_id,
            "document_type": doc_type.value,
            "confidence": 0.95
        }
        
    except Exception as e:
        logger.error(f"Document classification failed: {e}")
        raise


# Circuit breaker for external model APIs
@CircuitBreaker(failure_threshold=3, recovery_timeout=60)
def call_external_detection_api(image_data: bytes, api_endpoint: str) -> Dict:
    """Call external detection API with circuit breaker protection"""
    import requests
    
    response = requests.post(
        api_endpoint,
        files={"image": image_data},
        timeout=30
    )
    response.raise_for_status()
    return response.json()


@app.task(bind=True, base=DetectionTask, name="tasks.detection_tasks.validate_detections")
def validate_detections(self, detection_result_id: str, validation_rules: Dict = None) -> Dict:
    """Validate detection results against business rules"""
    validation_rules = validation_rules or {}
    
    try:
        with get_db_session() as session:
            detection_result = session.query(DetectionResult).filter_by(
                id=detection_result_id
            ).first()
            
            if not detection_result:
                raise ValueError(f"Detection result {detection_result_id} not found")
            
            detections = detection_result.detections
            
            # Validation checks
            issues = []
            
            # Check minimum confidence
            min_confidence = validation_rules.get("min_confidence", 0.5)
            low_confidence = [
                d for d in detections 
                if d.get("confidence", 0) < min_confidence
            ]
            if low_confidence:
                issues.append({
                    "type": "low_confidence",
                    "count": len(low_confidence),
                    "details": f"{len(low_confidence)} detections below {min_confidence} confidence"
                })
            
            # Check for required classes
            required_classes = validation_rules.get("required_classes", [])
            detected_classes = set(d.get("class_name") for d in detections)
            missing_classes = set(required_classes) - detected_classes
            if missing_classes:
                issues.append({
                    "type": "missing_classes",
                    "classes": list(missing_classes),
                    "details": f"Required classes not detected: {missing_classes}"
                })
            
            # Check for overlapping detections
            overlap_threshold = validation_rules.get("overlap_threshold", 0.5)
            overlapping = _find_overlapping_detections(detections, overlap_threshold)
            if overlapping:
                issues.append({
                    "type": "overlapping_detections",
                    "count": len(overlapping),
                    "details": f"{len(overlapping)} overlapping detection pairs found"
                })
            
            # Update validation status
            metadata = detection_result.metadata or {}
            metadata["validation"] = {
                "validated": True,
                "issues": issues,
                "passed": len(issues) == 0,
                "validated_at": datetime.utcnow().isoformat()
            }
            detection_result.metadata = metadata
            session.commit()
        
        return {
            "detection_result_id": detection_result_id,
            "validation_passed": len(issues) == 0,
            "issues": issues,
            "issue_count": len(issues)
        }
        
    except Exception as e:
        logger.error(f"Detection validation failed: {e}")
        raise


def _find_overlapping_detections(detections: List[Dict], threshold: float) -> List[tuple]:
    """Find pairs of overlapping detections"""
    overlapping_pairs = []
    
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            det1, det2 = detections[i], detections[j]
            
            # Calculate IoU
            x1 = max(det1["x1"], det2["x1"])
            y1 = max(det1["y1"], det2["y1"])
            x2 = min(det1["x2"], det2["x2"])
            y2 = min(det1["y2"], det2["y2"])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (det1["x2"] - det1["x1"]) * (det1["y2"] - det1["y1"])
                area2 = (det2["x2"] - det2["x1"]) * (det2["y2"] - det2["y1"])
                union = area1 + area2 - intersection
                
                iou = intersection / union if union > 0 else 0
                
                if iou > threshold:
                    overlapping_pairs.append((i, j))
    
    return overlapping_pairs