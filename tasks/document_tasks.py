"""
Document Processing Tasks
Main tasks for document upload, processing, and management
"""

import os
import uuid
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import io

from celery import group, chain, chord
from celery_app import app, DocumentTask, get_db_session
from document_cleaner import DocumentCleaner, CleaningConfig, estimate_document_quality
from storage import MinIOStorage
from models import Document, ProcessingJob, DetectionResult
from utils import create_thumbnail, extract_document_metadata

logger = logging.getLogger(__name__)


@app.task(bind=True, base=DocumentTask, name="tasks.document_tasks.process_document")
def process_document(self, document_id: str, options: Dict[str, Any] = None) -> Dict:
    """
    Main document processing pipeline
    
    Args:
        document_id: UUID of the document to process
        options: Processing options (cleaning, detection, storage settings)
    
    Returns:
        Processing results dictionary
    """
    options = options or {}
    
    try:
        # Update job status
        with get_db_session() as session:
            job = ProcessingJob(
                document_id=document_id,
                job_type="full_processing",
                stage="cleaning",
                celery_task_id=self.request.id,
                status="processing"
            )
            session.add(job)
            session.commit()
            job_id = job.id
        
        # Create processing workflow
        workflow = chain(
            load_document.si(document_id),
            clean_document.si(options.get("cleaning", {})),
            detect_objects.si(options.get("detection", {})),
            store_results.si(document_id),
            finalize_processing.si(document_id, job_id)
        )
        
        # Execute workflow
        result = workflow.apply_async()
        
        return {
            "success": True,
            "job_id": str(job_id),
            "workflow_id": result.id,
            "message": "Document processing started"
        }
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        
        # Update job status
        with get_db_session() as session:
            job = session.query(ProcessingJob).filter_by(
                celery_task_id=self.request.id
            ).first()
            if job:
                job.status = "failed"
                job.error_message = str(e)
                session.commit()
        
        raise


@app.task(bind=True, base=DocumentTask, name="tasks.document_tasks.load_document")
def load_document(self, document_id: str) -> Dict:
    """Load document from storage"""
    try:
        with get_db_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Load from storage
            storage = MinIOStorage()
            file_data = storage.get_object("documents", document.storage_path)
            
            # Convert to numpy array
            nparr = np.frombuffer(file_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            # Extract metadata
            metadata = extract_document_metadata(image)
            
            # Update document record
            document.metadata = metadata
            session.commit()
            
            # Cache the image data
            cache_key = f"document:image:{document_id}"
            self.redis_client.setex(
                cache_key,
                3600,  # 1 hour
                file_data.hex()
            )
            
            return {
                "document_id": document_id,
                "image_shape": image.shape,
                "metadata": metadata,
                "cached": True
            }
            
    except Exception as e:
        logger.error(f"Failed to load document: {e}")
        raise


@app.task(bind=True, base=DocumentTask, name="tasks.document_tasks.clean_document")
def clean_document(self, load_result: Dict, cleaning_options: Dict = None) -> Dict:
    """Apply document cleaning algorithms"""
    try:
        document_id = load_result["document_id"]
        
        # Get image from cache
        cache_key = f"document:image:{document_id}"
        cached_data = self.redis_client.get(cache_key)
        
        if not cached_data:
            raise ValueError("Document image not found in cache")
        
        # Decode image
        file_data = bytes.fromhex(cached_data)
        nparr = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Configure cleaning
        config = CleaningConfig(**cleaning_options) if cleaning_options else CleaningConfig()
        cleaner = DocumentCleaner(config)
        
        # Estimate quality before cleaning
        quality_before = estimate_document_quality(image)
        
        # Clean document
        cleaning_result = cleaner.clean_document(image, return_intermediate=True)
        cleaned_image = cleaning_result["cleaned"]
        
        # Estimate quality after cleaning
        quality_after = estimate_document_quality(cleaned_image)
        
        # Convert cleaned image to bytes
        _, buffer = cv2.imencode('.png', cleaned_image)
        cleaned_data = buffer.tobytes()
        
        # Cache cleaned image
        cleaned_cache_key = f"document:cleaned:{document_id}"
        self.redis_client.setex(
            cleaned_cache_key,
            3600,
            cleaned_data.hex()
        )
        
        # Store cleaning metrics
        with get_db_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if document:
                document.quality_metrics = {
                    "before_cleaning": quality_before,
                    "after_cleaning": quality_after,
                    "cleaning_steps": list(cleaning_result["steps"].keys()),
                    "improvement": {
                        k: quality_after[k] - quality_before[k]
                        for k in quality_before.keys()
                    }
                }
                session.commit()
        
        return {
            "document_id": document_id,
            "quality_before": quality_before,
            "quality_after": quality_after,
            "cleaning_applied": list(cleaning_result["steps"].keys()),
            "cached": True
        }
        
    except Exception as e:
        logger.error(f"Document cleaning failed: {e}")
        raise


@app.task(bind=True, base=DocumentTask, name="tasks.document_tasks.batch_process")
def batch_process_documents(self, document_ids: List[str], options: Dict = None) -> Dict:
    """Process multiple documents in batch"""
    try:
        # Create a group of parallel tasks
        job_group = group(
            process_document.s(doc_id, options) for doc_id in document_ids
        )
        
        # Execute all tasks in parallel
        result = job_group.apply_async()
        
        return {
            "success": True,
            "batch_size": len(document_ids),
            "group_id": result.id,
            "message": f"Batch processing started for {len(document_ids)} documents"
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise


@app.task(bind=True, base=DocumentTask, name="tasks.document_tasks.create_document_thumbnail")
def create_document_thumbnail(self, document_id: str, size: tuple = (300, 300)) -> Dict:
    """Create thumbnail for document"""
    try:
        # Get cleaned image from cache
        cache_key = f"document:cleaned:{document_id}"
        cached_data = self.redis_client.get(cache_key)
        
        if not cached_data:
            # Fallback to original
            cache_key = f"document:image:{document_id}"
            cached_data = self.redis_client.get(cache_key)
        
        if not cached_data:
            raise ValueError("Document image not found")
        
        # Decode image
        file_data = bytes.fromhex(cached_data)
        nparr = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Create thumbnail
        thumbnail = create_thumbnail(image, size)
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
        thumbnail_data = buffer.tobytes()
        
        # Store in MinIO
        storage = MinIOStorage()
        thumbnail_path = f"thumbnails/{document_id}.jpg"
        storage.put_object(
            "documents",
            thumbnail_path,
            io.BytesIO(thumbnail_data),
            len(thumbnail_data),
            content_type="image/jpeg"
        )
        
        # Update database
        with get_db_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if document:
                document.thumbnail_path = thumbnail_path
                session.commit()
        
        return {
            "document_id": document_id,
            "thumbnail_path": thumbnail_path,
            "thumbnail_size": size
        }
        
    except Exception as e:
        logger.error(f"Thumbnail creation failed: {e}")
        raise


@app.task(bind=True, base=DocumentTask, name="tasks.document_tasks.extract_document_pages")
def extract_document_pages(self, document_id: str) -> Dict:
    """Extract individual pages from multi-page documents"""
    try:
        with get_db_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # This is a placeholder for PDF/multi-page TIFF handling
            # For now, we'll treat single images as single-page documents
            
            # Load document
            storage = MinIOStorage()
            file_data = storage.get_object("documents", document.storage_path)
            
            # For PDFs, we would use PyPDF2 or pdf2image
            # For multi-page TIFFs, we would use PIL
            
            # Update page count
            document.page_count = 1  # Default for single images
            session.commit()
            
            # Create task for each page
            page_tasks = []
            for page_num in range(document.page_count):
                task = process_document_page.s(document_id, page_num)
                page_tasks.append(task)
            
            if page_tasks:
                workflow = group(page_tasks)
                workflow.apply_async()
            
            return {
                "document_id": document_id,
                "page_count": document.page_count,
                "pages_queued": len(page_tasks)
            }
            
    except Exception as e:
        logger.error(f"Page extraction failed: {e}")
        raise


@app.task(bind=True, base=DocumentTask, name="tasks.document_tasks.process_document_page")
def process_document_page(self, document_id: str, page_number: int) -> Dict:
    """Process a single page of a document"""
    # Placeholder for page-specific processing
    return {
        "document_id": document_id,
        "page_number": page_number,
        "processed": True
    }


@app.task(bind=True, base=DocumentTask, name="tasks.document_tasks.merge_detection_results")
def merge_detection_results(self, results: List[Dict]) -> Dict:
    """Merge detection results from multiple models or pages"""
    try:
        merged_detections = []
        total_confidence = 0
        detection_count = 0
        
        for result in results:
            if "detections" in result:
                merged_detections.extend(result["detections"])
                detection_count += len(result["detections"])
                
                for detection in result["detections"]:
                    total_confidence += detection.get("confidence", 0)
        
        avg_confidence = total_confidence / detection_count if detection_count > 0 else 0
        
        return {
            "merged_detections": merged_detections,
            "total_detections": detection_count,
            "average_confidence": avg_confidence,
            "source_count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Failed to merge detection results: {e}")
        raise


@app.task(bind=True, base=DocumentTask, name="tasks.document_tasks.validate_document")
def validate_document(self, file_path: str, file_size: int, mime_type: str) -> Dict:
    """Validate document before processing"""
    try:
        # Size validation
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File too large: {file_size} bytes (max: {MAX_FILE_SIZE})")
        
        # MIME type validation
        ALLOWED_TYPES = {
            "image/jpeg", "image/png", "image/tiff", "image/bmp",
            "application/pdf", "image/webp"
        }
        if mime_type not in ALLOWED_TYPES:
            raise ValueError(f"Unsupported file type: {mime_type}")
        
        # Additional validation can be added here
        # - Virus scanning
        # - Content validation
        # - Format-specific validation
        
        return {
            "valid": True,
            "file_size": file_size,
            "mime_type": mime_type
        }
        
    except Exception as e:
        logger.error(f"Document validation failed: {e}")
        return {
            "valid": False,
            "error": str(e)
        }


@app.task(bind=True, base=DocumentTask, name="tasks.document_tasks.finalize_processing")
def finalize_processing(self, store_result: Dict, document_id: str, job_id: str) -> Dict:
    """Finalize document processing and update status"""
    try:
        with get_db_session() as session:
            # Update document status
            document = session.query(Document).filter_by(id=document_id).first()
            if document:
                document.status = "completed"
                document.processed_at = datetime.utcnow()
                document.processing_time_ms = int(
                    (datetime.utcnow() - document.created_at).total_seconds() * 1000
                )
            
            # Update job status
            job = session.query(ProcessingJob).filter_by(id=job_id).first()
            if job:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                job.stage = "completed"
            
            session.commit()
        
        # Clear cache
        cache_keys = [
            f"document:image:{document_id}",
            f"document:cleaned:{document_id}"
        ]
        for key in cache_keys:
            self.redis_client.delete(key)
        
        # Send notification
        from tasks.notification_tasks import send_processing_complete
        send_processing_complete.delay(document_id)
        
        return {
            "document_id": document_id,
            "job_id": job_id,
            "status": "completed",
            "message": "Document processing completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to finalize processing: {e}")
        raise


# Helper task imports
from tasks.detection_tasks import detect_objects
from tasks.storage_tasks import store_results