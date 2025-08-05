"""
Storage Tasks
Tasks for managing document storage with MinIO/S3
"""

import os
import io
import logging
import hashlib
import mimetypes
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import cv2
import numpy as np
from PIL import Image
import boto3
from botocore.exceptions import ClientError

from celery_app import app, StorageTask, get_db_session, CircuitBreaker
from models import Document, DetectionResult
from utils import generate_presigned_url

logger = logging.getLogger(__name__)


class MinIOStorage:
    """MinIO/S3 compatible storage client"""
    
    def __init__(self):
        self.endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
        
        self.client = boto3.client(
            's3',
            endpoint_url=f"{'https' if self.secure else 'http'}://{self.endpoint}",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=boto3.session.Config(signature_version='s3v4')
        )
        
        # Ensure buckets exist
        self._ensure_buckets()
    
    def _ensure_buckets(self):
        """Ensure required buckets exist"""
        required_buckets = ['documents', 'processed', 'models', 'temp']
        
        try:
            existing_buckets = [b['Name'] for b in self.client.list_buckets()['Buckets']]
            
            for bucket in required_buckets:
                if bucket not in existing_buckets:
                    self.client.create_bucket(Bucket=bucket)
                    logger.info(f"Created bucket: {bucket}")
                    
                    # Set bucket policy for public read on certain buckets
                    if bucket in ['processed']:
                        self._set_public_read_policy(bucket)
        
        except Exception as e:
            logger.error(f"Failed to ensure buckets: {e}")
    
    def _set_public_read_policy(self, bucket_name: str):
        """Set public read policy for bucket"""
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": ["s3:GetObject"],
                    "Resource": f"arn:aws:s3:::{bucket_name}/*"
                }
            ]
        }
        
        import json
        self.client.put_bucket_policy(
            Bucket=bucket_name,
            Policy=json.dumps(policy)
        )
    
    def put_object(self, bucket: str, key: str, data: io.BytesIO, 
                   length: int, content_type: str = None) -> Dict:
        """Upload object to storage"""
        try:
            metadata = {
                'upload-timestamp': datetime.utcnow().isoformat(),
                'content-length': str(length)
            }
            
            if not content_type:
                content_type = 'application/octet-stream'
            
            self.client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                Metadata=metadata
            )
            
            return {
                'bucket': bucket,
                'key': key,
                'size': length,
                'etag': self.client.head_object(Bucket=bucket, Key=key)['ETag']
            }
            
        except Exception as e:
            logger.error(f"Failed to upload object: {e}")
            raise
    
    def get_object(self, bucket: str, key: str) -> bytes:
        """Download object from storage"""
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"Failed to get object: {e}")
            raise
    
    def delete_object(self, bucket: str, key: str):
        """Delete object from storage"""
        try:
            self.client.delete_object(Bucket=bucket, Key=key)
        except Exception as e:
            logger.error(f"Failed to delete object: {e}")
            raise
    
    def list_objects(self, bucket: str, prefix: str = "", limit: int = 1000) -> List[Dict]:
        """List objects in bucket"""
        try:
            response = self.client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=limit
            )
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag']
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"Failed to list objects: {e}")
            raise
    
    def generate_presigned_url(self, bucket: str, key: str, 
                             expires_in: int = 3600) -> str:
        """Generate presigned URL for object access"""
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise


@app.task(bind=True, base=StorageTask, name="tasks.storage_tasks.store_results")
def store_results(self, detection_result: Dict, document_id: str) -> Dict:
    """
    Store detection results and processed images
    
    Args:
        detection_result: Results from detection task
        document_id: Document UUID
        
    Returns:
        Storage information
    """
    try:
        storage = MinIOStorage()
        stored_files = []
        
        # Get processed image from cache
        cache_key = f"document:cleaned:{document_id}"
        cached_data = app.Task.redis_client.get(cache_key)
        
        if cached_data:
            # Store cleaned image
            file_data = bytes.fromhex(cached_data)
            
            cleaned_key = f"cleaned/{document_id}.png"
            result = storage.put_object(
                "processed",
                cleaned_key,
                io.BytesIO(file_data),
                len(file_data),
                "image/png"
            )
            stored_files.append(result)
            
            # Update database
            with get_db_session() as session:
                document = session.query(Document).filter_by(id=document_id).first()
                if document:
                    document.processed_path = cleaned_key
                    session.commit()
        
        # Store detection visualization if exists
        viz_path = detection_result.get("visualization_path")
        if viz_path:
            stored_files.append({
                'bucket': 'processed',
                'key': viz_path,
                'type': 'visualization'
            })
        
        # Generate and store detection report
        report_data = _generate_detection_report(detection_result)
        report_key = f"reports/{document_id}_detection_report.json"
        
        import json
        report_bytes = json.dumps(report_data, indent=2).encode('utf-8')
        
        result = storage.put_object(
            "processed",
            report_key,
            io.BytesIO(report_bytes),
            len(report_bytes),
            "application/json"
        )
        stored_files.append(result)
        
        # Create document package (ZIP with all outputs)
        package_task = create_document_package.delay(document_id)
        
        return {
            "document_id": document_id,
            "stored_files": stored_files,
            "cleaned_image_path": f"processed/cleaned/{document_id}.png",
            "report_path": f"processed/{report_key}",
            "package_task_id": package_task.id
        }
        
    except Exception as e:
        logger.error(f"Failed to store results: {e}")
        raise


@app.task(bind=True, base=StorageTask, name="tasks.storage_tasks.upload_document")
def upload_document(self, file_data: bytes, filename: str, 
                   user_id: str, metadata: Dict = None) -> Dict:
    """
    Upload raw document to storage
    
    Args:
        file_data: Raw file bytes
        filename: Original filename
        user_id: User UUID
        metadata: Additional metadata
        
    Returns:
        Upload information
    """
    try:
        storage = MinIOStorage()
        
        # Generate unique storage path
        file_hash = hashlib.sha256(file_data).hexdigest()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_ext = os.path.splitext(filename)[1]
        storage_key = f"uploads/{user_id}/{timestamp}_{file_hash[:8]}{file_ext}"
        
        # Detect MIME type
        mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        
        # Upload to storage
        result = storage.put_object(
            "documents",
            storage_key,
            io.BytesIO(file_data),
            len(file_data),
            mime_type
        )
        
        # Create database record
        with get_db_session() as session:
            document = Document(
                user_id=user_id,
                original_filename=filename,
                file_size=len(file_data),
                mime_type=mime_type,
                storage_path=storage_key,
                status='pending',
                metadata=metadata or {}
            )
            session.add(document)
            session.commit()
            document_id = document.id
        
        # Create thumbnail for images
        if mime_type.startswith('image/'):
            thumbnail_task = create_thumbnail.delay(str(document_id))
            thumbnail_task_id = thumbnail_task.id
        else:
            thumbnail_task_id = None
        
        return {
            "document_id": str(document_id),
            "storage_path": storage_key,
            "file_size": len(file_data),
            "mime_type": mime_type,
            "upload_result": result,
            "thumbnail_task_id": thumbnail_task_id
        }
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise


@app.task(bind=True, base=StorageTask, name="tasks.storage_tasks.create_thumbnail")
def create_thumbnail(self, document_id: str, size: tuple = (300, 300)) -> Dict:
    """Create and store document thumbnail"""
    try:
        storage = MinIOStorage()
        
        # Get document info
        with get_db_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            storage_path = document.storage_path
        
        # Download original image
        file_data = storage.get_object("documents", storage_path)
        
        # Create thumbnail
        image = Image.open(io.BytesIO(file_data))
        
        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Create thumbnail
        image.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Save to bytes
        thumb_io = io.BytesIO()
        image.save(thumb_io, 'JPEG', quality=85, optimize=True)
        thumb_data = thumb_io.getvalue()
        
        # Upload thumbnail
        thumb_key = f"thumbnails/{document_id}.jpg"
        result = storage.put_object(
            "processed",
            thumb_key,
            io.BytesIO(thumb_data),
            len(thumb_data),
            "image/jpeg"
        )
        
        # Update database
        with get_db_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if document:
                document.thumbnail_path = thumb_key
                session.commit()
        
        return {
            "document_id": document_id,
            "thumbnail_path": thumb_key,
            "thumbnail_size": size,
            "file_size": len(thumb_data)
        }
        
    except Exception as e:
        logger.error(f"Thumbnail creation failed: {e}")
        raise


@app.task(bind=True, base=StorageTask, name="tasks.storage_tasks.create_document_package")
def create_document_package(self, document_id: str) -> Dict:
    """Create ZIP package with all document outputs"""
    try:
        import zipfile
        storage = MinIOStorage()
        
        # Get document info
        with get_db_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            detection_results = session.query(DetectionResult).filter_by(
                document_id=document_id
            ).all()
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add original document
            try:
                original_data = storage.get_object("documents", document.storage_path)
                zipf.writestr(f"original/{document.original_filename}", original_data)
            except:
                logger.warning(f"Could not add original file to package")
            
            # Add cleaned image
            if document.processed_path:
                try:
                    cleaned_data = storage.get_object("processed", document.processed_path)
                    zipf.writestr("processed/cleaned_document.png", cleaned_data)
                except:
                    logger.warning(f"Could not add cleaned image to package")
            
            # Add detection results
            for i, result in enumerate(detection_results):
                if result.detections:
                    import json
                    detection_json = json.dumps(result.detections, indent=2)
                    zipf.writestr(
                        f"detections/detection_result_{i+1}.json",
                        detection_json
                    )
            
            # Add metadata
            metadata = {
                "document_id": str(document_id),
                "original_filename": document.original_filename,
                "processed_at": datetime.utcnow().isoformat(),
                "document_type": document.document_type,
                "quality_metrics": document.quality_metrics,
                "detection_count": sum(r.detection_count for r in detection_results)
            }
            
            import json
            zipf.writestr("metadata.json", json.dumps(metadata, indent=2))
        
        # Upload package
        zip_data = zip_buffer.getvalue()
        package_key = f"packages/{document_id}_package.zip"
        
        result = storage.put_object(
            "processed",
            package_key,
            io.BytesIO(zip_data),
            len(zip_data),
            "application/zip"
        )
        
        # Generate download URL
        download_url = storage.generate_presigned_url(
            "processed",
            package_key,
            expires_in=86400  # 24 hours
        )
        
        return {
            "document_id": document_id,
            "package_path": package_key,
            "package_size": len(zip_data),
            "download_url": download_url,
            "expires_in": 86400
        }
        
    except Exception as e:
        logger.error(f"Package creation failed: {e}")
        raise


@app.task(bind=True, base=StorageTask, name="tasks.storage_tasks.cleanup_old_files")
def cleanup_old_files(self, days_old: int = 30) -> Dict:
    """Clean up old files from storage"""
    try:
        storage = MinIOStorage()
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        deleted_count = 0
        deleted_size = 0
        
        # Check temporary files
        temp_objects = storage.list_objects("temp")
        
        for obj in temp_objects:
            # Parse last modified date
            last_modified = datetime.fromisoformat(
                obj['last_modified'].replace('Z', '+00:00')
            ).replace(tzinfo=None)
            
            if last_modified < cutoff_date:
                storage.delete_object("temp", obj['key'])
                deleted_count += 1
                deleted_size += obj['size']
                logger.info(f"Deleted old temp file: {obj['key']}")
        
        # Clean up orphaned processed files
        with get_db_session() as session:
            # Find documents marked for deletion
            old_documents = session.query(Document).filter(
                Document.status == 'archived',
                Document.updated_at < cutoff_date
            ).all()
            
            for doc in old_documents:
                # Delete from storage
                try:
                    if doc.storage_path:
                        storage.delete_object("documents", doc.storage_path)
                    if doc.processed_path:
                        storage.delete_object("processed", doc.processed_path)
                    if doc.thumbnail_path:
                        storage.delete_object("processed", doc.thumbnail_path)
                    
                    # Delete from database
                    session.delete(doc)
                    deleted_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to delete document {doc.id}: {e}")
            
            session.commit()
        
        return {
            "deleted_count": deleted_count,
            "deleted_size_bytes": deleted_size,
            "cutoff_date": cutoff_date.isoformat(),
            "message": f"Cleaned up {deleted_count} files older than {days_old} days"
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise


@app.task(bind=True, base=StorageTask, name="tasks.storage_tasks.backup_to_s3")
@CircuitBreaker(failure_threshold=3, recovery_timeout=300)
def backup_to_s3(self, bucket_name: str, prefix: str = "backups/") -> Dict:
    """Backup MinIO data to external S3"""
    try:
        # External S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )
        
        storage = MinIOStorage()
        
        backed_up = 0
        total_size = 0
        
        # Backup each bucket
        for bucket in ['documents', 'processed', 'models']:
            objects = storage.list_objects(bucket)
            
            for obj in objects:
                try:
                    # Download from MinIO
                    data = storage.get_object(bucket, obj['key'])
                    
                    # Upload to S3
                    s3_key = f"{prefix}{bucket}/{obj['key']}"
                    s3_client.put_object(
                        Bucket=bucket_name,
                        Key=s3_key,
                        Body=data
                    )
                    
                    backed_up += 1
                    total_size += obj['size']
                    
                except Exception as e:
                    logger.error(f"Failed to backup {obj['key']}: {e}")
        
        return {
            "backed_up_count": backed_up,
            "total_size_bytes": total_size,
            "destination": f"s3://{bucket_name}/{prefix}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"S3 backup failed: {e}")
        raise


@app.task(bind=True, base=StorageTask, name="tasks.storage_tasks.generate_signed_urls")
def generate_signed_urls(self, document_id: str, expires_in: int = 3600) -> Dict:
    """Generate signed URLs for document access"""
    try:
        storage = MinIOStorage()
        urls = {}
        
        with get_db_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if not document:
                raise ValueError(f"Document {document_id} not found")
            
            # Original document URL
            if document.storage_path:
                urls['original'] = storage.generate_presigned_url(
                    "documents",
                    document.storage_path,
                    expires_in
                )
            
            # Processed document URL
            if document.processed_path:
                urls['processed'] = storage.generate_presigned_url(
                    "processed",
                    document.processed_path,
                    expires_in
                )
            
            # Thumbnail URL
            if document.thumbnail_path:
                urls['thumbnail'] = storage.generate_presigned_url(
                    "processed",
                    document.thumbnail_path,
                    expires_in
                )
            
            # Package URL if exists
            package_key = f"packages/{document_id}_package.zip"
            try:
                # Check if package exists
                storage.client.head_object(Bucket="processed", Key=package_key)
                urls['package'] = storage.generate_presigned_url(
                    "processed",
                    package_key,
                    expires_in
                )
            except:
                pass
        
        return {
            "document_id": document_id,
            "urls": urls,
            "expires_in": expires_in,
            "expires_at": (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate signed URLs: {e}")
        raise


def _generate_detection_report(detection_result: Dict) -> Dict:
    """Generate detailed detection report"""
    detections = detection_result.get("detections", [])
    
    # Group by class
    class_summary = {}
    for det in detections:
        class_name = det.get("class_name", "unknown")
        if class_name not in class_summary:
            class_summary[class_name] = {
                "count": 0,
                "avg_confidence": 0,
                "confidences": []
            }
        
        class_summary[class_name]["count"] += 1
        class_summary[class_name]["confidences"].append(det.get("confidence", 0))
    
    # Calculate averages
    for class_name, summary in class_summary.items():
        if summary["confidences"]:
            summary["avg_confidence"] = sum(summary["confidences"]) / len(summary["confidences"])
        del summary["confidences"]  # Remove raw data
    
    report = {
        "document_id": detection_result.get("document_id"),
        "timestamp": datetime.utcnow().isoformat(),
        "total_detections": len(detections),
        "average_confidence": detection_result.get("average_confidence", 0),
        "processing_time_ms": detection_result.get("processing_time_ms", 0),
        "class_summary": class_summary,
        "detection_details": detections
    }
    
    return report