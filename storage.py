"""
Storage Module
MinIO/S3 compatible storage client and utilities
"""

import os
import io
import logging
import boto3
from botocore.exceptions import ClientError
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

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


class S3Storage(MinIOStorage):
    """AWS S3 storage client (extends MinIO client)"""
    
    def __init__(self):
        self.endpoint = None  # Use default AWS endpoint
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        
        self.client = boto3.client(
            's3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region
        )
    
    def _ensure_buckets(self):
        """Override - don't create buckets in S3"""
        pass
    
    def _set_public_read_policy(self, bucket_name: str):
        """Override - use S3-specific bucket policy"""
        pass


# Storage factory
def get_storage_client(storage_type: str = "minio") -> MinIOStorage:
    """
    Get appropriate storage client
    
    Args:
        storage_type: Type of storage ('minio' or 's3')
        
    Returns:
        Storage client instance
    """
    if storage_type.lower() == "s3":
        return S3Storage()
    else:
        return MinIOStorage()


__all__ = ['MinIOStorage', 'S3Storage', 'get_storage_client']