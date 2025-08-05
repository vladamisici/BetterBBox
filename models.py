"""
Database Models
SQLAlchemy models for document intelligence system
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    ForeignKey, Text, JSON, TIMESTAMP, Enum, BigInteger,
    Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
import enum

Base = declarative_base()


class DocumentStatus(enum.Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DocumentType(enum.Enum):
    """Document type classification"""
    ACADEMIC = "academic"
    MUSIC_SCORE = "music_score"
    FORM = "form"
    DIAGRAM = "diagram"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ProcessingStage(enum.Enum):
    """Processing pipeline stages"""
    UPLOADED = "uploaded"
    CLEANING = "cleaning"
    DETECTING = "detecting"
    POST_PROCESSING = "post_processing"
    STORING = "storing"
    COMPLETED = "completed"


class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login = Column(TIMESTAMP(timezone=True))
    metadata = Column(JSONB, default={})
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user")
    organizations = relationship("UserOrganization", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")
    feedback = relationship("Feedback", back_populates="user")
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format"""
        import re
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(pattern, email):
            raise ValueError("Invalid email format")
        return email.lower()
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class APIKey(Base):
    """API Key model for authentication"""
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    scopes = Column(ARRAY(Text), default=["read", "write"])
    rate_limit = Column(Integer, default=1000)
    expires_at = Column(TIMESTAMP(timezone=True))
    last_used_at = Column(TIMESTAMP(timezone=True))
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    metadata = Column(JSONB, default={})
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    performance_metrics = relationship("PerformanceMetric", back_populates="api_key")
    
    def is_expired(self) -> bool:
        """Check if API key is expired"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    def __repr__(self):
        return f"<APIKey(name='{self.name}', user_id='{self.user_id}')>"


class Organization(Base):
    """Organization model"""
    __tablename__ = "organizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False, index=True)
    domain = Column(String(255))
    is_active = Column(Boolean, default=True)
    settings = Column(JSONB, default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    users = relationship("UserOrganization", back_populates="organization")
    documents = relationship("Document", back_populates="organization")
    
    def __repr__(self):
        return f"<Organization(name='{self.name}')>"


class UserOrganization(Base):
    """User-Organization relationship"""
    __tablename__ = "user_organizations"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True)
    role = Column(String(50), default="member")
    joined_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="organizations")
    organization = relationship("Organization", back_populates="users")
    
    def __repr__(self):
        return f"<UserOrganization(user_id='{self.user_id}', org_id='{self.organization_id}', role='{self.role}')>"


class Document(Base):
    """Document model"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), index=True)
    original_filename = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    mime_type = Column(String(100), nullable=False)
    storage_path = Column(String(1000), nullable=False)
    processed_path = Column(String(1000))
    thumbnail_path = Column(String(1000))
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING, index=True)
    document_type = Column(Enum(DocumentType), default=DocumentType.UNKNOWN, index=True)
    page_count = Column(Integer, default=1)
    processing_time_ms = Column(Integer)
    error_message = Column(Text)
    metadata = Column(JSONB, default={})
    quality_metrics = Column(JSONB, default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    processed_at = Column(TIMESTAMP(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="documents")
    organization = relationship("Organization", back_populates="documents")
    detection_results = relationship("DetectionResult", back_populates="document", cascade="all, delete-orphan")
    processing_jobs = relationship("ProcessingJob", back_populates="document", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_documents_user_status", "user_id", "status"),
        Index("idx_documents_created_at_desc", created_at.desc()),
    )
    
    @validates('file_size')
    def validate_file_size(self, key, size):
        """Validate file size"""
        MAX_SIZE = 100 * 1024 * 1024  # 100MB
        if size > MAX_SIZE:
            raise ValueError(f"File size exceeds maximum allowed size of {MAX_SIZE} bytes")
        return size
    
    def __repr__(self):
        return f"<Document(id='{self.id}', filename='{self.original_filename}', status='{self.status.value}')>"


class DetectionResult(Base):
    """Detection results model"""
    __tablename__ = "detection_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = Column(Integer, default=1)
    detection_count = Column(Integer, default=0)
    confidence_avg = Column(Float)
    processing_time_ms = Column(Integer)
    model_version = Column(String(50))
    detections = Column(JSONB, nullable=False, default=[])
    metadata = Column(JSONB, default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    document = relationship("Document", back_populates="detection_results")
    feedback = relationship("Feedback", back_populates="detection_result")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('confidence_avg >= 0 AND confidence_avg <= 1', name='check_confidence_range'),
    )
    
    def get_class_summary(self) -> Dict[str, int]:
        """Get summary of detected classes"""
        summary = {}
        for detection in self.detections:
            class_name = detection.get('class_name', 'unknown')
            summary[class_name] = summary.get(class_name, 0) + 1
        return summary
    
    def __repr__(self):
        return f"<DetectionResult(document_id='{self.document_id}', count={self.detection_count})>"


class ProcessingJob(Base):
    """Processing job tracking model"""
    __tablename__ = "processing_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    job_type = Column(String(50), nullable=False)
    stage = Column(Enum(ProcessingStage), default=ProcessingStage.UPLOADED)
    celery_task_id = Column(String(255), unique=True, index=True)
    status = Column(String(50), default="pending", index=True)
    priority = Column(Integer, default=5)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    error_message = Column(Text)
    started_at = Column(TIMESTAMP(timezone=True))
    completed_at = Column(TIMESTAMP(timezone=True))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    metadata = Column(JSONB, default={})
    
    # Relationships
    document = relationship("Document", back_populates="processing_jobs")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('priority >= 0 AND priority <= 10', name='check_priority_range'),
        CheckConstraint('retry_count <= max_retries', name='check_retry_limit'),
    )
    
    def can_retry(self) -> bool:
        """Check if job can be retried"""
        return self.retry_count < self.max_retries
    
    def __repr__(self):
        return f"<ProcessingJob(id='{self.id}', type='{self.job_type}', status='{self.status}')>"


class AuditLog(Base):
    """Audit log model"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50))
    resource_id = Column(UUID(as_uuid=True))
    ip_address = Column(INET)
    user_agent = Column(Text)
    request_id = Column(UUID(as_uuid=True))
    changes = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index("idx_audit_logs_resource", "resource_type", "resource_id"),
        Index("idx_audit_logs_created_at_desc", created_at.desc()),
    )
    
    def __repr__(self):
        return f"<AuditLog(action='{self.action}', user_id='{self.user_id}', created_at='{self.created_at}')>"


class PerformanceMetric(Base):
    """Performance metrics model"""
    __tablename__ = "performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    endpoint = Column(String(255), nullable=False, index=True)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer)
    response_time_ms = Column(Integer)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="SET NULL"))
    request_size = Column(BigInteger)
    response_size = Column(BigInteger)
    error = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    user = relationship("User")
    api_key = relationship("APIKey", back_populates="performance_metrics")
    
    # Indexes
    __table_args__ = (
        Index("idx_performance_endpoint_created", "endpoint", "created_at"),
        Index("idx_performance_created_at_desc", created_at.desc()),
    )
    
    def __repr__(self):
        return f"<PerformanceMetric(endpoint='{self.endpoint}', status={self.status_code}, time={self.response_time_ms}ms)>"


class ModelVersion(Base):
    """Model version tracking"""
    __tablename__ = "model_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(100), nullable=False)
    file_path = Column(String(1000), nullable=False)
    accuracy = Column(Float)
    parameters = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    metadata = Column(JSONB, default={})
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_model_name_version'),
        CheckConstraint('accuracy >= 0 AND accuracy <= 1', name='check_accuracy_range'),
    )
    
    def __repr__(self):
        return f"<ModelVersion(name='{self.name}', version='{self.version}', active={self.is_active})>"


class Feedback(Base):
    """User feedback model"""
    __tablename__ = "feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    detection_result_id = Column(UUID(as_uuid=True), ForeignKey("detection_results.id", ondelete="CASCADE"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    rating = Column(Integer, CheckConstraint('rating >= 1 AND rating <= 5'))
    correct_detections = Column(Integer)
    missed_detections = Column(Integer)
    false_positives = Column(Integer)
    comments = Column(Text)
    corrections = Column(JSONB, default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="feedback")
    detection_result = relationship("DetectionResult", back_populates="feedback")
    user = relationship("User", back_populates="feedback")
    
    def __repr__(self):
        return f"<Feedback(document_id='{self.document_id}', rating={self.rating})>"


class CacheEntry(Base):
    """Cache entries for expensive operations"""
    __tablename__ = "cache_entries"
    
    key = Column(String(500), primary_key=True)
    value = Column(JSONB, nullable=False)
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    access_count = Column(Integer, default=0)
    last_accessed = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > self.expires_at
    
    def __repr__(self):
        return f"<CacheEntry(key='{self.key}', expires_at='{self.expires_at}')>"


# Create indexes and constraints after all models are defined
def create_indexes(engine):
    """Create additional indexes"""
    from sqlalchemy import text
    
    with engine.connect() as conn:
        # Full-text search index on document filenames
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_documents_filename_fts "
            "ON documents USING gin(to_tsvector('english', original_filename))"
        ))
        
        # Partial index for active API keys
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_api_keys_active "
            "ON api_keys(key_hash) WHERE is_active = true"
        ))
        
        # Composite index for document queries
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_documents_user_created "
            "ON documents(user_id, created_at DESC)"
        ))