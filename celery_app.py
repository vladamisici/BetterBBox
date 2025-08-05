"""
Enhanced Celery Configuration with RabbitMQ
Distributed task processing for document intelligence system
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import timedelta
from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, task_failure, task_retry
from kombu import Queue, Exchange
import redis
from prometheus_client import Counter, Histogram, Gauge
import time
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

# Configuration
RABBITMQ_URL = os.getenv(
    "RABBITMQ_URL",
    "amqp://admin:admin@localhost:5672/document_detection"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/1")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://detector:secretpassword@localhost:5432/document_detection"
)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
app = Celery(
    "document_intelligence",
    broker=RABBITMQ_URL,
    backend=REDIS_URL,
    include=[
        "tasks.document_tasks",
        "tasks.cleaning_tasks",
        "tasks.detection_tasks",
        "tasks.storage_tasks",
        "tasks.notification_tasks"
    ]
)

# Celery configuration
app.conf.update(
    # Task execution settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=3600 * 24,  # 24 hours
    timezone="UTC",
    enable_utc=True,
    
    # Worker settings
    worker_prefetch_multiplier=2,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Task routing
    task_routes={
        "tasks.document_tasks.*": {"queue": "document_processing"},
        "tasks.cleaning_tasks.*": {"queue": "document_cleaning"},
        "tasks.detection_tasks.*": {"queue": "model_inference"},
        "tasks.storage_tasks.*": {"queue": "storage"},
        "tasks.notification_tasks.*": {"queue": "notifications"}
    },
    
    # Task priorities
    task_default_priority=5,
    task_inherit_parent_priority=True,
    
    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Result backend settings
    result_backend_transport_options={
        "visibility_timeout": 3600,
        "fanout_prefix": True,
        "fanout_patterns": True
    },
    
    # Beat schedule
    beat_schedule={
        "cleanup-expired-cache": {
            "task": "tasks.maintenance.cleanup_expired_cache",
            "schedule": timedelta(hours=1)
        },
        "refresh-model-cache": {
            "task": "tasks.maintenance.refresh_model_cache",
            "schedule": timedelta(hours=6)
        },
        "generate-usage-reports": {
            "task": "tasks.reporting.generate_usage_reports",
            "schedule": timedelta(days=1)
        },
        "cleanup-old-documents": {
            "task": "tasks.maintenance.cleanup_old_documents",
            "schedule": timedelta(days=1)
        },
        "health-check": {
            "task": "tasks.monitoring.health_check",
            "schedule": timedelta(minutes=5)
        }
    }
)

# Define exchanges and queues
default_exchange = Exchange("default", type="direct")
document_exchange = Exchange("documents", type="topic")
priority_exchange = Exchange("priority", type="direct")

app.conf.task_queues = (
    Queue("default", default_exchange, routing_key="default"),
    Queue("document_processing", document_exchange, routing_key="document.*"),
    Queue("document_cleaning", document_exchange, routing_key="cleaning.*"),
    Queue("model_inference", document_exchange, routing_key="inference.*", 
          queue_arguments={"x-max-priority": 10}),
    Queue("storage", default_exchange, routing_key="storage"),
    Queue("notifications", default_exchange, routing_key="notifications"),
    Queue("priority", priority_exchange, routing_key="priority",
          queue_arguments={"x-max-priority": 10})
)

# Metrics
task_counter = Counter(
    "celery_tasks_total",
    "Total number of tasks",
    ["task_name", "status"]
)
task_duration = Histogram(
    "celery_task_duration_seconds",
    "Task execution duration",
    ["task_name"]
)
active_tasks = Gauge(
    "celery_active_tasks",
    "Number of active tasks",
    ["task_name"]
)
task_retry_counter = Counter(
    "celery_task_retries_total",
    "Total number of task retries",
    ["task_name"]
)

# Database setup
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=40)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db_session():
    """Database session context manager"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Redis client for caching
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


class DocumentTask(Task):
    """Base task class with enhanced error handling and monitoring"""
    
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3, "countdown": 60}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    
    def __init__(self):
        super().__init__()
        self._redis_client = None
        self._db_session = None
    
    @property
    def redis_client(self):
        if self._redis_client is None:
            self._redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        return self._redis_client
    
    def before_start(self, task_id, args, kwargs):
        """Called before task execution"""
        active_tasks.labels(task_name=self.name).inc()
        logger.info(f"Starting task {self.name} with ID {task_id}")
    
    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """Called after task execution"""
        active_tasks.labels(task_name=self.name).dec()
        task_counter.labels(task_name=self.name, status=status).inc()
        
        # Log to database
        try:
            with get_db_session() as session:
                # Update job status in database
                from models import ProcessingJob
                job = session.query(ProcessingJob).filter_by(
                    celery_task_id=task_id
                ).first()
                if job:
                    job.status = status
                    job.completed_at = time.time()
                    session.commit()
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure"""
        logger.error(f"Task {self.name} failed: {exc}")
        task_counter.labels(task_name=self.name, status="failure").inc()
        
        # Send notification
        from tasks.notification_tasks import send_failure_notification
        send_failure_notification.delay(
            task_name=self.name,
            task_id=task_id,
            error=str(exc)
        )
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called on task retry"""
        logger.warning(f"Retrying task {self.name}: {exc}")
        task_retry_counter.labels(task_name=self.name).inc()


# Task base classes for different types
class CleaningTask(DocumentTask):
    """Base class for document cleaning tasks"""
    queue = "document_cleaning"
    priority = 7


class DetectionTask(DocumentTask):
    """Base class for detection tasks"""
    queue = "model_inference"
    priority = 8


class StorageTask(DocumentTask):
    """Base class for storage tasks"""
    queue = "storage"
    priority = 5


# Circuit breaker implementation
class CircuitBreaker:
    """Circuit breaker for external service calls"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=60, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                    logger.info(f"Circuit breaker half-open for {func.__name__}")
                else:
                    raise Exception(f"Circuit breaker open for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info(f"Circuit breaker closed for {func.__name__}")
                return result
            
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error(f"Circuit breaker open for {func.__name__}")
                
                raise e
        
        return wrapper


# Rate limiter
class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_calls=100, time_window=60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.redis_client = redis.from_url(REDIS_URL)
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            key = f"rate_limit:{func.__name__}:{kwargs.get('user_id', 'anonymous')}"
            
            try:
                current = self.redis_client.incr(key)
                if current == 1:
                    self.redis_client.expire(key, self.time_window)
                
                if current > self.max_calls:
                    raise Exception(f"Rate limit exceeded for {func.__name__}")
                
                return func(*args, **kwargs)
            
            except redis.RedisError:
                # Allow if Redis is down
                logger.warning("Redis unavailable for rate limiting")
                return func(*args, **kwargs)
        
        return wrapper


# Signal handlers
@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **kw):
    """Handle task pre-run events"""
    logger.info(f"Task {task.name} starting with ID {task_id}")
    
    # Start timer for duration tracking
    task.request.start_time = time.time()


@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, **kw):
    """Handle task post-run events"""
    duration = time.time() - getattr(task.request, "start_time", time.time())
    task_duration.labels(task_name=task.name).observe(duration)
    
    logger.info(f"Task {task.name} completed in {duration:.2f}s")


@task_failure.connect
def task_failure_handler(task_id, exception, args, kwargs, traceback, einfo, **kw):
    """Handle task failures"""
    logger.error(f"Task {task_id} failed: {exception}")
    
    # Store failure details in Redis for analysis
    failure_key = f"task_failure:{task_id}"
    failure_data = {
        "task_id": task_id,
        "exception": str(exception),
        "traceback": str(traceback),
        "timestamp": time.time()
    }
    
    redis_client.setex(
        failure_key,
        3600 * 24,  # Keep for 24 hours
        json.dumps(failure_data)
    )


@task_retry.connect
def task_retry_handler(task_id, exception, args, kwargs, traceback, einfo, **kw):
    """Handle task retries"""
    logger.warning(f"Task {task_id} retrying due to: {exception}")


# Priority task decorator
def priority_task(priority_level=5):
    """Decorator to set task priority"""
    def decorator(func):
        func.priority = priority_level
        func.queue = "priority" if priority_level >= 8 else "default"
        return func
    return decorator


# Long running task support
def chunked_task(chunk_size=100):
    """Decorator for processing large datasets in chunks"""
    def decorator(func):
        def wrapper(items, *args, **kwargs):
            results = []
            for i in range(0, len(items), chunk_size):
                chunk = items[i:i + chunk_size]
                result = func(chunk, *args, **kwargs)
                results.extend(result)
            return results
        return wrapper
    return decorator


# Export task base classes
app.Task = DocumentTask

__all__ = [
    "app",
    "DocumentTask",
    "CleaningTask", 
    "DetectionTask",
    "StorageTask",
    "CircuitBreaker",
    "RateLimiter",
    "priority_task",
    "chunked_task",
    "get_db_session"
]