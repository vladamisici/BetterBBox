"""
Production-ready REST API for Document Detection Model
Includes authentication, rate limiting, caching, and monitoring
"""

import os
import io
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
import cv2
from PIL import Image
import redis
import jwt
from passlib.context import CryptContext
from prometheus_client import Counter, Histogram, generate_latest
import aiofiles

from enhanced_content_detector import EnhancedContentDetector, BoundingBox, visualize_results


# Configuration
class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pth")
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 3600  # 1 hour


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
detection_counter = Counter('document_detection_total', 'Total number of detection requests')
detection_histogram = Histogram('document_detection_duration_seconds', 'Detection request duration')
error_counter = Counter('document_detection_errors_total', 'Total number of errors')

# Initialize FastAPI app
app = FastAPI(
    title="Document Detection API",
    description="State-of-the-art document content detection service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

# Global model instance
model_instance = None
redis_client = None


# Pydantic models
class DetectionRequest(BaseModel):
    image_url: Optional[str] = Field(None, description="URL of the image to process")
    confidence_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Confidence threshold")
    nms_threshold: float = Field(0.5, ge=0.0, le=1.0, description="NMS threshold")
    use_ensemble: bool = Field(True, description="Use ensemble of models")
    return_visualization: bool = Field(False, description="Return visualized image")
    classes: Optional[List[str]] = Field(None, description="Filter results by class names")


class BoundingBoxResponse(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    area: float
    center_x: float
    center_y: float


class DetectionResponse(BaseModel):
    success: bool
    request_id: str
    timestamp: str
    num_detections: int
    processing_time: float
    image_dimensions: Dict[str, int]
    detections: List[BoundingBoxResponse]
    visualization_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool
    uptime: float
    version: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


# Helper functions
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify JWT token"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, Config.SECRET_KEY, algorithms=[Config.ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, Config.SECRET_KEY, algorithm=Config.ALGORITHM)
    return encoded_jwt


async def check_rate_limit(user_id: str):
    """Check rate limiting using Redis"""
    if not redis_client:
        return True
    
    key = f"rate_limit:{user_id}"
    try:
        current = await redis_client.incr(key)
        if current == 1:
            await redis_client.expire(key, Config.RATE_LIMIT_WINDOW)
        
        if current > Config.RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {Config.RATE_LIMIT_REQUESTS} requests per hour."
            )
        return True
    except redis.RedisError:
        logger.error("Redis error during rate limiting check")
        return True  # Allow request if Redis is down


def load_model():
    """Load the detection model"""
    global model_instance
    try:
        logger.info(f"Loading model from {Config.MODEL_PATH}")
        model_instance = EnhancedContentDetector()
        # Load pretrained weights
        import torch
        checkpoint = torch.load(Config.MODEL_PATH, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model_instance.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_instance.model.load_state_dict(checkpoint)
        model_instance.model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


async def process_image_async(image: np.ndarray, params: DetectionRequest) -> Dict:
    """Process image asynchronously"""
    loop = asyncio.get_event_loop()
    
    # Update detector configuration
    model_instance.config['thresholds']['confidence'] = params.confidence_threshold
    model_instance.config['thresholds']['nms'] = params.nms_threshold
    
    # Run detection in thread pool
    detections = await loop.run_in_executor(
        executor,
        model_instance.detect,
        image,
        params.use_ensemble
    )
    
    # Filter by classes if specified
    if params.classes:
        detections = [d for d in detections if d.class_name in params.classes]
    
    # Convert to response format
    detection_responses = []
    for det in detections:
        detection_responses.append(BoundingBoxResponse(
            x1=det.x1,
            y1=det.y1,
            x2=det.x2,
            y2=det.y2,
            confidence=det.confidence,
            class_id=det.class_id,
            class_name=det.class_name,
            area=(det.x2 - det.x1) * (det.y2 - det.y1),
            center_x=(det.x1 + det.x2) / 2,
            center_y=(det.y1 + det.y2) / 2
        ))
    
    return {
        'detections': detection_responses,
        'visualization': None
    }


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Load model
    load_model()
    
    # Connect to Redis
    global redis_client
    try:
        redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            decode_responses=True
        )
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Document Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "detect": "/api/v1/detect",
            "batch": "/api/v1/batch",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import psutil
    process = psutil.Process()
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_instance is not None,
        redis_connected=redis_client is not None and redis_client.ping(),
        uptime=time.time() - process.create_time(),
        version="1.0.0"
    )


@app.post("/auth/token", response_model=TokenResponse)
async def login(username: str, password: str):
    """Get authentication token"""
    # In production, verify against a user database
    if username == "demo" and password == "demo123":
        access_token = create_access_token(data={"sub": username})
        return TokenResponse(access_token=access_token, token_type="bearer")
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_single(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.3,
    nms_threshold: float = 0.5,
    use_ensemble: bool = True,
    return_visualization: bool = False,
    classes: Optional[str] = None,
    current_user: str = Depends(verify_token)
):
    """Detect objects in a single image"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Rate limiting
    await check_rate_limit(current_user)
    
    # Update metrics
    detection_counter.inc()
    
    try:
        # Validate file
        if file.size > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {Config.MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type. Allowed: {Config.ALLOWED_EXTENSIONS}"
            )
        
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file"
            )
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create request parameters
        params = DetectionRequest(
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            use_ensemble=use_ensemble,
            return_visualization=return_visualization,
            classes=classes.split(",") if classes else None
        )
        
        # Process image
        with detection_histogram.time():
            results = await process_image_async(image_rgb, params)
        
        # Generate visualization if requested
        visualization_url = None
        if return_visualization:
            viz_image = visualize_results(image_rgb, results['detections'])
            # In production, upload to S3 or similar
            # For now, we'll return base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(viz_image, cv2.COLOR_RGB2BGR))
            viz_base64 = base64.b64encode(buffer).decode('utf-8')
            visualization_url = f"data:image/jpeg;base64,{viz_base64}"
        
        processing_time = time.time() - start_time
        
        # Cache results if Redis available
        if redis_client:
            cache_key = f"result:{request_id}"
            cache_data = {
                'detections': [d.dict() for d in results['detections']],
                'processing_time': processing_time
            }
            await redis_client.setex(cache_key, 3600, json.dumps(cache_data))
        
        return DetectionResponse(
            success=True,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            num_detections=len(results['detections']),
            processing_time=processing_time,
            image_dimensions={
                'height': image.shape[0],
                'width': image.shape[1]
            },
            detections=results['detections'],
            visualization_url=visualization_url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_counter.inc()
        logger.error(f"Detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@app.post("/api/v1/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = 0.3,
    nms_threshold: float = 0.5,
    use_ensemble: bool = True,
    current_user: str = Depends(verify_token)
):
    """Process multiple images in batch"""
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 images per batch"
        )
    
    # Rate limiting (count as multiple requests)
    for _ in range(len(files)):
        await check_rate_limit(current_user)
    
    results = []
    for file in files:
        try:
            result = await detect_single(
                file=file,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                use_ensemble=use_ensemble,
                return_visualization=False,
                current_user=current_user
            )
            results.append({
                'filename': file.filename,
                'result': result.dict()
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {
        'success': True,
        'batch_size': len(files),
        'results': results
    }


@app.get("/api/v1/result/{request_id}")
async def get_cached_result(
    request_id: str,
    current_user: str = Depends(verify_token)
):
    """Get cached detection result"""
    if not redis_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache service unavailable"
        )
    
    cache_key = f"result:{request_id}"
    cached = await redis_client.get(cache_key)
    
    if not cached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result not found or expired"
        )
    
    return json.loads(cached)


@app.get("/api/v1/classes")
async def get_supported_classes():
    """Get list of supported detection classes"""
    from enhanced_content_detector import EnhancedClasses
    
    classes = []
    for class_name, class_id in EnhancedClasses.CLASSES.items():
        classes.append({
            'id': class_id,
            'name': class_name,
            'category': _get_class_category(class_name)
        })
    
    return {
        'total_classes': len(classes),
        'classes': sorted(classes, key=lambda x: x['id'])
    }


def _get_class_category(class_name: str) -> str:
    """Get category for a class"""
    categories = {
        'document': ['text', 'title', 'list', 'table', 'figure', 'caption', 
                    'header', 'footer', 'page_number'],
        'music': ['staff', 'measure', 'note', 'clef', 'time_signature', 'lyrics'],
        'form': ['checkbox', 'input_field', 'signature_field', 'dropdown'],
        'diagram': ['flowchart', 'graph', 'equation'],
        'special': ['barcode', 'qr_code', 'logo', 'stamp']
    }
    
    for category, items in categories.items():
        if class_name in items:
            return category
    return 'other'


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        io.BytesIO(generate_latest()),
        media_type="text/plain"
    )


@app.post("/api/v1/feedback")
async def submit_feedback(
    request_id: str,
    correct_detections: int,
    missed_detections: int,
    false_positives: int,
    comments: Optional[str] = None,
    current_user: str = Depends(verify_token)
):
    """Submit feedback for improving the model"""
    feedback = {
        'request_id': request_id,
        'user': current_user,
        'timestamp': datetime.utcnow().isoformat(),
        'correct_detections': correct_detections,
        'missed_detections': missed_detections,
        'false_positives': false_positives,
        'comments': comments
    }
    
    # Store feedback (in production, save to database)
    if redis_client:
        feedback_key = f"feedback:{request_id}"
        await redis_client.setex(feedback_key, 86400 * 7, json.dumps(feedback))
    
    return {
        'success': True,
        'message': 'Thank you for your feedback!'
    }


# WebSocket endpoint for real-time detection (optional)
from fastapi import WebSocket

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """WebSocket endpoint for real-time detection"""
    await websocket.accept()
    
    try:
        while True:
            # Receive image data
            data = await websocket.receive_bytes()
            
            # Decode image
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process image
                params = DetectionRequest()
                results = await process_image_async(image_rgb, params)
                
                # Send results
                await websocket.send_json({
                    'success': True,
                    'detections': [d.dict() for d in results['detections']]
                })
            else:
                await websocket.send_json({
                    'success': False,
                    'error': 'Invalid image data'
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'success': False,
            'error': exc.detail,
            'status_code': exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            'success': False,
            'error': 'Internal server error',
            'status_code': 500
        }
    )


# CLI for running the server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Document Detection API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker processes')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level="info"
    )