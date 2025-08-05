import os
from celery import Celery
from celery.utils.log import get_task_logger

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Initialize Celery
celery_app = Celery(
    "tasks",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/0",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
)

logger = get_task_logger(__name__)


@celery_app.task(name="process_batch_detection")
def process_batch_detection(image_data: bytes, filename: str):
    """
    Celery task to process a single image from a batch detection request.
    """
    from api_server import model_instance, process_image_async, DetectionRequest
    import numpy as np
    import cv2
    import asyncio

    logger.info(f"Processing file: {filename}")

    try:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image data")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        params = DetectionRequest()
        results = asyncio.run(process_image_async(image_rgb, params))

        return {
            "filename": filename,
            "result": {
                "success": True,
                "detections": [d.dict() for d in results["detections"]],
            },
        }
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return {
            "filename": filename,
            "error": str(e),
        }