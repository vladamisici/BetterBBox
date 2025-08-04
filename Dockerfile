# Multi-stage Dockerfile for Enhanced Document Detection System

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    gcc \
    g++ \
    make \
    cmake \
    wget \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-0 \
    libnotify-dev \
    libgstreamer1.0-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    # For music score processing
    musescore3 \
    lilypond \
    # For PDF generation
    texlive-latex-base \
    texlive-fonts-recommended \
    # Fonts
    fonts-liberation \
    fonts-dejavu-core \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 detector && \
    mkdir -p /app /data /models && \
    chown -R detector:detector /app /data /models

# Stage 2: Python dependencies
FROM base AS dependencies

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install Python packages
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    # Install additional packages for production
    python3 -m pip install --no-cache-dir \
        gunicorn \
        celery \
        flower \
        python-json-logger

# Stage 3: Development image
FROM dependencies AS development

# Install development tools
RUN python3 -m pip install --no-cache-dir \
    jupyterlab \
    ipdb \
    line_profiler \
    memory_profiler

# Set working directory
WORKDIR /app

# Switch to app user
USER detector

# Expose ports
EXPOSE 8000 8888 6006

# Development command
CMD ["python3", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Production image
FROM dependencies AS production

# Copy application code
COPY --chown=detector:detector . /app

# Set working directory
WORKDIR /app

# Download models (if not mounted)
RUN mkdir -p /app/models /app/data

# Switch to app user
USER detector

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command with gunicorn
CMD ["gunicorn", "api_server:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]

# Stage 5: Model optimization image
FROM dependencies AS optimizer

# Install optimization tools
RUN python3 -m pip install --no-cache-dir \
    onnx-simplifier \
    torch2trt \
    nvidia-tensorrt

WORKDIR /app
COPY --chown=detector:detector . /app

USER detector

# Command for model optimization
CMD ["python3", "optimize_model.py"]

# Stage 6: Training image with full CUDA toolkit
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS training

# Copy from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Install additional training dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install horovod for distributed training
RUN HOROVOD_GPU_OPERATIONS=NCCL python3 -m pip install --no-cache-dir horovod

# Copy application
COPY --chown=detector:detector . /app
WORKDIR /app

# Training command
CMD ["python3", "train_enhanced.py", "--config", "training_config.yaml"]