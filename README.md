# Enhanced Document Detection System

This project provides a production-ready REST API for an enhanced document detection system. The system can identify various elements in a document, such as text, titles, tables, figures, and more. It also includes features like authentication, rate limiting, caching, and monitoring.

## Features

-   **High-Accuracy Detection**: Utilizes an enhanced content detector model to accurately identify various document elements.
-   **Production-Ready API**: Built with FastAPI, providing a robust and scalable REST API.
-   **Authentication**: Secure your API with JWT-based authentication.
-   **Rate Limiting**: Protect your API from abuse with rate limiting.
-   **Caching**: Improve performance by caching detection results with Redis.
-   **Monitoring**: Monitor your API with Prometheus metrics.
-   **Batch Processing**: Process multiple images in a single request.
-   **WebSocket Support**: Real-time detection with WebSocket.
-   **Dockerized**: Comes with a multi-stage Dockerfile for easy deployment in development and production environments.

## Getting Started

### Prerequisites

-   Docker
-   Docker Compose

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/better-bbox.git
    cd better-bbox
    ```

2.  **Set up environment variables:**

    Create a `.env` file by copying the example file:

    ```bash
    cp .env.example .env
    ```

    Update the `.env` file with your desired configurations, such as `SECRET_KEY`.

3.  **Build and run with Docker Compose:**

    ```bash
    docker-compose up --build
    ```

    This will start the API server, which will be accessible at `http://localhost:8000`.

## API Usage

### Authentication

First, you need to obtain an authentication token by sending a POST request to the `/auth/token` endpoint with a demo username and password:

```bash
curl -X POST "http://localhost:8000/auth/token" -d "username=demo" -d "password=demo123"
```

The response will contain an `access_token` that you will need to include in the `Authorization` header for all subsequent requests.

### Endpoints

-   `POST /api/v1/detect`: Detect objects in a single image.
-   `POST /api/v1/batch`: Process multiple images in a single request.
-   `GET /api/v1/result/{request_id}`: Get cached detection results.
-   `GET /api/v1/classes`: Get a list of supported detection classes.
-   `POST /api/v1/feedback`: Submit feedback to improve the model.
-   `GET /health`: Health check endpoint.
-   `GET /metrics`: Prometheus metrics endpoint.

### Example: Single Image Detection

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
-H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" \
-F "file=@/path/to/your/image.jpg"
```

## Testing

To run the test suite, you can use the following command:

```bash
docker-compose exec api python test_system.py
```

This will run all unit, integration, and performance tests.