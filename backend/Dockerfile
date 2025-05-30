FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    FLASK_CONFIG=custom_domain \
    FLASK_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn==21.2.0

# Copy the backend directory
COPY backend/ ./backend/
COPY model_config.json saved_model.json training_config.json ./

# Create necessary directories for session-based datasets
RUN mkdir -p exports export datasets/sessions

# Set the PYTHONPATH
ENV PYTHONPATH=.

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 8080

# Set health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Command to run the application with gunicorn and eventlet
CMD exec gunicorn --bind :$PORT --worker-class eventlet --workers 1 --threads 8 --timeout 0 backend.main:app 