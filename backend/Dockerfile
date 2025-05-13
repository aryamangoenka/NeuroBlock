FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY test-requirements.txt ./

# Install all dependencies + gunicorn
RUN pip install --no-cache-dir -r test-requirements.txt

# Copy app code
COPY . .

# Create directory for exports
RUN mkdir -p exports

# Expose Flask port
EXPOSE 5000

# ✅ Fix 1: Tell Python where to find `backend/`
ENV PYTHONPATH=/app

# ✅ Fix 2: Correct entrypoint for gunicorn
CMD ["gunicorn", "--bind", ":5000", "--workers", "1", "--threads", "8", "--timeout", "0", "backend.main:wsgi_app"]
