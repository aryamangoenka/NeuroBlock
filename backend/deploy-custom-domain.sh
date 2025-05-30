#!/bin/bash

# Exit on error
set -e

echo "🚀 Deploying backend to api.neuroblock.co..."

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t dnd-neural-backend:custom -f Dockerfile.custom .

# Stop any existing container
echo "🛑 Stopping existing container..."
docker stop dnd-neural-backend || true
docker rm dnd-neural-backend || true

# Run the new container
echo "🚀 Starting new container..."
docker run -d \
  --name dnd-neural-backend \
  -p 5000:5000 \
  -e FLASK_CONFIG=custom_domain \
  -e SECRET_KEY=$(openssl rand -hex 32) \
  -e LOG_LEVEL=INFO \
  dnd-neural-backend:custom

echo "✅ Deployment complete! Backend is running at api.neuroblock.co"
echo "📝 Make sure to:"
echo "  1. Configure your DNS to point api.neuroblock.co to this server"
echo "  2. Set up SSL certificates for api.neuroblock.co"
echo "  3. Configure your reverse proxy (nginx/apache) to forward requests to localhost:5000" 