#!/bin/bash

# GCP Deployment Script for Session-Based Dataset Application
# This script sets up and deploys your application to Google Cloud Platform

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 DND Neural Network - GCP Deployment${NC}"
echo "========================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed. Please install it first.${NC}"
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}⚠️  No project set. Please set your GCP project:${NC}"
    echo "gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo -e "${GREEN}✅ Using project: $PROJECT_ID${NC}"

# Enable required APIs
echo -e "\n${YELLOW}📡 Enabling required GCP APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Create secret for Flask session key
echo -e "\n${YELLOW}🔐 Setting up Secret Manager...${NC}"
SECRET_KEY=$(openssl rand -base64 32)

# Check if secret already exists
if gcloud secrets describe flask-secret-key &>/dev/null; then
    echo -e "${YELLOW}⚠️  Secret 'flask-secret-key' already exists. Updating...${NC}"
    echo -n "$SECRET_KEY" | gcloud secrets versions add flask-secret-key --data-file=-
else
    echo -e "${GREEN}✅ Creating new secret 'flask-secret-key'${NC}"
    echo -n "$SECRET_KEY" | gcloud secrets create flask-secret-key --data-file=-
fi

# Grant Cloud Build access to secrets
echo -e "\n${YELLOW}🔑 Setting up IAM permissions...${NC}"
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet

# Grant Cloud Run access to secrets
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet

echo -e "${GREEN}✅ IAM permissions configured${NC}"

# Build and deploy
echo -e "\n${YELLOW}🏗️  Building and deploying application...${NC}"
gcloud builds submit --config cloudbuild.yaml

# Get the service URL
SERVICE_URL=$(gcloud run services describe dnd-neural-backend --region=us-central1 --format="value(status.url)")

echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
echo "========================================"
echo -e "${BLUE}Service URL:${NC} $SERVICE_URL"
echo -e "${BLUE}Health Check:${NC} $SERVICE_URL/health"
echo -e "${BLUE}API Endpoint:${NC} $SERVICE_URL/api/datasets/available"

# Test the deployment
echo -e "\n${YELLOW}🧪 Testing deployment...${NC}"
if curl -s "$SERVICE_URL/health" | grep -q "healthy"; then
    echo -e "${GREEN}✅ Health check passed!${NC}"
else
    echo -e "${RED}❌ Health check failed. Check Cloud Run logs.${NC}"
fi

# Display monitoring information
echo -e "\n${BLUE}📊 Monitoring & Management:${NC}"
echo "- Cloud Run Console: https://console.cloud.google.com/run/detail/us-central1/dnd-neural-backend"
echo "- Cloud Build History: https://console.cloud.google.com/cloud-build/builds"
echo "- Secret Manager: https://console.cloud.google.com/security/secret-manager"
echo "- Logs: gcloud logs tail projects/$PROJECT_ID/logs/run.googleapis.com%2Fstdout"

echo -e "\n${GREEN}🎯 Next Steps:${NC}"
echo "1. Update your frontend API_BASE_URL to: $SERVICE_URL"
echo "2. Test session isolation by opening multiple browser tabs"
echo "3. Monitor usage in Cloud Run console"
echo "4. Set up custom domain if needed"

echo -e "\n${BLUE}💡 Useful Commands:${NC}"
echo "- View logs: gcloud logs tail projects/$PROJECT_ID/logs/run.googleapis.com%2Fstdout"
echo "- Redeploy: gcloud builds submit --config cloudbuild.yaml"
echo "- Update secrets: echo 'new-secret' | gcloud secrets versions add flask-secret-key --data-file=-" 