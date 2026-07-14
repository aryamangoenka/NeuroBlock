# Google Cloud Platform Deployment Guide

## 🚀 Session-Based Dataset Application on GCP

This guide walks you through deploying your session-based dataset application to Google Cloud Platform using Cloud Run.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **gcloud CLI** installed and configured
3. **Docker** (optional, Cloud Build handles this)

## Quick Deployment

### Option 1: Automated Deployment (Recommended)

```bash
# 1. Set your GCP project
gcloud config set project YOUR_PROJECT_ID

# 2. Run the automated deployment script
./deploy-gcp.sh
```

### Option 2: Manual Deployment

```bash
# 1. Set your project
gcloud config set project YOUR_PROJECT_ID

# 2. Enable required APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com secretmanager.googleapis.com

# 3. Create secret key
echo "$(openssl rand -base64 32)" | gcloud secrets create flask-secret-key --data-file=-

# 4. Deploy
gcloud builds submit --config cloudbuild.yaml
```

## Configuration Details

### Environment Variables

Your application will be deployed with these environment variables:

```bash
FLASK_CONFIG=production
SESSION_CLEANUP_INTERVAL=12    # Clean up every 12 hours
SESSION_MAX_AGE=168           # Sessions last 7 days (168 hours)
GOOGLE_CLOUD_PROJECT=your-project-id
SECRET_KEY=[from Secret Manager]
PORT=8080
```

### Cloud Run Settings

```yaml
Memory: 2Gi
CPU: 2 cores
Max Instances: 10
Min Instances: 0
Concurrency: 80 requests per instance
Timeout: 300 seconds
```

## Session Management on GCP

### How Sessions Work in Production

1. **User Visits**: New session created with UUID
2. **Dataset Upload**: Stored in `/app/datasets/sessions/{session-id}/`
3. **Session Persistence**: 7 days with secure cookies
4. **Automatic Cleanup**: Every 12 hours, removes sessions older than 7 days
5. **Container Restart**: Sessions are ephemeral (cleared on restart)

### Session Isolation

```
User A (Chrome) → Session: abc123 → Datasets: [dataset1, dataset2]
User B (Firefox) → Session: def456 → Datasets: [dataset3]
User C (Incognito) → Session: ghi789 → Datasets: []
```

Each user gets completely isolated storage.

## Monitoring & Management

### Cloud Console Links

After deployment, access these consoles:

- **Cloud Run**: https://console.cloud.google.com/run
- **Cloud Build**: https://console.cloud.google.com/cloud-build
- **Secret Manager**: https://console.cloud.google.com/security/secret-manager
- **Logs**: https://console.cloud.google.com/logs

### Useful Commands

```bash
# View real-time logs
gcloud logs tail projects/YOUR_PROJECT_ID/logs/run.googleapis.com%2Fstdout

# Check service status
gcloud run services describe dnd-neural-backend --region=us-central1

# Update environment variables
gcloud run services update dnd-neural-backend \
  --region=us-central1 \
  --set-env-vars="SESSION_MAX_AGE=336"  # 14 days

# Scale service
gcloud run services update dnd-neural-backend \
  --region=us-central1 \
  --max-instances=20

# View metrics
gcloud run services describe dnd-neural-backend \
  --region=us-central1 \
  --format="table(status.traffic[].percent,status.traffic[].revisionName)"
```

## Cost Optimization

### Pricing Breakdown

```
Cloud Run Pricing (us-central1):
- CPU: $0.00002400 per vCPU-second
- Memory: $0.00000250 per GiB-second
- Requests: $0.40 per million requests

Estimated Monthly Costs:
- 1,000 users/month: ~$15-25
- 10,000 users/month: ~$50-100
- 100,000 users/month: ~$200-400
```

### Cost Optimization Tips

1. **Auto-scaling**: Scales to zero when not in use
2. **Session cleanup**: Prevents storage bloat
3. **Efficient memory**: 2Gi handles multiple concurrent sessions
4. **Request-based billing**: Only pay for actual usage

## Security Features

### Built-in Security

- ✅ **HTTPS Only**: Automatic SSL/TLS certificates
- ✅ **Secret Management**: Flask secret key in Secret Manager
- ✅ **Session Security**: HttpOnly, Secure, SameSite cookies
- ✅ **Container Security**: Non-root user, minimal attack surface
- ✅ **Network Security**: VPC-native networking
- ✅ **IAM Integration**: Fine-grained access controls

### Session Security

```python
# Production session configuration
SESSION_COOKIE_SECURE = True      # HTTPS only
SESSION_COOKIE_HTTPONLY = True    # No JavaScript access
SESSION_COOKIE_SAMESITE = 'Lax'   # CSRF protection
PERMANENT_SESSION_LIFETIME = 7 days # Auto-expiry
```

## Troubleshooting

### Common Issues

#### 1. Deployment Fails

```bash
# Check build logs
gcloud builds log $(gcloud builds list --limit=1 --format="value(id)")

# Common fixes:
- Ensure all APIs are enabled
- Check IAM permissions
- Verify secret exists
```

#### 2. Health Check Fails

```bash
# Check service logs
gcloud logs tail projects/YOUR_PROJECT_ID/logs/run.googleapis.com%2Fstdout

# Test locally first:
docker build -f backend/Dockerfile -t test-app .
docker run -p 8080:8080 -e FLASK_CONFIG=development test-app
```

#### 3. Sessions Not Working

```bash
# Check environment variables
gcloud run services describe dnd-neural-backend --region=us-central1 --format="value(spec.template.spec.template.spec.containers[0].env[].name,spec.template.spec.template.spec.containers[0].env[].value)"

# Verify secret access
gcloud secrets versions access latest --secret="flask-secret-key"
```

#### 4. High Memory Usage

```bash
# Monitor memory usage
gcloud run services describe dnd-neural-backend --region=us-central1 --format="value(status.conditions[].message)"

# Increase memory if needed
gcloud run services update dnd-neural-backend --region=us-central1 --memory=4Gi
```

## Frontend Integration

### Update API Configuration

```typescript
// frontend/src/utils/apiConfig.ts
const API_BASE_URL =
  process.env.NODE_ENV === "production"
    ? "https://your-service-url.run.app" // Replace with your actual URL
    : "http://localhost:5000";

export default API_BASE_URL;
```

### CORS Configuration

Your backend is already configured for CORS with credentials:

```python
# In backend/main.py
CORS(app, supports_credentials=True)
```

## Advanced Configuration

### Custom Domain

```bash
# Map custom domain
gcloud run domain-mappings create \
  --service=dnd-neural-backend \
  --domain=api.yourdomain.com \
  --region=us-central1
```

### Cloud Storage Integration (Optional)

For persistent session storage across deployments:

```python
# Add to requirements.txt
google-cloud-storage==2.10.0

# Update session_manager.py for GCS integration
from google.cloud import storage

def get_session_datasets_dir(session_id=None):
    if os.environ.get('GCS_BUCKET_NAME'):
        return f"gs://{os.environ['GCS_BUCKET_NAME']}/sessions/{session_id}"
    # Fallback to local storage
    return local_session_dir
```

### Monitoring & Alerting

```bash
# Create uptime check
gcloud alpha monitoring uptime create \
  --display-name="DND Neural Backend" \
  --http-check-path="/health" \
  --hostname="your-service-url.run.app"

# Create alert policy
gcloud alpha monitoring policies create \
  --policy-from-file=monitoring-policy.yaml
```

## Performance Optimization

### Recommended Settings

```yaml
# For high-traffic applications
Memory: 4Gi
CPU: 4 cores
Max Instances: 50
Min Instances: 1 # Keep warm
Concurrency: 100
```

### Database Integration (Future)

If you need persistent user accounts:

```bash
# Cloud SQL for user management
gcloud sql instances create dnd-neural-db \
  --database-version=POSTGRES_13 \
  --tier=db-f1-micro \
  --region=us-central1
```

## Backup & Recovery

### Session Data Backup

```bash
# Create backup of session directories (if using persistent storage)
gsutil -m cp -r gs://your-bucket/sessions gs://your-backup-bucket/sessions-$(date +%Y%m%d)
```

### Configuration Backup

```bash
# Export current configuration
gcloud run services describe dnd-neural-backend \
  --region=us-central1 \
  --format="export" > service-config-backup.yaml
```

## Next Steps

1. **Deploy**: Run `./deploy-gcp.sh`
2. **Test**: Verify session isolation works
3. **Monitor**: Set up alerts and monitoring
4. **Scale**: Adjust resources based on usage
5. **Secure**: Add custom domain and additional security measures

## Support

- **GCP Documentation**: https://cloud.google.com/run/docs
- **Cloud Run Pricing**: https://cloud.google.com/run/pricing
- **Support**: https://cloud.google.com/support

Your session-based dataset application is now ready for production deployment on Google Cloud Platform! 🚀
