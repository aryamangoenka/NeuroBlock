# Backend Commands

## Setting Environment Variables

### Option 1: Direct in Terminal (Temporary)

```bash
# Development
export FLASK_CONFIG=development
export PORT=5000
export HOST=localhost

# Production
export FLASK_CONFIG=custom_domain
export PORT=8080
export HOST=0.0.0.0
export SECRET_KEY=your-secret-key
```

### Option 2: Using .env file (Recommended)

```bash
# Create .env file in project root
touch .env

# Add these lines to .env file
FLASK_CONFIG=development
PORT=5000
HOST=localhost
SECRET_KEY=your-secret-key
```

### Option 3: Using Poetry's env

```bash
# Development
poetry env use python3.10
poetry run export FLASK_CONFIG=development
poetry run export PORT=5000
poetry run export HOST=localhost

# Production
poetry env use python3.10
poetry run export FLASK_CONFIG=custom_domain
poetry run export PORT=8080
poetry run export HOST=0.0.0.0
poetry run export SECRET_KEY=your-secret-key
```

## Development (Local)

```bash
# 1. Navigate to project directory
cd DND-Neural-Network

# 2. Install dependencies
poetry install

# 3. Set environment variables
export FLASK_CONFIG=development
export PORT=5000
export HOST=localhost

# 4. Run the server
poetry run python backend/main.py
```

## Production Deployment

### Using Cloud Build (Recommended)

```bash
# 1. Make sure you have the secret key in Secret Manager
echo -n "your-secret-key" | gcloud secrets create flask-secret-key --data-file=-

# 2. Deploy using Cloud Build
gcloud builds submit
```

Your `cloudbuild.yaml` already includes:

- Building and pushing Docker image
- Deploying to Cloud Run
- Setting environment variables:
  - `FLASK_CONFIG=custom_domain`
  - `SESSION_CLEANUP_INTERVAL=12`
  - `SESSION_MAX_AGE=168`
  - `GOOGLE_CLOUD_PROJECT=$PROJECT_ID`
- Secret key configuration
- 30-minute timeout
- Public access enabled

### Manual Production Server

```bash
# 1. Set environment variables
export FLASK_CONFIG=custom_domain
export PORT=8080
export HOST=0.0.0.0
export SECRET_KEY=your-secret-key

# 2. Install production dependencies
poetry install --no-dev

# 3. Run with Gunicorn
poetry run gunicorn -w 4 -k eventlet -b 0.0.0.0:8080 backend.main:app
```

## Common Commands

```bash
# Check if port is in use
lsof -i :5000

# Kill process using port
kill -9 $(lsof -t -i:5000)

# View Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=backend" --limit 50

# Run tests
poetry run pytest
```
