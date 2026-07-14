import os
from datetime import timedelta

class Config:
    """Base configuration class with common settings."""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key')
    DEBUG = False
    
    # Application paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    EXPORT_FOLDER = os.path.join(PROJECT_ROOT, 'exports')
    MODEL_ARCHITECTURE_FILE = os.path.join(PROJECT_ROOT, 'saved_model.json')
    TRAINED_MODEL_PATH = os.path.join(EXPORT_FOLDER, 'trained_model.keras')
    TRAINING_CONFIG_FILE = os.path.join(PROJECT_ROOT, 'training_config.json')
    
    # SocketIO settings
    SOCKETIO_PING_TIMEOUT = 300
    SOCKETIO_PING_INTERVAL = 25
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    
    # Session settings (default)
    SESSION_PERMANENT = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Session cleanup settings
    SESSION_CLEANUP_INTERVAL_HOURS = int(os.environ.get('SESSION_CLEANUP_INTERVAL', 6))
    SESSION_MAX_AGE_HOURS = int(os.environ.get('SESSION_MAX_AGE', 24))
    
    # Common allowed origins
    ALLOWED_ORIGINS = [
        'http://localhost:5173',
        'http://localhost:3000',
        'https://app.neuroblock.co',
        'https://api.neuroblock.co',
        'https://neuroblock.co'
    ]

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    
    # Development session settings
    SESSION_COOKIE_SECURE = False  # Allow HTTP in development
    SESSION_COOKIE_DOMAIN = None  # No domain restriction in development
    
    # Development server settings
    PORT = int(os.environ.get('PORT', 5000))
    HOST = os.environ.get('HOST', 'localhost')
    
    # Development allowed origins
    ALLOWED_ORIGINS = [
        'http://localhost:5173',
        'http://localhost:3000',
        'http://127.0.0.1:5173',
        'http://127.0.0.1:3000'
    ]

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = 'DEBUG'
    
    # Use test directories
    EXPORT_FOLDER = os.path.join(Config.PROJECT_ROOT, 'test_exports')
    MODEL_ARCHITECTURE_FILE = os.path.join(Config.PROJECT_ROOT, 'test_saved_model.json')
    
    # Testing session settings
    SESSION_COOKIE_SECURE = False
    SESSION_CLEANUP_INTERVAL_HOURS = 1  # Faster cleanup for tests
    SESSION_MAX_AGE_HOURS = 2

class ProductionConfig(Config):
    """Production configuration for GCP deployment."""
    # Security settings
    SECRET_KEY = os.environ.get('SECRET_KEY')  # Required from Secret Manager
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Production session settings
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)  # 7-day sessions
    # HTTPS-only cookies by default; set SESSION_COOKIE_SECURE=false when
    # serving plain HTTP (e.g. classroom EC2 on a bare IP without TLS)
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'true').lower() != 'false'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    SESSION_COOKIE_DOMAIN = None  # Allow cross-domain cookies
    
    # GCP-optimized session cleanup
    SESSION_CLEANUP_INTERVAL_HOURS = int(os.environ.get('SESSION_CLEANUP_INTERVAL', 12))
    SESSION_MAX_AGE_HOURS = int(os.environ.get('SESSION_MAX_AGE', 168))  # 7 days
    
    # GCP-specific settings
    GCP_PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT')
    GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')
    
    # Cloud Run settings
    PORT = int(os.environ.get('PORT', 8080))
    HOST = os.environ.get('HOST', '0.0.0.0')
    
    # Production allowed origins (EXTRA_ALLOWED_ORIGINS env var is appended
    # for all configs in create_app — see backend/main.py).
    ALLOWED_ORIGINS = [
        'https://dnd-neural-frontend-76136455379.us-central1.run.app',
        'https://dnd-neural-backend-76136455379.us-central1.run.app',
        'https://app.neuroblock.co',
        'https://api.neuroblock.co',
        'https://neuroblock.co'
    ]

class CustomDomainConfig(ProductionConfig):
    """Configuration for custom domain deployment."""
    # Override session domain for custom domain
    SESSION_COOKIE_DOMAIN = '.neuroblock.co'  # Allow subdomain cookies
    
    # Custom domain allowed origins
    ALLOWED_ORIGINS = [
        'https://app.neuroblock.co',
        'https://api.neuroblock.co',
        'https://neuroblock.co',
        'https://*.neuroblock.co'  # Allow all subdomains
    ]

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'custom_domain': CustomDomainConfig,
    'default': DevelopmentConfig
}

# Function to get the appropriate config based on environment
def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    if env == 'production':
        # Check if we're using custom domain
        host = os.environ.get('HOST', '')
        if 'neuroblock.co' in host:
            return config['custom_domain']
    return config.get(env, config['default']) 