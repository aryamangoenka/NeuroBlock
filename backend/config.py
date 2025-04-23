import os

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
    
    # SocketIO settings
    SOCKETIO_PING_TIMEOUT = 300
    SOCKETIO_PING_INTERVAL = 25
    
    # Logging settings
    LOG_LEVEL = 'INFO'

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = 'DEBUG'
    
    # Use test directories
    EXPORT_FOLDER = os.path.join(Config.PROJECT_ROOT, 'test_exports')
    MODEL_ARCHITECTURE_FILE = os.path.join(Config.PROJECT_ROOT, 'test_saved_model.json')

class ProductionConfig(Config):
    """Production configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY')  # Should be set in environment variables
    LOG_LEVEL = 'WARNING'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 