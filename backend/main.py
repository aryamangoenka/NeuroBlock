from flask import Flask, redirect
from flask_cors import CORS
from flask_socketio import SocketIO
import os


from backend.config import config
from backend.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

def create_app(config_name="default"):
    """
    Application factory function that creates and configures the Flask app.
    
    Args:
        config_name: The configuration to use (default, development, production, testing)
    
    Returns:
        The configured Flask application
    """
    # Create the Flask application
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Enable CORS
    CORS(app)
    
    # Ensure export folder exists
    export_folder = app.config.get("EXPORT_FOLDER", os.path.join(os.path.dirname(os.path.dirname(__file__)), "exports"))
    os.makedirs(export_folder, exist_ok=True)
    
    logger.info("Application initialized", extra={"context": {"export_folder": export_folder}})
    
    # Register blueprints
    from backend.api.routes import api_blueprint
    app.register_blueprint(api_blueprint)
    
    # Add health check endpoint for Google Cloud
    @app.route('/health')
    def health():
        return {'status': 'healthy'}, 200
    
    # Add root route handler
    @app.route('/')
    def root():
        return {'message': 'DND Neural Network Backend API. Please use /api/ for endpoints.'}, 200
    
    return app

def create_socketio(app):
    """
    Create and configure the SocketIO instance.
    
    Args:
        app: The Flask application
    
    Returns:
        Configured SocketIO instance
    """
    socketio = SocketIO(
        app, 
        cors_allowed_origins="*", 
        ping_timeout=app.config.get("SOCKETIO_PING_TIMEOUT", 300),
        ping_interval=app.config.get("SOCKETIO_PING_INTERVAL", 25),
        async_mode="threading"
    )
    
    # Register socket events
    from backend.api.sockets import register_socket_events
    register_socket_events(socketio)
    
    return socketio

# Create the application and socketio instances
app = create_app(os.environ.get("FLASK_CONFIG", "default"))
socketio = create_socketio(app)

# For Cloud Run or WSGI
# This line exposes the Flask application for Cloud Run and similar platforms
wsgi_app = app

if __name__ == "__main__":
    host = "0.0.0.0"  # Always bind to all interfaces for Cloud Run
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    logger.info("Starting server", extra={"context": {"host": host, "port": port, "debug": debug}})
    socketio.run(app, host=host, port=port, debug=debug)
