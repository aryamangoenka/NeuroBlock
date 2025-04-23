from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import os
import eventlet

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
        ping_interval=app.config.get("SOCKETIO_PING_INTERVAL", 25)
    )
    
    # Register socket events
    from backend.api.sockets import register_socket_events
    register_socket_events(socketio)
    
    return socketio

# Create the application and socketio instances
app = create_app(os.environ.get("FLASK_CONFIG", "default"))
socketio = create_socketio(app)

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    logger.info("Starting server", extra={"context": {"host": host, "port": port, "debug": debug}})
    socketio.run(app, host=host, port=port, debug=debug)
