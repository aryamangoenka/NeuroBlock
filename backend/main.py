from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
import os
import eventlet

from backend.api.routes import register_routes
from backend.api.sockets import register_socket_events
from backend.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Create the Flask application
app = Flask(__name__)
CORS(app)

# Configure the application
app.config["SECRET_KEY"] = "your-secret-key"

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=300, ping_interval=25)

# Register API routes and socket events
register_routes(app)
register_socket_events(socketio)

# Ensure export folder exists
EXPORT_FOLDER = "exports"
os.makedirs(EXPORT_FOLDER, exist_ok=True)

logger.info("Application initialized", extra={"context": {"export_folder": EXPORT_FOLDER}})

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    logger.info("Starting server", extra={"context": {"host": host, "port": port, "debug": debug}})
    socketio.run(app, host=host, port=port, debug=debug)
