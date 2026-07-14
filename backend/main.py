import eventlet
eventlet.monkey_patch()  # Add this at the very top
from flask import Flask, redirect, session, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
import os
import uuid



# Set matplotlib backend early to prevent GUI threading issues on macOS
import matplotlib
matplotlib.use('Agg')

from backend.config import config
from backend.utils.logging import get_logger
from backend.utils.session_manager import start_cleanup_thread

# Initialize logger
logger = get_logger(__name__)

def create_app(config_name="default"):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Allow extra CORS origins via env (Cloud Run injects the deployed
    # frontend URL here — see cloudbuild.yaml "wire-cors" step)
    _extra_origins = [
        o.strip()
        for o in os.environ.get("EXTRA_ALLOWED_ORIGINS", "").split(",")
        if o.strip()
    ]
    if _extra_origins:
        app.config["ALLOWED_ORIGINS"] = list(app.config["ALLOWED_ORIGINS"]) + _extra_origins

    # Enable CORS with credentials support (important for sessions)
    CORS(app, supports_credentials=True, resources={
        r"/*": {
            "origins": app.config['ALLOWED_ORIGINS'],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })
    
    # Ensure export folder exists
    export_folder = app.config.get("EXPORT_FOLDER", os.path.join(os.path.dirname(os.path.dirname(__file__)), "exports"))
    os.makedirs(export_folder, exist_ok=True)
    
    # Add session initialization middleware
    @app.before_request
    def ensure_session():
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            logger.info(f"New session created: {session['session_id']}")
    
    # Start session cleanup thread
    cleanup_interval = app.config.get('SESSION_CLEANUP_INTERVAL_HOURS', 6)
    max_age = app.config.get('SESSION_MAX_AGE_HOURS', 24)
    start_cleanup_thread(
        cleanup_interval_hours=cleanup_interval, 
        max_age_hours=max_age, 
        app=app
    )
    
    # Register blueprints
    from backend.api.routes import api_blueprint
    from backend.api.dataset_routes import dataset_blueprint
    app.register_blueprint(api_blueprint)
    app.register_blueprint(dataset_blueprint)

    # Serve the built React app when a production build exists
    # (single-container deployment: same origin for app, API, and sockets).
    # In development the Vite dev server handles the frontend instead.
    frontend_dist = os.path.join(
        app.config.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(__file__))),
        "frontend",
        "dist",
    )
    if os.path.isdir(frontend_dist):
        @app.route("/")
        @app.route("/<path:path>")
        def serve_frontend(path=""):
            full_path = os.path.join(frontend_dist, path)
            if path and os.path.isfile(full_path):
                return send_from_directory(frontend_dist, path)
            return send_from_directory(frontend_dist, "index.html")

        logger.info(f"Serving frontend from {frontend_dist}")

    return app

def create_socketio(app):
    # "*" so sockets work on any deployment origin (single-container serving
    # means the browser is always same-origin anyway; the app is public and
    # unauthenticated by design, HTTP routes keep the explicit CORS list).
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        ping_timeout=app.config.get("SOCKETIO_PING_TIMEOUT", 300),
        ping_interval=app.config.get("SOCKETIO_PING_INTERVAL", 25),
        async_mode="eventlet",  # Change to eventlet
        manage_session=False  # Important: let Flask handle sessions
    )
    
    # Register socket events
    from backend.api.sockets import register_socket_events
    register_socket_events(socketio)
    
    return socketio

# Create the application and socketio instances
app = create_app(os.environ.get("FLASK_CONFIG", "default"))
socketio = create_socketio(app)

# Log startup configuration
host = "0.0.0.0"
port = int(os.environ.get("PORT", 8080))
debug = app.config.get('DEBUG', False)

print(f"[ENV CHECK] FLASK_CONFIG={os.environ.get('FLASK_CONFIG', 'default')}, DEBUG={debug}")

logger.info("Starting server", extra={
    "context": {
        "host": host,
        "port": port,
        "debug": debug,
        "config": os.environ.get("FLASK_CONFIG", "default"),
        "allowed_origins": app.config['ALLOWED_ORIGINS']
    }
})

# Only run with socketio.run() in development
if __name__ == "__main__" :
    socketio.run(app, host=host, port=port, debug=debug)