"""
Session-based dataset management utilities.

This module provides functions to manage datasets on a per-session basis,
ensuring that each user's custom datasets are isolated and automatically cleaned up.
"""

import os
import shutil
import time
import threading
from datetime import datetime, timedelta
from flask import session, current_app
from backend.utils.logging import get_logger

logger = get_logger(__name__)

# Global cleanup thread and app reference
_cleanup_thread = None
_cleanup_running = False
_app_instance = None

def get_session_id():
    """
    Get the current session ID, creating one if it doesn't exist.
    
    Returns:
        str: Session ID
    """
    if 'session_id' not in session:
        import uuid
        session['session_id'] = str(uuid.uuid4())
        logger.info(f"New session created: {session['session_id']}")
    
    return session['session_id']

def get_session_datasets_dir(session_id=None):
    """
    Get the datasets directory for a specific session.
    
    Args:
        session_id (str, optional): Session ID. If None, uses current session.
        
    Returns:
        str: Path to session-specific datasets directory
    """
    if session_id is None:
        session_id = get_session_id()
    
    project_root = current_app.config.get('PROJECT_ROOT', '')
    session_dir = os.path.join(project_root, 'sessions', session_id, 'datasets')
    
    # Ensure directory exists
    os.makedirs(session_dir, exist_ok=True)
    
    # Update access time for cleanup purposes
    touch_file = os.path.join(session_dir, '.last_access')
    with open(touch_file, 'w') as f:
        f.write(str(time.time()))
    
    return session_dir

def cleanup_old_sessions(max_age_hours=24, app=None):
    """
    Clean up session directories older than the specified age.
    
    Args:
        max_age_hours (int): Maximum age in hours before cleanup
        app: Flask app instance (optional, uses stored reference if None)
    """
    try:
        # Use provided app or stored app instance
        if app is None:
            app = _app_instance
        
        if app is None:
            logger.warning("No Flask app instance available for cleanup")
            return
        
        with app.app_context():
            project_root = app.config.get('PROJECT_ROOT', '')
            sessions_dir = os.path.join(project_root, 'sessions')
            
            if not os.path.exists(sessions_dir):
                return
            
            cutoff_time = time.time() - (max_age_hours * 3600)
            cleaned_count = 0
            
            for session_id in os.listdir(sessions_dir):
                session_path = os.path.join(sessions_dir, session_id)
                
                if not os.path.isdir(session_path):
                    continue
                
                # Check last access time
                touch_file = os.path.join(session_path, 'datasets', '.last_access')
                
                try:
                    if os.path.exists(touch_file):
                        with open(touch_file, 'r') as f:
                            last_access = float(f.read().strip())
                    else:
                        # Use directory modification time as fallback
                        last_access = os.path.getmtime(session_path)
                    
                    if last_access < cutoff_time:
                        logger.info(f"Cleaning up old session: {session_id}")
                        shutil.rmtree(session_path)
                        cleaned_count += 1
                        
                except (ValueError, OSError) as e:
                    logger.warning(f"Error checking session {session_id}: {str(e)}")
                    continue
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old session directories")
                
    except Exception as e:
        logger.error(f"Error during session cleanup: {str(e)}", exc_info=True)

def start_cleanup_thread(cleanup_interval_hours=6, max_age_hours=24, app=None):
    """
    Start a background thread to periodically clean up old sessions.
    
    Args:
        cleanup_interval_hours (int): How often to run cleanup (in hours)
        max_age_hours (int): Maximum age before cleanup (in hours)
        app: Flask app instance
    """
    global _cleanup_thread, _cleanup_running, _app_instance
    
    if _cleanup_running:
        return
    
    # Store app instance for cleanup thread
    if app is None:
        app = current_app._get_current_object()
    _app_instance = app
    
    def cleanup_worker():
        global _cleanup_running
        _cleanup_running = True
        
        while _cleanup_running:
            try:
                # Use stored app instance for cleanup
                cleanup_old_sessions(max_age_hours, _app_instance)
                
                # Sleep for the specified interval
                time.sleep(cleanup_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in cleanup thread: {str(e)}", exc_info=True)
                time.sleep(300)  # Sleep 5 minutes on error
    
    _cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    _cleanup_thread.start()
    logger.info(f"Session cleanup thread started (interval: {cleanup_interval_hours}h, max_age: {max_age_hours}h)")

def stop_cleanup_thread():
    """Stop the cleanup thread."""
    global _cleanup_running
    _cleanup_running = False
    logger.info("Session cleanup thread stopped")

def get_session_dataset_count(session_id=None):
    """
    Get the number of datasets in a session.
    
    Args:
        session_id (str, optional): Session ID. If None, uses current session.
        
    Returns:
        int: Number of datasets in the session
    """
    session_dir = get_session_datasets_dir(session_id)
    
    try:
        count = 0
        for filename in os.listdir(session_dir):
            if filename.endswith('_metadata.json'):
                count += 1
        return count
    except OSError:
        return 0

def clear_session_datasets(session_id=None):
    """
    Clear all datasets for a specific session.
    
    Args:
        session_id (str, optional): Session ID. If None, uses current session.
        
    Returns:
        bool: True if successful, False otherwise
    """
    if session_id is None:
        session_id = get_session_id()
    
    try:
        project_root = current_app.config.get('PROJECT_ROOT', '')
        session_path = os.path.join(project_root, 'sessions', session_id)
        
        if os.path.exists(session_path):
            shutil.rmtree(session_path)
            logger.info(f"Cleared all datasets for session: {session_id}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error clearing session datasets: {str(e)}", exc_info=True)
        return False 