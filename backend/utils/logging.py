import logging
import json
import os
import uuid

class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Generate trace_id if not present
        if not hasattr(record, 'trace_id'):
            record.trace_id = str(uuid.uuid4())
            
        # Ensure context exists
        if not hasattr(record, 'context'):
            record.context = {}
        
        log_record = {
            'timestamp': self.formatTime(record),
            'severity': record.levelname,
            'message': record.getMessage(),
            'service': 'dnd-neural-network',
            'module': record.module,
            'function': record.funcName,
            'filename': record.filename,
            'lineno': record.lineno,
            'trace_id': record.trace_id,
            'context': record.context
        }
        return json.dumps(log_record)

def get_logger(name):
    """
    Get a configured logger with the specified name.
    
    Args:
        name: Name of the logger (typically module name)
        
    Returns:
        Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    
    # Set log level from environment variable or default to INFO
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    return logger 