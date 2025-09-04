import logging
import sys
from pathlib import Path
from datetime import datetime

# Global flag to track if logging has been initialized
_logging_initialized = False

def setup_logging(log_level=logging.INFO, log_file=None, module_name=None):
    """
    Set up consistent logging configuration across all metatlas2 modules.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file
        module_name: Name of the calling module (for logger naming)
    
    Returns:
        logger: Configured logger instance
    """
    global _logging_initialized
    
    # Create logger name
    if module_name:
        logger_name = f"metatlas2.{module_name}"
    else:
        logger_name = "metatlas2"
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")
    
    # Configure all existing metatlas2 loggers
    configure_existing_loggers(formatter, console_handler, log_level)
    
    _logging_initialized = True
    return logger

def configure_existing_loggers(formatter, console_handler, log_level):
    """Configure any existing metatlas2 loggers that were created before setup_logging was called."""
    
    # Get all existing loggers
    existing_loggers = [name for name in logging.Logger.manager.loggerDict 
                       if name.startswith('metatlas2.')]
    
    for logger_name in existing_loggers:
        existing_logger = logging.getLogger(logger_name)
        existing_logger.handlers.clear()  # Remove any existing handlers
        existing_logger.setLevel(log_level)
        
        # Create a new console handler with the same formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        existing_logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate messages
        existing_logger.propagate = False

def ensure_logging_initialized():
    """Ensure logging is initialized with default settings if not already done."""
    global _logging_initialized
    if not _logging_initialized:
        setup_logging()

def get_logger(module_name):
    """
    Get a logger for a specific module with consistent naming.
    Automatically initializes logging if not already done.
    
    Args:
        module_name: Name of the module (e.g., 'database_interact', 'targeted_analysis')
    
    Returns:
        logger: Logger instance
    """
    # Ensure logging is initialized
    ensure_logging_initialized()
    
    logger = logging.getLogger(f"metatlas2.{module_name}")
    
    # Ensure the logger doesn't propagate to avoid duplicate messages
    logger.propagate = False
    
    return logger
