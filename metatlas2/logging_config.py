import logging
import sys
from pathlib import Path

# Global flag to track if logging has been initialized
_logging_initialized = False
_global_log_level = logging.INFO

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
    global _logging_initialized, _global_log_level
    _global_log_level = log_level
    
    # Create logger name
    if module_name:
        logger_name = f"metatlas2.{module_name}"
    else:
        logger_name = "metatlas2"
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
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
    configure_existing_loggers(formatter, log_level)
    
    _logging_initialized = True
    return logger

def configure_existing_loggers(formatter, log_level):
    """Configure any existing metatlas2 loggers that were created before setup_logging was called."""
    
    # Get all existing loggers
    existing_loggers = [name for name in logging.Logger.manager.loggerDict 
                       if name.startswith('metatlas2.')]
    
    for logger_name in existing_loggers:
        existing_logger = logging.getLogger(logger_name)
        existing_logger.handlers.clear()  # Remove any existing handlers
        existing_logger.setLevel(log_level)
        existing_logger.propagate = False  # Prevent propagation to avoid duplicates
        
        # Create a new console handler with the same formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        existing_logger.addHandler(handler)

def ensure_logging_initialized():
    """Ensure logging is initialized with default settings if not already done."""
    global _logging_initialized
    if not _logging_initialized:
        setup_logging()

def get_logger(module_name, log_level=None):
    """Get or update a logger for a specific module."""
    global _global_log_level
    
    if log_level is None:
        log_level = _global_log_level
        
    ensure_logging_initialized()
    logger = logging.getLogger(f"metatlas2.{module_name}")
    logger.setLevel(log_level)
    
    # ALWAYS update handlers to current level
    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(log_level)
    else:
        # Create handlers if none exist
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger
