import logging
import sys
from pathlib import Path

# Global flag to track if logging has been initialized
_logging_initialized = False
_global_log_level = logging.INFO
_log_to_stdout = True  # updated by setup_logging
_log_file = None       # updated by setup_logging


def _create_formatter():
    return logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def _has_stdout_handler(logger):
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout:
            return True
    return False


def _has_file_handler(logger, log_file):
    if not log_file:
        return False
    target = Path(log_file).resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if Path(handler.baseFilename).resolve() == target:
                    return True
            except (OSError, RuntimeError):
                continue
    return False


def _add_handlers(logger, formatter, log_level, log_to_stdout=True, log_file=None):
    if log_to_stdout and not _has_stdout_handler(logger):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file and not _has_file_handler(logger, log_file):
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(log_level)
        handler.setFormatter(formatter)

def setup_logging(log_level=logging.INFO, log_file=None, log_to_stdout=True, module_name=None, reconfigure_existing=True):
    """
    Set up consistent logging configuration across all metatlas2 modules.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file
        log_to_stdout: If True, write to stdout; if False, write only to log_file
        module_name: Name of the calling module (for logger naming)
        reconfigure_existing: If True, clear/rebuild handlers. If False, preserve
            existing handlers and only add missing requested handlers.
    
    Returns:
        logger: Configured logger instance
    """
    global _logging_initialized, _global_log_level, _log_to_stdout, _log_file
    _global_log_level = log_level
    _log_to_stdout = log_to_stdout
    _log_file = log_file
    
    # Create logger name
    if module_name:
        logger_name = f"metatlas2.{module_name}"
    else:
        logger_name = "metatlas2"
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Prevent propagation to avoid duplicate messages
    logger.propagate = False

    formatter = _create_formatter()

    if reconfigure_existing:
        logger.handlers.clear()

    _add_handlers(logger, formatter, log_level, log_to_stdout, log_file)

    if log_file:
        logger.info(f"Logging to file: {Path(log_file)}")

    # Configure all existing metatlas2 loggers
    configure_existing_loggers(
        formatter,
        log_level,
        log_to_stdout,
        log_file,
        reset_handlers=reconfigure_existing,
    )
    
    _logging_initialized = True
    return logger

def configure_existing_loggers(formatter, log_level, log_to_stdout=True, log_file=None, reset_handlers=True):
    """Configure any existing metatlas2 loggers that were created before setup_logging was called."""
    
    # Get all existing loggers
    existing_loggers = [name for name in logging.Logger.manager.loggerDict 
                       if name.startswith('metatlas2.')]
    
    for logger_name in existing_loggers:
        existing_logger = logging.getLogger(logger_name)
        existing_logger.setLevel(log_level)
        existing_logger.propagate = False  # Prevent propagation to avoid duplicates

        if reset_handlers:
            existing_logger.handlers.clear()

        _add_handlers(existing_logger, formatter, log_level, log_to_stdout, log_file)

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
        formatter = _create_formatter()
        _add_handlers(logger, formatter, log_level, _log_to_stdout, _log_file)
        logger.propagate = False
    
    return logger
