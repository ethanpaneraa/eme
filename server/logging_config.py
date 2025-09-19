import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up comprehensive logging configuration for the server.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files (defaults to ./logs)
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        max_file_size: Maximum size of each log file in bytes
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """

    # Create log directory if it doesn't exist
    if log_dir is None:
        log_dir = Path(__file__).parent / "logs"
    else:
        log_dir = Path(log_dir)

    log_dir.mkdir(exist_ok=True)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)

    # File handlers
    if enable_file:
        # Main application log
        app_log_file = log_dir / "app.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        app_handler.setLevel(logging.DEBUG)  # Always debug level for files
        app_handler.setFormatter(detailed_formatter)
        logger.addHandler(app_handler)

        # Error log (only errors and critical)
        error_log_file = log_dir / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)

        # Request log (for API requests)
        request_log_file = log_dir / "requests.log"
        request_handler = logging.handlers.RotatingFileHandler(
            request_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        request_handler.setLevel(logging.INFO)
        request_handler.setFormatter(detailed_formatter)

        # Create a separate logger for requests
        request_logger = logging.getLogger("requests")
        request_logger.setLevel(logging.INFO)
        request_logger.addHandler(request_handler)
        request_logger.propagate = False  # Don't propagate to root logger

    # Log startup information
    logger.info(f"Logging initialized - Level: {log_level.upper()}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Console logging: {enable_console}")
    logger.info(f"File logging: {enable_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_request_info(logger: logging.Logger, method: str, path: str, status_code: int,
                    response_time: float, user_agent: str = None, ip: str = None):
    """
    Log HTTP request information in a structured format.

    Args:
        logger: Logger instance
        method: HTTP method
        path: Request path
        status_code: Response status code
        response_time: Response time in seconds
        user_agent: User agent string
        ip: Client IP address
    """
    request_logger = logging.getLogger("requests")

    log_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "response_time": f"{response_time:.3f}s",
        "timestamp": datetime.utcnow().isoformat()
    }

    if user_agent:
        log_data["user_agent"] = user_agent
    if ip:
        log_data["ip"] = ip

    request_logger.info(f"REQUEST | {log_data}")


def log_bot_interaction(logger: logging.Logger, query: str, response: str,
                       response_time: float, user_id: str = None, group_id: str = None):
    """
    Log bot interactions for debugging and analytics.

    Args:
        logger: Logger instance
        query: User query
        response: Bot response
        response_time: Response time in seconds
        user_id: User ID
        group_id: Group ID
    """
    interaction_data = {
        "query": query[:100] + "..." if len(query) > 100 else query,
        "response_length": len(response),
        "response_time": f"{response_time:.3f}s",
        "timestamp": datetime.utcnow().isoformat()
    }

    if user_id:
        interaction_data["user_id"] = user_id
    if group_id:
        interaction_data["group_id"] = group_id

    logger.info(f"BOT_INTERACTION | {interaction_data}")


def log_rag_operation(logger: logging.Logger, operation: str, query: str = None,
                     results_count: int = None, processing_time: float = None):
    """
    Log RAG pipeline operations for debugging.

    Args:
        logger: Logger instance
        operation: Operation type (retrieve, generate, add_messages)
        query: Search query
        results_count: Number of results returned
        processing_time: Processing time in seconds
    """
    rag_data = {
        "operation": operation,
        "timestamp": datetime.utcnow().isoformat()
    }

    if query:
        rag_data["query"] = query[:50] + "..." if len(query) > 50 else query
    if results_count is not None:
        rag_data["results_count"] = results_count
    if processing_time is not None:
        rag_data["processing_time"] = f"{processing_time:.3f}s"

    logger.info(f"RAG_OPERATION | {rag_data}")


# Environment-based configuration
def setup_logging_from_env():
    """
    Set up logging based on environment variables.

    Environment variables:
        LOG_LEVEL: Logging level (default: INFO)
        LOG_DIR: Log directory (default: ./logs)
        LOG_CONSOLE: Enable console logging (default: true)
        LOG_FILE: Enable file logging (default: true)
    """
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_dir = os.getenv("LOG_DIR")
    enable_console = os.getenv("LOG_CONSOLE", "true").lower() == "true"
    enable_file = os.getenv("LOG_FILE", "true").lower() == "true"

    return setup_logging(
        log_level=log_level,
        log_dir=log_dir,
        enable_console=enable_console,
        enable_file=enable_file
    )
