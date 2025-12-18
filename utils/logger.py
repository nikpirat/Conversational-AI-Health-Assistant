"""
Structured logging configuration with color output and JSON formatting.
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import colorlog
from pythonjsonlogger import jsonlogger

from config.settings import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional context fields."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        if hasattr(record, 'conversation_id'):
            log_record['conversation_id'] = record.conversation_id


def setup_logger(
        name: str,
        log_level: Optional[str] = None,
        log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up a logger with both console (colored) and file (JSON) handlers.

    Args:
        name: Logger name (typically __name__)
        log_level: Override default log level
        log_file: Optional file path for JSON logs

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level or settings.log_level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with colors
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    console_format = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler with JSON (if log_file specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        json_format = CustomJsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        file_handler.setFormatter(json_format)
        logger.addHandler(file_handler)

    return logger


def log_api_call(
        logger: logging.Logger,
        service: str,
        endpoint: str,
        duration_ms: float,
        status: str = "success",
        **kwargs
):
    """
    Log API call with structured data.

    Args:
        logger: Logger instance
        service: Service name (e.g., 'anthropic', 'openai')
        endpoint: API endpoint called
        duration_ms: Call duration in milliseconds
        status: Call status (success/error)
        **kwargs: Additional context to log
    """
    logger.info(
        f"API call: {service}.{endpoint}",
        extra={
            'service': service,
            'endpoint': endpoint,
            'duration_ms': round(duration_ms, 2),
            'status': status,
            **kwargs
        }
    )


async def log_conversation_turn(
        logger: logging.Logger,
        conversation_id: str,
        turn_number: int,
        user_message: str,
        assistant_message: str,
        language: str,
        duration_ms: float
):
    """
    Log a complete conversation turn.

    Args:
        logger: Logger instance
        conversation_id: Unique conversation identifier
        turn_number: Turn number in conversation
        user_message: User's input
        assistant_message: Assistant's response
        language: Detected language
        duration_ms: Processing duration
    """
    logger.info(
        f"Conversation turn {turn_number}",
        extra={
            'conversation_id': conversation_id,
            'turn_number': turn_number,
            'user_message_length': len(user_message),
            'assistant_message_length': len(assistant_message),
            'language': language,
            'duration_ms': round(duration_ms, 2)
        }
    )


# Pre-configured loggers for main components
main_logger = setup_logger("main", log_file=Path("logs/app.log"))
api_logger = setup_logger("api", log_file=Path("logs/api.log"))
db_logger = setup_logger("database", log_file=Path("logs/database.log"))