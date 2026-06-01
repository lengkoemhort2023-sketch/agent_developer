"""
Structured logging configuration for Django
Provides consistent logging across all modules
"""

import logging
import logging.config
import json
import sys
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for easy parsing and aggregation"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Inject OTel trace/span IDs so Grafana can correlate logs ↔ traces
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            ctx = span.get_span_context()
            if ctx and ctx.is_valid:
                log_data["trace_id"] = format(ctx.trace_id, "032x")
                log_data["span_id"] = format(ctx.span_id, "016x")
        except Exception:
            pass

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Include extra fields
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        if hasattr(record, 'user_id'):
            log_data["user_id"] = record.user_id
        if hasattr(record, 'extra'):
            log_data["extra"] = record.extra

        return json.dumps(log_data)


class ReadableFormatter(logging.Formatter):
    """Format logs in human-readable format for development"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record in readable format"""
        # Add colors if terminal supports it
        level_color = self.COLORS.get(record.levelname, '')
        reset_color = self.COLORS['RESET']
        
        # Basic format
        formatted = (
            f"{level_color}[{record.levelname:8}]{reset_color} "
            f"{record.name}:{record.lineno} - {record.getMessage()}"
        )
        
        # Add exception if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


def configure_logging(debug: bool = False) -> None:
    """
    Configure Django logging with structured format
    
    Args:
        debug: If True, use readable format for development.
               If False, use JSON format for production.
    """
    
    # Choose formatter based on environment
    if debug:
        formatter = ReadableFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        log_format = 'readable'
    else:
        formatter = JSONFormatter()
        log_format = 'json'
    
    # Configure logging
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            # Console: colourised readable text (ANSI OK — goes to terminal)
            'console': {
                '()': ReadableFormatter if debug else JSONFormatter,
            },
            # File: clean text without ANSI — Promtail reads this
            'file': {
                'format': '[%(levelname)-8s] %(name)s:%(lineno)d - %(message)s',
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'stream': sys.stdout,
                'formatter': 'console',
                'level': 'DEBUG' if debug else 'INFO',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'logs/app.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10,
                'formatter': 'file',
                'level': 'INFO',
                'encoding': 'utf-8',
            },
            'security': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': 'logs/security.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 30,
                'formatter': 'file',
                'level': 'WARNING',
                'encoding': 'utf-8',
            },
        },
        'loggers': {
            # Root logger
            '': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG' if debug else 'INFO',
                'propagate': True,
            },
            # Security logger (goes to security.log)
            'security': {
                'handlers': ['console', 'security'],
                'level': 'INFO',
                'propagate': False,
            },
            # Chat logger
            'chat': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG' if debug else 'INFO',
                'propagate': False,
            },
            # Document logger
            'document': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG' if debug else 'INFO',
                'propagate': False,
            },
            # Base services logger (RAG pipeline, generative service, etc.)
            'base': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG' if debug else 'INFO',
                'propagate': False,
            },
            # User logger
            'user': {
                'handlers': ['console', 'security'],
                'level': 'DEBUG' if debug else 'INFO',
                'propagate': False,
            },
            # Django loggers
            'django': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': False,
            },
            'django.security': {
                'handlers': ['security'],
                'level': 'WARNING',
                'propagate': False,
            },
            'django.db': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG' if debug else 'WARNING',
                'propagate': False,
            },
            'django.request': {
                'handlers': ['console', 'file'],
                'level': 'ERROR',
                'propagate': False,
            },
            # HTTP request lines from `runserver` (e.g. "GET /api/v1/ 200")
            'django.server': {
                'handlers': ['console', 'file'],
                'level': 'INFO',
                'propagate': False,
            },
        },
    }
    
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('logs', exist_ok=True)
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with {log_format} formatter")


# Convenience logger function
def get_logger(name: str) -> logging.Logger:
    """Get configured logger for module"""
    return logging.getLogger(name)
