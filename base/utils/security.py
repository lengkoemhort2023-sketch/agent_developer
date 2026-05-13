"""
Security utilities for validating and protecting sensitive configuration
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_required_env(key: str, description: str = "") -> str:
    """
    Get required environment variable.
    Raises error if not set (prevents using hardcoded defaults in production).
    
    Args:
        key: Environment variable name
        description: Description of what this variable is for
        
    Returns:
        Environment variable value
        
    Raises:
        ValueError: If variable not set in production
    """
    value = os.environ.get(key)
    
    # Allow missing values only in development
    if not value:
        is_prod = os.environ.get('DEBUG', 'False').lower() != 'true'
        if is_prod:
            raise ValueError(
                f"Required environment variable '{key}' not set. "
                f"This is required for production. {description}"
            )
        logger.warning(
            f"Environment variable '{key}' not set. "
            f"This should be configured in production. {description}"
        )
        return ""
    
    return value


def get_optional_env(key: str, default: Optional[str] = None) -> str:
    """
    Get optional environment variable with default fallback.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Environment variable value or default
    """
    return os.environ.get(key, default or "")


class EnvironmentValidator:
    """Validate all required environment variables on startup"""
    
    REQUIRED_VARS = {
        'SECRET_KEY': 'Django secret key for sessions and crypto',
        'AUTH_LDAP_BIND_PASSWORD': 'LDAP bind password (required for LDAP auth)',
    }
    
    OPTIONAL_VARS = {
        'DEBUG': 'Debug mode (default: False)',
        'DJANGO_ALLOWED_HOSTS': 'Comma-separated list of allowed hosts',
        'DATABASE_URL': 'Database connection string',
        'CELERY_BROKER_URL': 'Celery broker URL',
    }
    
    @classmethod
    def validate(cls) -> None:
        """Validate all required environment variables are set"""
        is_prod = os.environ.get('DEBUG', 'False').lower() != 'true'
        missing = []
        
        for var, description in cls.REQUIRED_VARS.items():
            if not os.environ.get(var):
                if is_prod:
                    missing.append(f"  • {var}: {description}")
                else:
                    logger.warning(f"Missing optional: {var} - {description}")
        
        if missing:
            raise ValueError(
                f"Missing required environment variables in production:\n" +
                "\n".join(missing) +
                "\n\nPlease set these in your .env file or environment."
            )
        
        logger.info("✓ All required environment variables validated")
    
    @classmethod
    def log_configuration(cls) -> None:
        """Log configuration status (without exposing secrets)"""
        logger.info("Configuration loaded:")
        
        for var in cls.REQUIRED_VARS.keys():
            is_set = bool(os.environ.get(var))
            logger.info(f"  • {var}: {'✓ Set' if is_set else '✗ Missing'}")
        
        for var, default in cls.OPTIONAL_VARS.items():
            value = os.environ.get(var)
            if value:
                logger.info(f"  • {var}: ✓ Set (length: {len(value)})")
            else:
                logger.info(f"  • {var}: Using default")
