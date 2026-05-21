"""
WSGI config for core project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os
import logging

from django.core.wsgi import get_wsgi_application

from .env import load_environment

load_environment()
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.core.settings')

# Initialize logging
from base.utils.logging_config import configure_logging
from base.utils.security import EnvironmentValidator

# Configure logging and validate environment
try:
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    configure_logging(debug=debug_mode)
    EnvironmentValidator.validate()
    EnvironmentValidator.log_configuration()
except ValueError as e:
    logger = logging.getLogger(__name__)
    logger.critical(f"Configuration error: {e}")
    raise

application = get_wsgi_application()
logger = logging.getLogger(__name__)
logger.info("✓ WSGI application initialized successfully")








