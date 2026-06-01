"""
ASGI config for core project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

from .env import load_environment

load_environment()
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.core.settings')

from base.utils.logging_config import configure_logging
configure_logging(debug=os.environ.get('DEBUG', 'False').lower() == 'true')

application = get_asgi_application()







