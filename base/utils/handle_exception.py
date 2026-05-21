import logging
from .api_response import error_response
from django.http import Http404
from rest_framework import status
from rest_framework.exceptions import AuthenticationFailed, NotAuthenticated

logger = logging.getLogger(__name__)

def custom_exception_handler(exc, context):
    """Custom exception handler for DRF"""
    if isinstance(exc, NotAuthenticated):
        return error_response(
            "Authentication credentials were not provided.",
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    elif isinstance(exc, AuthenticationFailed):
        return error_response(
            "Authentication failed.",
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    # Let DRF handle other exceptions
    return None

def handle_exception(func):
    """Decorator to handle common exceptions and prevent information disclosure"""
    def wrapper(request, *args, **kwargs):
        try:
            return func(request, *args, **kwargs)
        except Http404:
            # Don't log or expose details about 404 - could leak system information
            raise
        except AuthenticationFailed:
            logger.warning(f"Authentication failed in {func.__name__}")
            raise
        except Exception as e:
            # Log the full error for debugging, but don't expose to client
            logger.error(
                f"Error in {func.__name__}: {type(e).__name__}",
                exc_info=True
            )
            # Return generic error message to client - never expose internal details
            return error_response(
                "An error occurred processing your request. Please try again or contact support.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    return wrapper







