"""
Custom exceptions and exception handlers for the API
Provides consistent error reporting and logging
"""

import logging
from typing import Optional, Dict, Any
from rest_framework.exceptions import APIException
from rest_framework import status

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class APIBaseException(APIException):
    """Base exception for all API errors"""
    
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = "An error occurred"
    error_code = "INTERNAL_ERROR"
    
    def __init__(
        self,
        detail: Optional[str] = None,
        error_code: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs: Any
    ):
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code
        
        super().__init__(detail or self.default_detail)
        self.extra_data = kwargs


class ValidationError(APIBaseException):
    """Validation failed"""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = "Validation error"
    error_code = "VALIDATION_ERROR"


class NotFoundError(APIBaseException):
    """Resource not found"""
    status_code = status.HTTP_404_NOT_FOUND
    default_detail = "Resource not found"
    error_code = "NOT_FOUND"


class PermissionDeniedError(APIBaseException):
    """User does not have permission"""
    status_code = status.HTTP_403_FORBIDDEN
    default_detail = "Permission denied"
    error_code = "PERMISSION_DENIED"


class UnauthorizedError(APIBaseException):
    """Authentication required"""
    status_code = status.HTTP_401_UNAUTHORIZED
    default_detail = "Unauthorized"
    error_code = "UNAUTHORIZED"


class ConflictError(APIBaseException):
    """Resource conflict (e.g., duplicate)"""
    status_code = status.HTTP_409_CONFLICT
    default_detail = "Resource conflict"
    error_code = "CONFLICT"


class RateLimitError(APIBaseException):
    """Rate limit exceeded"""
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    default_detail = "Rate limit exceeded"
    error_code = "RATE_LIMIT_EXCEEDED"


class BadRequestError(APIBaseException):
    """Bad request"""
    status_code = status.HTTP_400_BAD_REQUEST
    default_detail = "Bad request"
    error_code = "BAD_REQUEST"


class InternalServerError(APIBaseException):
    """Internal server error"""
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_detail = "Internal server error"
    error_code = "INTERNAL_ERROR"


class ServiceUnavailableError(APIBaseException):
    """Service temporarily unavailable"""
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_detail = "Service unavailable"
    error_code = "SERVICE_UNAVAILABLE"


# ============================================================================
# SPECIFIC DOMAIN EXCEPTIONS
# ============================================================================

class ChatError(APIBaseException):
    """Error in chat operations"""
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "CHAT_ERROR"


class DocumentError(APIBaseException):
    """Error in document operations"""
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "DOCUMENT_ERROR"


class DocumentNotFoundError(NotFoundError):
    """Document not found"""
    default_detail = "Document not found"
    error_code = "DOCUMENT_NOT_FOUND"


class InvalidDocumentError(ValidationError):
    """Invalid document (format, size, etc)"""
    default_detail = "Invalid document"
    error_code = "INVALID_DOCUMENT"


class LDAPAuthenticationError(UnauthorizedError):
    """LDAP authentication failed"""
    default_detail = "LDAP authentication failed"
    error_code = "LDAP_AUTH_FAILED"


# ============================================================================
# EXCEPTION HANDLER
# ============================================================================

def custom_exception_handler(exc: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Custom exception handler for DRF
    Logs exceptions and returns standardized format
    """
    from rest_framework.views import exception_handler as drf_exception_handler
    
    # Get DRF response
    response = drf_exception_handler(exc, context)
    
    # Log the exception
    request = context.get('request')
    view = context.get('view')
    
    if isinstance(exc, APIBaseException):
        log_level = logging.WARNING if 400 <= exc.status_code < 500 else logging.ERROR
        logger.log(
            log_level,
            f"{exc.error_code}: {exc.detail}",
            extra={
                "error_code": exc.error_code,
                "status_code": exc.status_code,
                "view": view.__class__.__name__ if view else None,
                "method": request.method if request else None,
                "path": request.path if request else None,
                "extra_data": exc.extra_data,
            }
        )
    else:
        logger.exception(
            f"Unhandled exception: {type(exc).__name__}",
            extra={
                "view": view.__class__.__name__ if view else None,
                "method": request.method if request else None,
                "path": request.path if request else None,
            }
        )
    
    # Format response
    if response is None:
        response = {
            "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "message": str(exc),
            "error_code": getattr(exc, 'error_code', 'INTERNAL_ERROR')
        }
    else:
        response.data = {
            "status": response.status_code,
            "message": response.data.get('detail', str(exc)),
            "error_code": getattr(exc, 'error_code', 'UNKNOWN_ERROR'),
            "details": response.data.get('detail') if isinstance(response.data, dict) else None,
        }
    
    return response


# ============================================================================
# EXCEPTION DECORATOR
# ============================================================================

def handle_exceptions(view_func):
    """
    Decorator to catch exceptions in views and return proper responses
    
    Usage:
        @handle_exceptions
        def my_view(request):
            ...
    """
    from functools import wraps
    
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        try:
            return view_func(*args, **kwargs)
        except APIBaseException:
            raise  # Let DRF handle it
        except Exception as e:
            logger.exception(f"Unhandled exception in {view_func.__name__}")
            raise InternalServerError(f"An error occurred: {str(e)}")
    
    return wrapper
