"""
API Metrics and Tracing Middleware

Captures request metrics and creates traces for API calls.
"""

import re
import time
import logging
from typing import Callable
from django.utils.deprecation import MiddlewareMixin
from django.http import HttpRequest, HttpResponse
from opentelemetry import trace
from app.core.observability import observability, set_request_context, clear_request_context

_UUID_RE = re.compile(r'/[a-f0-9-]{36}')
_ID_RE = re.compile(r'/\d+')

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class APIMetricsMiddleware(MiddlewareMixin):
    """
    Middleware to track API request metrics.
    
    Tracks:
    - Request count by method and endpoint
    - Request latency (p95, p99)
    - Error rates
    """
    
    # Endpoints to exclude from metrics
    EXCLUDED_PATHS = {"/health/", "/metrics/", "/admin/"}
    
    def __init__(self, get_response: Callable):
        self.get_response = get_response
        super().__init__(get_response)
    
    def process_request(self, request: HttpRequest):
        """Start timing the request"""
        request._start_time = time.time()

        # Extract endpoint
        path = request.path
        request._endpoint = self._normalize_path(path)

        if path not in self.EXCLUDED_PATHS:
            metrics = observability.metrics
            if metrics:
                try:
                    metrics["active_requests"].inc()
                except Exception:
                    pass

        return None
    
    def process_response(self, request: HttpRequest, response: HttpResponse) -> HttpResponse:
        """Record metrics after response"""
        
        # Skip excluded paths
        if request.path in self.EXCLUDED_PATHS:
            return response
        
        # Calculate latency
        if hasattr(request, "_start_time"):
            duration = time.time() - request._start_time
            endpoint = getattr(request, "_endpoint", request.path)
            method = request.method
            status = response.status_code
            
            metrics = observability.metrics
            if metrics:
                try:
                    # Record request count
                    metrics["api_requests"].labels(
                        method=method,
                        endpoint=endpoint,
                        status=status
                    ).inc()
                    
                    # Record latency
                    metrics["api_latency"].labels(
                        method=method,
                        endpoint=endpoint
                    ).observe(duration)
                    
                    # Track user activity
                    metrics["user_requests"].labels(
                        authenticated=str(request.user.is_authenticated)
                    ).inc()
                    
                    # Record errors
                    if status >= 400:
                        error_type = "client_error" if status < 500 else "server_error"
                        metrics["api_errors"].labels(
                            method=method,
                            endpoint=endpoint,
                            error_type=error_type
                        ).inc()
                    
                    # Structured logging
                    logger.info(
                        f"API Request: {method} {endpoint}",
                        extra={
                            "method": method,
                            "endpoint": endpoint,
                            "status": status,
                            "duration_ms": duration * 1000,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error recording metrics: {e}")

            metrics = observability.metrics
            if metrics:
                try:
                    metrics["active_requests"].dec()
                except Exception:
                    pass

        return response

    def process_exception(self, request: HttpRequest, exception: Exception):
        """Record exception metrics"""
        endpoint = getattr(request, "_endpoint", request.path)
        method = request.method

        metrics = observability.metrics
        if metrics:
            try:
                metrics["api_errors"].labels(
                    method=method,
                    endpoint=endpoint,
                    error_type="exception"
                ).inc()
                metrics["active_requests"].dec()
            except Exception:
                pass

        logger.error(
            f"API Exception: {method} {endpoint}",
            exc_info=True,
            extra={
                "method": method,
                "endpoint": endpoint,
            }
        )

        return None
    
    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path to reduce cardinality (e.g., /api/chats/123 -> /api/chats/:id)"""
        path = _UUID_RE.sub('/:id', path)
        return _ID_RE.sub('/:id', path)


class RequestIDMiddleware(MiddlewareMixin):
    """Add request ID for distributed tracing"""
    
    def process_request(self, request: HttpRequest):
        """Extract or generate request ID, link to active OTel span, and set thread-local context."""
        import uuid

        request_id = request.META.get("HTTP_X_REQUEST_ID") or str(uuid.uuid4())
        request.request_id = request_id

        # Correlate request ID with the active trace span
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("http.request_id", request_id)

        # Populate thread-local so every log record on this thread carries request_id/user_id
        try:
            user_id = str(request.user.pk) if request.user.is_authenticated else None
        except Exception:
            user_id = None
        set_request_context(request_id=request_id, user_id=user_id)
        return None

    def process_response(self, request: HttpRequest, response: HttpResponse) -> HttpResponse:
        """Add request ID to response and clear thread-local context."""
        if hasattr(request, "request_id"):
            response["X-Request-ID"] = request.request_id
        clear_request_context()
        return response
