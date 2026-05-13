"""
Rate Limiting Middleware for RAG API

Prevents request spam and implements simple rate limiting.
"""

from django.core.cache import cache
from django.http import JsonResponse
import time
import logging

logger = logging.getLogger(__name__)


class RateLimitMiddleware:
    """
    Simple rate limiting middleware.
    Limits requests per user per time window.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        # Configuration
        self.max_requests = 10  # Max requests per window
        self.time_window = 60  # Time window in seconds (1 minute)
    
    def __call__(self, request):
        # Only apply rate limiting to chat message creation endpoint
        if request.path.startswith('/api/chat/messages') and request.method == 'POST':
            # Get user identifier (user ID or IP address)
            if hasattr(request, 'user') and request.user.is_authenticated:
                user_id = f"rate_limit_user_{request.user.id}"
            else:
                # Fallback to IP address for unauthenticated requests
                user_id = f"rate_limit_ip_{self._get_client_ip(request)}"
            
            # Check rate limit
            current_time = time.time()
            cache_key = f"{user_id}_{int(current_time // self.time_window)}"
            
            # Get current request count
            request_count = cache.get(cache_key, 0)
            
            if request_count >= self.max_requests:
                logger.warning(f"Rate limit exceeded for {user_id}")
                return JsonResponse({
                    'error': 'Rate limit exceeded. Please wait before making more requests.',
                    'retry_after': int(self.time_window - (current_time % self.time_window))
                }, status=429)
            
            # Increment request count
            cache.set(cache_key, request_count + 1, self.time_window)
        
        response = self.get_response(request)
        return response
    
    def _get_client_ip(self, request):
        """Extract client IP address from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip







