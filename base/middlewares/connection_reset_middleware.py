"""
Middleware to handle connection reset errors during file uploads.
This middleware catches UnreadablePostError and returns a proper error response.
"""

from django.http import JsonResponse
from django.core.exceptions import MiddlewareNotUsed
import logging

logger = logging.getLogger(__name__)

class ConnectionResetMiddleware:
    """
    Middleware to handle connection reset errors during file uploads.
    Catches UnreadablePostError and returns a proper error response.
    """
    
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def process_exception(self, request, exception):
        """
        Process exceptions and handle connection reset errors.
        """
        from django.http.request import UnreadablePostError
        
        # Check if this is a connection reset error during upload
        if isinstance(exception, UnreadablePostError):
            logger.error(f"Connection reset during upload: {exception}")
            
            # Check if this is an upload endpoint
            if request.path.startswith('/api/documents/upload'):
                return JsonResponse({
                    'status': 'error',
                    'status_code': 400,
                    'message': 'Upload failed due to connection issue. Please try again.',
                    'data': {
                        'error': 'Connection reset during upload',
                        'hint': 'This usually happens when the connection is lost during large file upload. Try uploading a smaller file or check your network connection.'
                    }
                }, status=400)
            
            # For other endpoints, log the error but let Django handle it
            logger.warning(f"Connection reset on non-upload endpoint {request.path}: {exception}")
        
        # For other exceptions, let Django handle them normally
        return None

