import logging

logger = logging.getLogger(__name__)

class CsrfLoggingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        origin = request.META.get('HTTP_ORIGIN', 'No Origin')
        logger.info(f"Request Origin: {origin}, Path: {request.path}, Method: {request.method}")
        response = self.get_response(request)
        return response






