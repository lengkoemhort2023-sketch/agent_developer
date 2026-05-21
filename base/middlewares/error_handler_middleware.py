from django.http import JsonResponse
from .url_pattern_middleware import URLPatternMiddlewareMixin

class CustomErrorHandlerMiddleware(URLPatternMiddlewareMixin):
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not self.should_process(request):
            return self.get_response(request)

        response = self.get_response(request)
        return self.process_response(request, response)

    def process_response(self, request, response):
        if response.status_code == 403:
            return self.handle_forbidden(request)
        elif response.status_code == 401:
            return self.handle_unauthorized(request, response)
        elif response.status_code == 500:
            return self.handle_server_error(request, response)
        else:
            return response

    def handle_forbidden(self, request):
        return self.error_response(message="Forbidden", status_code=403)

    def handle_unauthorized(self, request, response):
        return self.error_response(message="Unauthorized", status_code=401)

    def handle_server_error(self, request, response):
        return self.error_response(message="Server Error", status_code=500)

    def error_response(self, message, data=None, status_code=500):
        response_data = {
            "status": "error",
            "status_code": status_code,
            "message": message,
            "data": data,
        }
        return JsonResponse(response_data, status=status_code)







