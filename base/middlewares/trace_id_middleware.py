import uuid
import time
from .url_pattern_middleware import URLPatternMiddlewareMixin

class TraceIDMiddleware(URLPatternMiddlewareMixin):
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not self.should_process(request):
            return self.get_response(request)

        response = self.get_response(request)

        trace_id = f"{int(time.time())}-{uuid.uuid4()}"

        response["X-Trace-ID"] = trace_id

        return response 







