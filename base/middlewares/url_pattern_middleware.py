class URLPatternMiddlewareMixin:
    def should_process(self, request):
        excluded_patterns = ['admin'] 
        return not any(pattern in request.path for pattern in excluded_patterns) 







