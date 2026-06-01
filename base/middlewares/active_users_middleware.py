import os
import time

import redis as redis_lib


class ActiveUsersMiddleware:
    """
    Tracks unique authenticated users active in the last 15 minutes.

    Because DRF's JWTAuthentication authenticates at view level (not Django
    middleware level), request.user is always AnonymousUser here. Instead we
    decode the Authorization: Bearer header directly using simplejwt's
    AccessToken — no database lookup required.

    On every request that carries a valid JWT:
      1. Extracts user_id from the token payload.
      2. Adds it to a Redis sorted set scored by current timestamp.
      3. Prunes entries older than WINDOW_SECONDS.
      4. Updates the `active_users_total` Prometheus Gauge with the live count.

    Failures are silently swallowed so the middleware never breaks requests.
    """

    REDIS_KEY = "docbot:active_users"
    WINDOW_SECONDS = 900  # 15 minutes

    def __init__(self, get_response):
        self.get_response = get_response
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            url = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
            self._redis = redis_lib.from_url(url, socket_connect_timeout=1, socket_timeout=1)
        return self._redis

    def _get_user_id(self, request) -> str | None:
        """Decode the JWT Bearer token and return the user_id claim, or None."""
        auth = request.META.get("HTTP_AUTHORIZATION", "")
        if not auth.startswith("Bearer "):
            return None
        try:
            from rest_framework_simplejwt.tokens import AccessToken
            token = AccessToken(auth[7:])
            return str(token["user_id"])
        except Exception:
            return None

    def __call__(self, request):
        response = self.get_response(request)

        user_id = self._get_user_id(request)
        if not user_id:
            return response

        try:
            r = self._get_redis()
            now = time.time()
            cutoff = now - self.WINDOW_SECONDS

            pipe = r.pipeline(transaction=False)
            pipe.zadd(self.REDIS_KEY, {user_id: now})          # upsert last-seen
            pipe.zremrangebyscore(self.REDIS_KEY, 0, cutoff)   # prune stale
            pipe.zcard(self.REDIS_KEY)                          # unique active users
            pipe.expire(self.REDIS_KEY, self.WINDOW_SECONDS * 2)
            results = pipe.execute()

            active_count = results[2]

            from app.core.observability import observability as _obs
            _metrics = getattr(_obs, "metrics", None)
            if _metrics and "active_users" in _metrics:
                _metrics["active_users"].set(active_count)

        except Exception:
            pass

        return response
