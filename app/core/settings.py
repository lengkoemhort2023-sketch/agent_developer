from pathlib import Path
import os
import locale
from typing import List
from django.contrib.auth.signals import user_logged_in, user_login_failed
import logging

from .env import load_environment

try:
    import ldap
    from django_auth_ldap.config import LDAPSearch, ActiveDirectoryGroupType
    LDAP_AUTH_DEPENDENCIES_AVAILABLE = True
except ImportError:
    class _LDAPStub:
        OPT_REFERRALS = 0
        OPT_PROTOCOL_VERSION = 3
        OPT_NETWORK_TIMEOUT = 5.0
        VERSION3 = 3
        SCOPE_SUBTREE = 2

    class LDAPSearch:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class ActiveDirectoryGroupType:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    ldap = _LDAPStub()
    LDAP_AUTH_DEPENDENCIES_AVAILABLE = False

from .celerybeat_schedule import CELERY_BEAT_SCHEDULE

load_environment()
# Disable django-prometheus thread exporter early to avoid autoreloader conflicts (prevents AssertionError in development)
os.environ.setdefault('PROMETHEUS_DISABLE_THREAD_EXPORTER', '1')
logger = logging.getLogger(__name__)


def env_bool(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in {"1", "true", "yes", "on"}


def env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    return int(value) if value not in (None, "") else default


def env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    return float(value) if value not in (None, "") else default


def env_list(key: str, default: List[str] | None = None) -> List[str]:
    value = os.environ.get(key)
    if value is None or not value.strip():
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]

def log_user_login(sender, user, request=None, **kwargs):
    remote = request.META.get('REMOTE_ADDR') if request is not None and hasattr(request, 'META') else 'Unknown'
    ua = request.META.get('HTTP_USER_AGENT', 'Unknown') if request is not None and hasattr(request, 'META') else 'Unknown'
    logger.info(f"User {user.username} logged in successfully from {remote} via {ua}")

def log_user_login_failed(sender, credentials, request=None, **kwargs):
    remote = request.META.get('REMOTE_ADDR') if request is not None and hasattr(request, 'META') else 'Unknown'
    ua = request.META.get('HTTP_USER_AGENT', 'Unknown') if request is not None and hasattr(request, 'META') else 'Unknown'
    logger.warning(f"Failed login attempt for user {credentials.get('username', 'Unknown')} from {remote} via {ua}")

user_logged_in.connect(log_user_login)
user_login_failed.connect(log_user_login_failed) 
try:
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
except locale.Error:
    # Fallback if locale is missing in container
    locale.setlocale(locale.LC_ALL, "C.UTF-8")
CELERY_BEAT_SCHEDULER_OPTIONS = {
    "locale_code": "en"
}
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get("SECRET_KEY")  # required — set in .env.dev

DEBUG = env_bool("DEBUG", False)

# ---------------------------------------------------------------------------
# Host & Origin Configuration for nginx deployments
#
# When deploying behind nginx, you must ensure:
# 1. Your nginx server's domain name or IP address is included in ALLOWED_HOSTS.
#    Example: DJANGO_ALLOWED_HOSTS="localhost 127.0.0.1 [::1] yourdomain.com nginx_ip"
#
# 2. If nginx serves your frontend or proxies requests from another domain,
#    add that domain to CORS_ALLOWED_ORIGINS.
#    Example: CORS_ALLOWED_ORIGINS="http://localhost:9898 https://yourfrontend.com https://nginx_ip"
#
# 3. For CSRF-protected POST requests from your frontend (served by nginx),
#    add the frontend domain to CSRF_TRUSTED_ORIGINS.
#    Example: CSRF_TRUSTED_ORIGINS="http://localhost:9898 https://yourfrontend.com https://nginx_ip"
#
# This ensures Django will accept requests from nginx and your frontend,
# and will not block them due to host, CORS, or CSRF restrictions.
# ---------------------------------------------------------------------------
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS")

# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django_celery_results',
    'django_celery_beat',
    'django_prometheus',
    'user.apps.UserConfig',
    'rest_framework_simplejwt.token_blacklist',
    'base',
    'department',
    'docs_type',
    'document',
    'chat',
    'management_commands'

]

AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
]

LDAP_AUTH_ENABLED = (
    os.environ.get("ENABLE_LDAP_AUTH", "True").lower() == "true"
    and LDAP_AUTH_DEPENDENCIES_AVAILABLE
)

if LDAP_AUTH_ENABLED:
    AUTHENTICATION_BACKENDS.append('django_auth_ldap.backend.LDAPBackend')

X_FRAME_OPTIONS = "ALLOWALL"
SILENCED_SYSTEM_CHECKS = ["security.W019"]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    'app.core.middleware.RequestIDMiddleware',
    'app.core.middleware.APIMetricsMiddleware',
    'django_prometheus.middleware.PrometheusBeforeMiddleware',
    'base.middleware.TraceIDMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'base.middlewares.csrf_logging_middleware.CsrfLoggingMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    "django.middleware.common.CommonMiddleware",
    "base.middleware.CustomErrorHandlerMiddleware",
    "base.middlewares.connection_reset_middleware.ConnectionResetMiddleware",
    'django_prometheus.middleware.PrometheusAfterMiddleware',
]

# Proxy header configuration for nginx reverse proxy
SECURE_PROXY_SSL_HEADER = (
    os.environ.get("SECURE_PROXY_SSL_HEADER_NAME", "HTTP_X_FORWARDED_PROTO"),
    os.environ.get("SECURE_PROXY_SSL_HEADER_VALUE", "https"),
)
USE_X_FORWARDED_HOST = True
USE_X_FORWARDED_PORT = True

# CSRF_TRUSTED_ORIGINS configuration
# This setting defines a list of trusted origins for CSRF protection.
# It should be a space-separated string of URLs in your .env file.
# Example: CSRF_TRUSTED_ORIGINS="http://localhost:9898 https://yourdomain.com"
# Django will allow cross-site POST requests from these origins.
# Note: Browsers send Origin header without default ports (443 for HTTPS, 80 for HTTP)
CSRF_TRUSTED_ORIGINS = env_list("CSRF_TRUSTED_ORIGINS")

# CSRF Cookie settings for HTTPS/reverse proxy
CSRF_COOKIE_SECURE = env_bool("CSRF_COOKIE_SECURE", False)
CSRF_COOKIE_HTTPONLY = False  # Must be False for JavaScript to read CSRF token
CSRF_COOKIE_SAMESITE = 'Lax'

# CORS configuration
# CORS_ALLOWED_ORIGINS should be a space-separated string of allowed origins in your .env file.
# Example: CORS_ALLOWED_ORIGINS="http://localhost:9898 http://127.0.0.1:9898 https://yourdomain.com"
CORS_ALLOWED_ORIGINS = env_list("CORS_ALLOWED_ORIGINS")

# Set to True to allow cookies and credentials in cross-origin requests.
CORS_ALLOW_CREDENTIALS = env_bool("CORS_ALLOW_CREDENTIALS", True)

# In development mode (DEBUG=True), allow all origins for CORS requests.
# This is useful for local testing and development, but should NOT be enabled in production.
# When DEBUG=False, only the origins listed in CORS_ALLOWED_ORIGINS are allowed.
if DEBUG:
    CORS_ALLOW_ALL_ORIGINS = True
else:
    CORS_ALLOW_ALL_ORIGINS = env_bool("CORS_ALLOW_ALL_ORIGINS", False)

ROOT_URLCONF = 'app.core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'app.core.wsgi.application'


# Database
# https://docs.djangoproject.com/en/5.2/ref/settings/#databases

DB_ENGINE = os.environ.get("DB_ENGINE", "django.db.backends.sqlite3")
DB_NAME = os.environ.get("DB_DATABASE", str(PROJECT_ROOT / "db.sqlite3"))

if DB_ENGINE == "django.db.backends.sqlite3":
    DATABASES = {
        "default": {
            "ENGINE": DB_ENGINE,
            "NAME": DB_NAME,
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": DB_ENGINE,
            "NAME": DB_NAME,
            "USER": os.environ.get("DB_USER", ""),
            "PASSWORD": os.environ.get("DB_PASSWORD", ""),
            "HOST": os.environ.get("DB_HOST", ""),
            "PORT": os.environ.get("DB_PORT", ""),
        }
    }

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "")
CELERY_RESULT_EXTENDED = True
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "django-db")

# Qdrant Vector Database
QDRANT_HOST = os.environ.get("QDRANT_HOST", "")
QDRANT_PORT = env_int("QDRANT_PORT", 0)

CHROMA_HOST = os.environ.get("CHROMA_HOST", "")
CHROMA_PORT = env_int("CHROMA_PORT", 0)

DJANGO_HOST_URL = os.environ.get("DJANGO_HOST_URL", "")
DOC_CONVERTER_URL = os.environ.get("DOC_CONVERTER_URL", "")

# Model / embedding paths
BGE_M3_MODEL_PATH = os.environ.get("BGE_M3_MODEL_PATH", "")
RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH", "")

# Ollama LLM Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "")
OLLAMA_NUM_CTX = env_int("OLLAMA_NUM_CTX", 16384)
OLLAMA_NUM_PREDICT = env_int("OLLAMA_NUM_PREDICT", 1024)
OLLAMA_TEMPERATURE = env_float("OLLAMA_TEMPERATURE", 0.1)

# RAG Configuration
# Set to True to use Generative RAG (LLM generates answers from context)
# Set to False to use Extractive RAG (return formatted document chunks directly)
USE_GENERATIVE_RAG = env_bool("USE_GENERATIVE_RAG", True)


# Password validation
# https://docs.djangoproject.com/en/5.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.2/topics/i18n/

LANGUAGE_CODE = os.environ.get("LANGUAGE_CODE", "en-us")

TIME_ZONE = os.environ.get("TIME_ZONE", "Asia/Phnom_Penh")

USE_I18N = env_bool("USE_I18N", False)

USE_TZ = env_bool("USE_TZ", True)

AUTH_USER_MODEL = "user.User"

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.2/howto/static-files/

STATIC_URL = "/static/"
STATIC_ROOT = Path(os.environ.get("STATIC_ROOT", ""))
STATICFILES_FINDERS = [
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
]

# WhiteNoise configuration for serving static files in production
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

MEDIA_URL = os.environ.get("MEDIA_URL", "")
MEDIA_ROOT = Path(os.environ.get("MEDIA_ROOT", ""))

# File upload settings for large files
FILE_UPLOAD_MAX_MEMORY_SIZE = env_int("FILE_UPLOAD_MAX_MEMORY_SIZE", 100 * 1024 * 1024)
DATA_UPLOAD_MAX_MEMORY_SIZE = env_int("DATA_UPLOAD_MAX_MEMORY_SIZE", 2000 * 1024 * 1024)
FILE_UPLOAD_CHUNK_SIZE = env_int("FILE_UPLOAD_CHUNK_SIZE", 2621440)

# Use temporary file handler for streaming uploads to avoid memory issues
FILE_UPLOAD_HANDLERS = [
    'django.core.files.uploadhandler.TemporaryFileUploadHandler',
]

# Default primary key field type
# https://docs.djangoproject.com/en/5.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# FIXED REST_FRAMEWORK configuration
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_RENDERER_CLASSES': (
        'rest_framework.renderers.JSONRenderer',
    ),
    'DEFAULT_PARSER_CLASSES': (
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
        'rest_framework.parsers.JSONParser',
    ),
    'EXCEPTION_HANDLER': 'base.utils.custom_exception_handler',
}
from datetime import timedelta
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'SLIDING_TOKEN_REFRESH_LIFETIME': timedelta(days=1),
    'SLIDING_TOKEN_LIFETIME': timedelta(days=30),
    'SLIDING_TOKEN_REFRESH_LIFETIME_LATE_USER': timedelta(days=1),
    'SLIDING_TOKEN_LIFETIME_LATE_USER': timedelta(days=30),
    'ROTATE_REFRESH_TOKENS': True,
}
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG" if DEBUG else "INFO").upper()
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': LOG_LEVEL,
    },
    'loggers': {
        'django_auth_ldap': {'handlers': ['console'], 'level': LOG_LEVEL, 'propagate': False},
        'ldap': {'handlers': ['console'], 'level': LOG_LEVEL, 'propagate': False},
    },
}


# LDAP server
# https://django-auth-ldap.readthedocs.io/en/latest/
AUTH_LDAP_SERVER_URI = os.environ.get("AUTH_LDAP_SERVER_URI", "")
AUTH_LDAP_BIND_DN = os.environ.get("AUTH_LDAP_BIND_DN", "")
AUTH_LDAP_BIND_PASSWORD = os.environ.get("AUTH_LDAP_BIND_PASSWORD", "")

# CRITICAL: Connection options for Active Directory
# These options fix "bind must be completed on connection" errors
AUTH_LDAP_CONNECTION_OPTIONS = {
    ldap.OPT_REFERRALS: 0,                    # Disable referrals (AD-specific)
    ldap.OPT_PROTOCOL_VERSION: ldap.VERSION3, # Use LDAP v3
    ldap.OPT_NETWORK_TIMEOUT: env_float("AUTH_LDAP_NETWORK_TIMEOUT", 5.0),
}

# Search entire domain (users can be in various OUs)
AUTH_LDAP_USER_SEARCH = LDAPSearch(
    os.environ.get("AUTH_LDAP_USER_BASE_DN", ""),
    ldap.SCOPE_SUBTREE,
    os.environ.get("AUTH_LDAP_USER_SEARCH_FILTER", "(|(sAMAccountName=%(user)s)(mail=%(user)s))"),
)

# PERFORMANCE: Scope group search to the app-specific OU only (not full domain).
# With 20,000 user-groups searching DC root is extremely slow.
# Set AUTH_LDAP_GROUP_BASE_DN in .env to a narrow OU, e.g.:
#   OU=AppGroups,DC=amkcambodia,DC=com
# Only groups relevant to this application should live there.
AUTH_LDAP_GROUP_SEARCH = LDAPSearch(
    os.environ.get("AUTH_LDAP_GROUP_BASE_DN", ""),
    ldap.SCOPE_SUBTREE,
    os.environ.get("AUTH_LDAP_OBJECT_CLASS_FOR_USER_ROLE", "(objectClass=group)"),
)

AUTH_LDAP_GROUP_TYPE = ActiveDirectoryGroupType()

# Map AD attributes to Django user fields
AUTH_LDAP_USER_ATTR_MAP = {
    "username": os.environ.get("AUTH_LDAP_USERNAME_ATTR", "sAMAccountName"),  # AD username
    "first_name": os.environ.get("AUTH_LDAP_FIRST_NAME_ATTR", "givenName"),
    "last_name": os.environ.get("AUTH_LDAP_LAST_NAME_ATTR", "sn"),
    "email": os.environ.get("AUTH_LDAP_EMAIL_ATTR", "mail"),
}

# Group-based permissions
AUTH_LDAP_USER_FLAGS_BY_GROUP = {
    "is_superuser": os.environ.get("AUTH_LDAP_SUPERUSER_GROUP", ""),
    "is_staff": os.environ.get("AUTH_LDAP_STAFF_GROUP", os.environ.get("AUTH_LDAP_SUPERUSER_GROUP", "")),
}

# PERFORMANCE: Only mirror the specific app groups (Admin, User, etc.) — NOT all 20,000 user groups.
# List only the CN values of groups that should be synced to Django.
_ldap_mirror_groups_env = os.environ.get("AUTH_LDAP_MIRROR_GROUPS", "")
AUTH_LDAP_MIRROR_GROUPS = (
    [g.strip() for g in _ldap_mirror_groups_env.split(",") if g.strip()]
    if _ldap_mirror_groups_env
    else True  # fallback: mirror all (only safe if GROUP_BASE_DN is already narrowed)
)

AUTH_LDAP_ALWAYS_UPDATE_USER = env_bool("AUTH_LDAP_ALWAYS_UPDATE_USER", True)
AUTH_LDAP_FIND_GROUP_PERMS = env_bool("AUTH_LDAP_FIND_GROUP_PERMS", True)

# PERFORMANCE: Cache LDAP lookups so repeated logins don't re-query AD.
# 3600 = 1 hour. Increase if AD data changes infrequently.
AUTH_LDAP_CACHE_TIMEOUT = env_int("AUTH_LDAP_CACHE_TIMEOUT", 3600)

# LDAP Group to Django Permission Group Mapping
AUTH_LDAP_GROUP_TO_DJANGO_GROUP_MAP = {
    os.environ.get("AUTH_LDAP_ADMIN_GROUP_NAME", "admin"): os.environ.get("AUTH_LDAP_DJANGO_ADMIN_GROUP", "Admin"),
}
# LDAP admin group name (case-insensitive matching)
AUTH_LDAP_ADMIN_GROUP_NAME = os.environ.get("AUTH_LDAP_ADMIN_GROUP_NAME", "admin")
# Django groups to assign (must exist in Django)
AUTH_LDAP_DJANGO_ADMIN_GROUP = os.environ.get("AUTH_LDAP_DJANGO_ADMIN_GROUP", "Admin")
AUTH_LDAP_DJANGO_USER_GROUP = os.environ.get("AUTH_LDAP_DJANGO_USER_GROUP", "User")

# ============================================================================
# OBSERVABILITY & MONITORING
# ============================================================================

# Initialize observability stack
from app.core.observability import observability
observability.initialize(globals())

# Prometheus metrics endpoint configuration
PROMETHEUS_METRICS_EXPORT_PORT = env_int("PROMETHEUS_METRICS_EXPORT_PORT", 8000)

# OpenTelemetry configuration
OTEL_ENABLED = env_bool("OTEL_ENABLED", True)
OTEL_METRICS_ENABLED = env_bool("OTEL_METRICS_ENABLED", True)
OTEL_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "amk-agent")
OTEL_SERVICE_VERSION = os.environ.get("OTEL_SERVICE_VERSION", "1.0.0")
OTEL_ENVIRONMENT = os.environ.get("OTEL_ENVIRONMENT", "development")
OTEL_EXPORTER_OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4317")

# Langfuse configuration (for LLM observability)
LANGFUSE_ENABLED = env_bool("LANGFUSE_ENABLED", False)
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")

# Loki logging configuration
LOKI_ENABLED = env_bool("LOKI_ENABLED", True)
LOKI_ENDPOINT = os.environ.get("LOKI_ENDPOINT", "http://loki:3100")

# Django-Prometheus configuration
# Disable automatic thread-based exporter (we use URL exporter instead at /api/metrics/)
# This prevents conflicts with Django's autoreloader in development
PROMETHEUS_EXPORT_MIGRATIONS = env_bool("PROMETHEUS_EXPORT_MIGRATIONS", True)
if os.environ.get("RUN_MAIN") == "true":
    # In autoreloader child process - disable thread exporter
    os.environ["PROMETHEUS_DISABLE_THREAD_EXPORTER"] = "1"

