"""
Observability Configuration Module

This module configures:
- Prometheus metrics
- OpenTelemetry tracing
- Structured logging for Loki
- Langfuse integration for LLM observability
"""

import os
import logging
import logging.handlers
import json
import threading
import warnings
from datetime import datetime, timezone
from typing import Optional

# Must come before OpenTelemetry imports — pkg_resources warns at import time.
warnings.filterwarnings("ignore", category=UserWarning, module="opentelemetry")
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
logging.getLogger("opentelemetry.instrumentation").setLevel(logging.CRITICAL)

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.django import DjangoInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry import trace, metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

# Thread-local storage for per-request context (populated by RequestIDMiddleware)
_request_context = threading.local()


def set_request_context(request_id: str = None, user_id: str = None):
    """Set per-request context so all log records carry request_id and user_id."""
    if request_id is not None:
        _request_context.request_id = request_id
    if user_id is not None:
        _request_context.user_id = user_id


def clear_request_context():
    """Clear request context after the request completes."""
    _request_context.request_id = None
    _request_context.user_id = None


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging (Loki compatible)"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Inject OpenTelemetry trace/span IDs for log-trace correlation in Grafana
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            log_data["trace_id"] = format(ctx.trace_id, "032x")
            log_data["span_id"] = format(ctx.span_id, "016x")

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add request context if available (set by RequestContextFilter)
        if hasattr(record, "request_id") and record.request_id:
            log_data["request_id"] = record.request_id
        if hasattr(record, "user_id") and record.user_id:
            log_data["user_id"] = record.user_id
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms

        return json.dumps(log_data)


class RequestContextFilter(logging.Filter):
    """Inject per-request context (request_id, user_id) from thread-local into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(_request_context, "request_id", None)
        record.user_id = getattr(_request_context, "user_id", None)
        return True


def init_prometheus_metrics():
    """Initialize Prometheus metrics"""
    
    # API metrics
    api_requests = Counter(
        "api_requests_total",
        "Total API requests",
        ["method", "endpoint", "status"],
        registry=REGISTRY
    )
    
    api_latency = Histogram(
        "api_request_duration_seconds",
        "API request latency",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        registry=REGISTRY
    )
    
    api_errors = Counter(
        "api_errors_total",
        "Total API errors",
        ["method", "endpoint", "error_type"],
        registry=REGISTRY
    )
    
    # RAG pipeline metrics
    rag_retrieval_duration = Histogram(
        "rag_retrieval_duration_seconds",
        "RAG retrieval latency",
        ["retriever_type"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        registry=REGISTRY
    )
    
    rag_llm_duration = Histogram(
        "rag_llm_duration_seconds",
        "LLM response generation latency",
        ["model_name"],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        registry=REGISTRY
    )
    
    rag_total_duration = Histogram(
        "rag_total_duration_seconds",
        "Total RAG pipeline latency",
        [],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        registry=REGISTRY
    )
    
    rag_retrieval_count = Counter(
        "rag_retrievals_total",
        "Total retrieval operations",
        ["retriever_type", "status"],
        registry=REGISTRY
    )

    # Per-phase RAG timing histograms
    rag_embed_duration = Histogram(
        "rag_embed_duration_seconds",
        "Embedding generation latency",
        [],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        registry=REGISTRY
    )

    rag_process_duration = Histogram(
        "rag_process_duration_seconds",
        "Response processing (chunk extraction, doc ref build) latency",
        [],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        registry=REGISTRY
    )

    rag_followup_duration = Histogram(
        "rag_followup_duration_seconds",
        "Follow-up question generation latency",
        [],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        registry=REGISTRY
    )

    # User & chat activity metrics
    user_requests = Counter(
        "user_requests_total",
        "Total authenticated user requests",
        ["authenticated"],
        registry=REGISTRY
    )

    chat_sessions = Counter(
        "chat_sessions_total",
        "Total chat sessions created",
        registry=REGISTRY
    )

    chat_messages = Counter(
        "chat_messages_total",
        "Total chat messages sent",
        registry=REGISTRY
    )
    
    # Database metrics
    db_query_duration = Histogram(
        "db_query_duration_seconds",
        "Database query latency",
        ["table", "operation"],
        buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        registry=REGISTRY
    )
    
    db_connection_pool = Gauge(
        "db_connection_pool_size",
        "Database connection pool size",
        registry=REGISTRY
    )
    
    # Celery task metrics
    celery_task_duration = Histogram(
        "celery_task_duration_seconds",
        "Celery task execution time",
        ["task_name", "status"],
        buckets=[1.0, 5.0, 10.0, 30.0, 60.0],
        registry=REGISTRY
    )

    celery_tasks_total = Counter(
        "celery_tasks_total",
        "Total Celery tasks processed",
        ["task_name", "status"],
        registry=REGISTRY
    )

    # Concurrency / saturation metrics (section F)
    active_requests = Gauge(
        "api_active_requests",
        "Number of requests currently being processed",
        registry=REGISTRY
    )

    timeout_errors = Counter(
        "api_timeout_errors_total",
        "Total request timeout errors",
        ["endpoint"],
        registry=REGISTRY
    )

    return {
        "api_requests": api_requests,
        "api_latency": api_latency,
        "api_errors": api_errors,
        "rag_retrieval_duration": rag_retrieval_duration,
        "rag_llm_duration": rag_llm_duration,
        "rag_total_duration": rag_total_duration,
        "rag_retrieval_count": rag_retrieval_count,
        "rag_embed_duration": rag_embed_duration,
        "rag_process_duration": rag_process_duration,
        "rag_followup_duration": rag_followup_duration,
        "user_requests": user_requests,
        "chat_sessions": chat_sessions,
        "chat_messages": chat_messages,
        "db_query_duration": db_query_duration,
        "db_connection_pool": db_connection_pool,
        "celery_task_duration": celery_task_duration,
        "celery_tasks_total": celery_tasks_total,
        "active_requests": active_requests,
        "timeout_errors": timeout_errors,
    }


def init_otel_tracing():
    """Initialize OpenTelemetry tracing"""
    
    otel_enabled = os.environ.get("OTEL_ENABLED", "true").lower() in {"true", "1", "yes"}
    if not otel_enabled:
        return None
    
    # Create resource
    resource = Resource.create({
        "service.name": os.environ.get("OTEL_SERVICE_NAME", "amk-agent"),
        "service.version": os.environ.get("OTEL_SERVICE_VERSION", "1.0.0"),
        "deployment.environment": os.environ.get("OTEL_ENVIRONMENT", "development"),
    })
    
    # Setup OTLP exporter for Tempo
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    
    try:
        trace_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
        trace.set_tracer_provider(tracer_provider)
        
        # Instrument frameworks
        for instr in [DjangoInstrumentor, RequestsInstrumentor, CeleryInstrumentor]:
            try:
                instr().instrument()
            except Exception:
                pass

        return tracer_provider
    except Exception as e:
        logging.warning(f"Failed to initialize OpenTelemetry: {e}")
        return None


def init_otel_metrics():
    """Initialize OpenTelemetry metrics"""
    
    otel_metrics_enabled = os.environ.get("OTEL_METRICS_ENABLED", "true").lower() in {"true", "1", "yes"}
    if not otel_metrics_enabled:
        return None
    
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    
    try:
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
        )
        meter_provider = MeterProvider(metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        return meter_provider
    except Exception as e:
        logging.warning(f"Failed to initialize OpenTelemetry metrics: {e}")
        return None


def init_langfuse(enabled: bool = True):
    """Initialize Langfuse for LLM observability"""
    
    if not enabled:
        return None
    
    try:
        from langfuse.decorators import observe
        from langfuse import Langfuse
        
        langfuse = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3002"),
        )
        
        return langfuse
    except Exception as e:
        logging.warning(f"Langfuse initialization failed (optional): {e}")
        return None


def setup_structured_logging(log_level: str = "INFO"):
    """Setup structured logging for all loggers"""
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup console handler with JSON formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    console_handler.addFilter(RequestContextFilter())
    root_logger.addHandler(console_handler)

    # Also ship JSON logs to file for Promtail → Loki
    try:
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            "logs/app.log", maxBytes=10_485_760, backupCount=3
        )
        file_handler.setFormatter(JSONFormatter())
        file_handler.addFilter(RequestContextFilter())
        root_logger.addHandler(file_handler)
    except Exception:
        pass
    
    # Setup Django loggers
    django_logger = logging.getLogger("django")
    django_logger.setLevel(getattr(logging, log_level, logging.INFO))
    
    for logger_name in ["django.request", "django.db.backends", "celery"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level, logging.INFO))


class ObservabilityContext:
    """Context manager for observability configuration"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, "initialized"):
            return  # Already set up — prevent singleton re-init resetting state
        self.metrics = None
        self.tracer_provider = None
        self.meter_provider = None
        self.langfuse = None
        self.initialized = False
    
    def initialize(self, django_settings):
        """Initialize all observability components"""
        if self.initialized:
            return
        
        # Setup logging
        log_level = os.environ.get("LOG_LEVEL", "INFO")
        setup_structured_logging(log_level)
        
        # Initialize Prometheus metrics
        self.metrics = init_prometheus_metrics()
        
        # Initialize OpenTelemetry
        self.tracer_provider = init_otel_tracing()
        self.meter_provider = init_otel_metrics()
        
        # Initialize Langfuse
        langfuse_enabled = os.environ.get("LANGFUSE_ENABLED", "false").lower() in {"true", "1", "yes"}
        self.langfuse = init_langfuse(enabled=langfuse_enabled)
        
        self.initialized = True
        logging.info("Observability stack initialized successfully")


# Global observability context
observability = ObservabilityContext()
