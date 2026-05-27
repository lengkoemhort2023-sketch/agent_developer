import logging
import os

logger = logging.getLogger(__name__)


def setup_tracing() -> None:
    """Initialise OpenTelemetry → Tempo. No-op when OTEL_EXPORTER_OTLP_ENDPOINT is unset."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        logger.info("[OTel] OTEL_EXPORTER_OTLP_ENDPOINT not set — tracing disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({
            SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", "amk-agent"),
        })
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)
        logger.info(f"[OTel] Tracing active → {endpoint}")
    except Exception as exc:
        logger.warning(f"[OTel] Setup failed (non-fatal): {exc}")


def get_tracer(name: str = "docbot"):
    """Return an OTel tracer, or a silent no-op tracer if OTel is unavailable."""
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except Exception:
        return _NoopTracer()


class _NoopTracer:
    def start_as_current_span(self, name, **kwargs):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            yield _NoopSpan()

        return _cm()


class _NoopSpan:
    def set_attribute(self, *a, **kw): pass
    def record_exception(self, *a, **kw): pass
    def set_status(self, *a, **kw): pass
