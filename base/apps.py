from django.apps import AppConfig
from django.db.models.signals import post_migrate


def bootstrap_default_groups(sender, **kwargs):
    """Ensure the default auth groups exist after migrations complete."""

    app_config = kwargs.get("app_config")
    # Wait for the last installed app so Django has already created permissions
    # for the whole project before we attach them to groups.
    if app_config is None or app_config.label != "management_commands":
        return

    from base.roles import ensure_default_groups

    ensure_default_groups()


class BaseConfig(AppConfig):
    name = 'base'

    def ready(self):
        import base.admins

        # OpenTelemetry — initialise tracing and auto-instrument Django internals.
        # Safe to call when OTel packages are absent (all errors are caught inside).
        from base.tracing import setup_tracing
        setup_tracing()

        try:
            from opentelemetry.instrumentation.django import DjangoInstrumentor
            from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
            from opentelemetry.instrumentation.redis import RedisInstrumentor
            from opentelemetry.instrumentation.celery import CeleryInstrumentor
            DjangoInstrumentor().instrument()
            Psycopg2Instrumentor().instrument()
            RedisInstrumentor().instrument()
            CeleryInstrumentor().instrument()
        except ImportError:
            pass

        post_migrate.connect(
            bootstrap_default_groups,
            dispatch_uid="base.bootstrap_default_groups",
        )




