import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from celery import Celery
from celery.signals import task_postrun

from .env import load_environment

load_environment()
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'app.core.settings')

celery_app = Celery('app.core')
celery_app.config_from_object('django.conf:settings', namespace='CELERY')
celery_app.autodiscover_tasks()

# Enable GPU support for Celery workers
celery_app.conf.update(
    # Enable GPU support
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    # Allow workers to use GPU
    worker_enable_remote_control=True,
    # Thread pool avoids CUDA fork issues (threads share the same process)
    # and allows multiple tasks to run in parallel, hitting Ollama concurrently.
    worker_pool='threads',
)


@task_postrun.connect
def flush_langfuse_after_task(sender=None, **kwargs):
    """Flush Langfuse SDK buffer after every Celery task so traces are not lost when the worker exits."""
    try:
        from langfuse import Langfuse
        Langfuse().flush()
    except Exception:
        pass
