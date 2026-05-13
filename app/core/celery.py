import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from celery import Celery

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
    worker_max_tasks_per_child=1,
    # Configure worker to use GPU when available
    worker_enable_remote_control=True,
    # Use solo pool to avoid forking and CUDA issues
    worker_pool='solo',
)
