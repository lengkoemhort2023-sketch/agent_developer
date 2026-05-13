from celery.schedules import crontab
CELERY_BEAT_SCHEDULE = {
    "process_unprocessed_documents_task": {
        "task": "document.tasks.process_unprocessed_documents",
        "schedule": crontab(minute=0, hour=0),
    },
}
