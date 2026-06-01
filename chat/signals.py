from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver


def _sync_active_sessions():
    """Update the chat_sessions_active Prometheus gauge with the live DB count."""
    try:
        from .models import ChatSession
        from app.core.observability import observability as _obs
        metrics = getattr(_obs, "metrics", None)
        if metrics and "chat_sessions_active" in metrics:
            count = ChatSession.objects.filter(is_archived=False).count()
            metrics["chat_sessions_active"].set(count)
    except Exception:
        pass


@receiver(post_save, sender="chat.ChatSession")
def on_session_save(sender, **kwargs):
    _sync_active_sessions()


@receiver(post_delete, sender="chat.ChatSession")
def on_session_delete(sender, **kwargs):
    _sync_active_sessions()
