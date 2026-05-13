from celery import shared_task
from chat.utils import ask_question

@shared_task(bind=True)
def generate_response_task(self, question, file_type=None, user_token=None, session_id=None):
    """Celery task wrapper around the synchronous ask_question flow.

    The task returns the same structure as `ask_question` (dict with answers/message).
    """
    # The ask_question function will internally call the RagProviders pipeline
    # which already checks `cancelled_sessions` at many checkpoints. We additionally
    # rely on Celery revoke (terminate=True) to interrupt long-running tasks.
    result = ask_question(question=question, file_type=file_type, user_token=user_token, session_id=session_id)
    return result







