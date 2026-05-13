from django.urls import path
from . import views
from .streaming import stream_chat_message

app_name = 'chat'

urlpatterns = [
    # Chat Session URLs
    path('sessions/', views.list_chat_sessions, name='list_chat_sessions'),
    path('sessions/<uuid:session_id>/', views.get_chat_session, name='get_session'),
    path('sessions/<uuid:session_id>/archive/', views.archive_chat_session, name='archive_session'),
    path('sessions/<uuid:session_id>/restore/', views.restore_chat_session, name='restore_session'),
    path('sessions/<uuid:session_id>/delete/', views.delete_chat_session, name='delete_session'),
    path('sessions/<uuid:session_id>/rename/', views.rename_chat_session, name='rename_session'),
    path('sessions/<uuid:session_id>/bump/', views.bump_chat_session, name='bump_session'),

    # Chat Message URLs
    path('session/messages/', views.create_chat_message, name='create_message'),
    path('sessions/<uuid:session_id>/messages/<uuid:message_id>/', views.get_chat_message, name='get_message'),
    path('sessions/<uuid:session_id>/memos/', views.get_session_memos, name='get_session_memos'),

    # Streaming endpoint - always register, let the view handle errors
    path('stream/', stream_chat_message, name='stream_message'),

    # Queue API endpoints
    path('queue/enqueue/', views.enqueue_question, name='enqueue_question'),
    path('queue/status/<str:job_id>/', views.get_queue_job_status, name='queue_job_status'),
    path('queue/stats/', views.get_queue_stats, name='queue_stats'),
    path('queue/cancel/<str:job_id>/', views.cancel_queue_job, name='queue_cancel'),

    # Feedback endpoints
    path('feedback/', views.submit_feedback, name='submit_feedback'),
    path('feedback/list/', views.list_feedback, name='list_feedback'),
]







