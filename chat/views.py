import json
import logging
import mimetypes
import os
import tempfile

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from base.application.chat_queries import AnswerUserQueryHandler
from base.utils import error_response, handle_exception, success_response
from base.utils.validators import (
    ChatHistoryParams,
    ChatMessageRequest,
    ChatSessionListParams,
)
from base.utils.exceptions import (
    ChatError,
    InternalServerError as APIInternalServerError,
)
from pydantic import ValidationError

from .models import ChatMessage, ChatSession, MessageFeedback
from .serializers import (
    ChatMessageCreateSerializer,
    ChatMessageDetailSerializer,
    ChatSessionDetailSerializer,
    ChatSessionListSerializer,
)

logger = logging.getLogger(__name__)

VOICE_UPLOAD_EXTENSION_MAP = {
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}


def ensure_voice_file_extension(uploaded_file):
    filename = getattr(uploaded_file, "name", "") or "voice-input"
    if os.path.splitext(filename)[1]:
        return uploaded_file

    content_type = getattr(uploaded_file, "content_type", "") or ""
    guessed_extension = VOICE_UPLOAD_EXTENSION_MAP.get(content_type) or mimetypes.guess_extension(content_type)
    if guessed_extension == ".weba":
        guessed_extension = ".webm"
    if not guessed_extension and content_type.startswith("audio/"):
        guessed_extension = ".webm"

    if guessed_extension:
        uploaded_file.name = f"{filename}{guessed_extension}"

    return uploaded_file


def parse_archived_flag(value, default=False):
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    return default


@api_view(["GET", "POST"])
@handle_exception
def list_chat_sessions(request):
    if request.method == "GET":
        archived = parse_archived_flag(request.query_params.get("archived"), default=False)
        queryset = ChatSession.objects.filter(
            user=request.user,
            is_archived=archived,
        ).order_by("-updated_at")

        title = request.query_params.get("title")
        if title:
            queryset = queryset.filter(title__icontains=title)

        try:
            limit = int(request.query_params.get("limit", 100))
            offset = int(request.query_params.get("offset", 0))
        except (ValueError, TypeError):
            limit = 100
            offset = 0

        total_count = queryset.count()
        queryset = queryset[offset : offset + limit]

        serializer = ChatSessionListSerializer(queryset, many=True)
        return success_response(
            message="Chat sessions retrieved successfully",
            data={"count": total_count, "results": serializer.data},
        )

    session = ChatSession.get_last_empty(user=request.user)
    if not session:
        session = ChatSession.objects.create(user=request.user)
    response_serializer = ChatSessionDetailSerializer(session)
    return success_response(
        "Chat session created successfully",
        response_serializer.data,
        status.HTTP_201_CREATED,
    )


@api_view(["GET"])
@handle_exception
def get_chat_session(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    serializer = ChatSessionDetailSerializer(session)
    return success_response("Chat session retrieved successfully", serializer.data)


@api_view(["GET"])
@handle_exception
def chat_session_history(request):
    try:
        query_params = {
            "take": int(request.query_params.get("take", 10)),
            "skip": int(request.query_params.get("skip", 0)),
        }
        validated_params = ChatHistoryParams(**query_params)
    except (ValueError, ValidationError) as e:
        logger.warning(f"Validation error in chat_session_history: {e}")
        return error_response(
            "Invalid pagination parameters",
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"error": str(e)} if isinstance(e, ValueError) else e.errors(),
        )

    title = request.query_params.get("title", "").strip()
    archived = parse_archived_flag(request.query_params.get("archived"), default=False)
    queryset = ChatSession.objects.filter(user=request.user, is_archived=archived)

    if title:
        queryset = queryset.filter(title__icontains=title)

    total_count = queryset.count()
    queryset = queryset.order_by("-created_at")[
        validated_params.skip : validated_params.skip + validated_params.take
    ]
    serializer = ChatSessionListSerializer(queryset, many=True)

    return success_response(
        message="Chat session history retrieved successfully",
        data={"count": total_count, "results": serializer.data},
    )


@api_view(["POST"])
@handle_exception
def create_chat_message(request):
    from .utils import ask_question, audio_processing

    logger.info(
        f"Received request - Data keys: {list(request.data.keys())}, Files: {list(request.FILES.keys())}"
    )

    session_id = request.query_params.get("session_id", None)
    if session_id and "," in session_id:
        session_id = session_id.split(",")[0].strip()
        logger.warning(f"Multiple session_ids detected, using first one: {session_id}")

    normalized_data = {}
    for key, value in request.data.items():
        if isinstance(value, list) and len(value) > 0:
            normalized_data[key] = value[0]
        else:
            normalized_data[key] = value

    logger.info(f"Normalized data: {normalized_data}")

    uploaded_voice_file = None
    if "voice_file" in request.FILES:
        uploaded_voice_file = ensure_voice_file_extension(request.FILES["voice_file"])
        normalized_data["voice_file"] = uploaded_voice_file
        logger.info(f"Voice file found: {uploaded_voice_file.name}")

    try:
        validated_data = ChatMessageRequest(**normalized_data)
        logger.info(f"Validation successful: input_type={validated_data.input_type}")
    except ValidationError as e:
        logger.warning(f"Validation error in create_chat_message: {e.errors()}")
        return error_response("Invalid message data", data=e.errors())

    serializer_data = {}
    for key, value in request.data.items():
        if key == "voice_file":
            continue
        if isinstance(value, list) and len(value) > 0:
            serializer_data[key] = value[0]
        else:
            serializer_data[key] = value

    if uploaded_voice_file is not None:
        serializer_data["voice_file"] = uploaded_voice_file

    serializer = ChatMessageCreateSerializer(data=serializer_data)
    if not serializer.is_valid():
        logger.error(f"Serializer validation failed: {serializer.errors}")
        return error_response("Invalid message data", data=serializer.errors)

    input_type = serializer.validated_data.get("input_type", "text")
    question = validated_data.question
    voice_file = serializer.validated_data.get("voice_file", None)
    document_type = validated_data.file_type or serializer.validated_data.get("document_type")
    doc_id = serializer.validated_data.get("doc_id")

    if session_id:
        session = get_object_or_404(ChatSession, id=session_id, user=request.user)
        if session.is_archived:
            session.is_archived = False
            session.archived_at = None
            session.save(update_fields=["is_archived", "archived_at", "updated_at"])

    if input_type == "voice" and voice_file:
        logger.info("Processing voice upload before query execution")
        temp_voice_path = None
        try:
            if hasattr(voice_file, "temporary_file_path"):
                voice_path = voice_file.temporary_file_path()
            else:
                suffix = ""
                if getattr(voice_file, "name", None) and "." in voice_file.name:
                    suffix = os.path.splitext(voice_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    for chunk in voice_file.chunks():
                        temp_file.write(chunk)
                    temp_voice_path = temp_file.name
                voice_path = temp_voice_path

            transcription_text = audio_processing(voice_path)
            if isinstance(transcription_text, JsonResponse):
                return transcription_text
            question = transcription_text
            logger.info(f"Transcription successful: {question[:100]}...")
        except Exception as e:
            logger.error(f"Audio processing failed: {e}", exc_info=True)
            error_msg = str(e)
            
            # Return user-friendly error message
            if "unavailable" in error_msg or "corrupted" in error_msg or "not installed" in error_msg:
                return error_response(
                    error_msg,
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    error_code="VOICE_UNAVAILABLE",
                )
            else:
                return error_response(
                    "Failed to process voice input. Please try again or use text input.",
                    data={"details": error_msg},
                    error_code="AUDIO_PROCESSING_ERROR",
                )
        finally:
            if temp_voice_path and os.path.exists(temp_voice_path):
                os.unlink(temp_voice_path)

    try:
        answer_query_handler = AnswerUserQueryHandler(answer_fn=ask_question)
        result = answer_query_handler.execute(
            user=request.user,
            question=question,
            user_token=request.META.get("HTTP_AUTHORIZATION", "").replace("Bearer ", ""),
            session_id=session_id,
            file_type=document_type,
            doc_id=doc_id,
            input_type=input_type,
            voice_file=voice_file,
            missing_session_policy="error",
        )
    except ChatSession.DoesNotExist:
        return error_response("Chat session not found", status_code=status.HTTP_404_NOT_FOUND)
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid input to AI service: {str(e)}", exc_info=True)
        raise ChatError(f"Invalid message format: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating bot response: {str(e)}", exc_info=True)
        raise APIInternalServerError(f"Failed to generate AI response: {str(e)}")

    if isinstance(result.response_data, dict):
        doc_refs = result.response_data.get("document_references", [])
        logger.info(
            f"Bot response contains {len(doc_refs)} document_references: "
            f"{[ref.get('file_name') for ref in doc_refs]}"
        )

    response_serializer = ChatMessageDetailSerializer(result.message)
    return success_response(
        "Chat message created successfully",
        response_serializer.data,
        status.HTTP_201_CREATED,
    )


@api_view(["GET"])
@handle_exception
def get_chat_message(request, session_id, message_id):
    message = get_object_or_404(
        ChatMessage,
        id=message_id,
        session_id=session_id,
        session__user=request.user,
    )
    serializer = ChatMessageDetailSerializer(message)
    return success_response("Chat message retrieved successfully", serializer.data)


@api_view(["DELETE"])
@handle_exception
def delete_chat_session(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    session.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(["PATCH"])
@handle_exception
def archive_chat_session(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    if not session.is_archived:
        session.is_archived = True
        session.archived_at = timezone.now()
        session.save(update_fields=["is_archived", "archived_at", "updated_at"])

    serializer = ChatSessionDetailSerializer(session)
    return success_response("Chat session archived successfully", serializer.data)


@api_view(["PATCH"])
@handle_exception
def restore_chat_session(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    if session.is_archived:
        session.is_archived = False
        session.archived_at = None
        session.save(update_fields=["is_archived", "archived_at", "updated_at"])

    serializer = ChatSessionDetailSerializer(session)
    return success_response("Chat session restored successfully", serializer.data)


@api_view(["PATCH"])
@handle_exception
def rename_chat_session(request, session_id):
    new_title = request.data.get("title")
    if not new_title or not isinstance(new_title, str) or not new_title.strip():
        return error_response(
            "Failed to rename chat session",
            {"title": "A non-empty title string is required."},
            status.HTTP_400_BAD_REQUEST,
        )
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    session.title = new_title.strip()
    session.save()
    serializer = ChatSessionDetailSerializer(session)
    return success_response("Chat session renamed successfully", serializer.data)


@api_view(["GET"])
@handle_exception
def get_session_memos(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    messages = ChatMessage.objects.filter(session=session).order_by("sequence")

    memos = []
    for message in messages:
        document_references = []
        try:
            if getattr(message, "document_references", None):
                if isinstance(message.document_references, str):
                    try:
                        document_references = json.loads(message.document_references)
                    except Exception:
                        document_references = []
                else:
                    document_references = message.document_references or []
            elif message.answer:
                answer_data = message.answer
                if isinstance(answer_data, str):
                    answer_data = json.loads(answer_data)
                if isinstance(answer_data, dict) and "document_references" in answer_data:
                    document_references = answer_data.get("document_references", [])
        except (json.JSONDecodeError, ValueError, TypeError):
            document_references = []

        memos.append(
            {
                "id": str(message.id),
                "question": message.question,
                "answer": message.answer,
                "document_references": document_references,
                "sequence": message.sequence,
                "created_at": message.created_at.isoformat(),
            }
        )

    return success_response(
        "Session memos retrieved successfully",
        {
            "session_id": str(session_id),
            "session_title": session.title,
            "memos": memos,
        },
    )


@api_view(["POST"])
@handle_exception
def bump_chat_session(request, session_id):
    session = get_object_or_404(ChatSession, id=session_id, user=request.user)
    if session.is_archived:
        session.is_archived = False
        session.archived_at = None
    session.created_at = timezone.now()
    session.save(update_fields=["is_archived", "archived_at", "created_at", "updated_at"])
    serializer = ChatSessionDetailSerializer(session)
    return success_response("Chat session bumped successfully", serializer.data)


@api_view(["POST"])
@handle_exception
def enqueue_question(request):
    """
    Enqueue a question for async processing.
    
    Request body:
        - question: str (required)
        - file_type: str (optional)
        - doc_id: str (optional)
        - priority: int (optional, 1-10, lower = higher priority)
        
    Returns:
        job_id: str - use this to poll for results
    """
    from .utils import ask_question
    
    body = json.loads(request.body) if request.body else {}
    question = body.get("question")
    
    if not question:
        return error_response("Question is required", status.HTTP_400_BAD_REQUEST)
    
    result = ask_question(
        question=question,
        file_type=body.get("file_type"),
        user_token=str(request.user.id) if request.user else None,
        session_id=body.get("session_id"),
        doc_id=body.get("doc_id"),
        async_processing=True,
        priority=body.get("priority", 5),
    )
    
    return success_response("Question enqueued successfully", result)


@api_view(["GET"])
@handle_exception
def get_queue_job_status(request, job_id):
    """
    Get status of an enqueued job.
    
    Returns:
        status: pending | processing | completed | failed
        result: The answer if completed
        error: Error message if failed
    """
    from base.services.agent.rag.queue_manager import get_query_queue
    
    queue = get_query_queue()
    job = queue.get_job_status(job_id)
    
    if job is None:
        return error_response("Job not found", status.HTTP_404_NOT_FOUND)
    
    response_data = {
        "job_id": job.job_id,
        "status": job.status.value,
        "question": job.question[:100],
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }
    
    if job.status.value == "completed":
        response_data["result"] = job.result
    elif job.status.value == "failed":
        response_data["error"] = job.error
    
    return success_response("Job status retrieved", response_data)


@api_view(["GET"])
@handle_exception
def get_queue_stats(request):
    """
    Get queue statistics.
    """
    from base.services.agent.rag.queue_manager import get_query_queue
    
    queue = get_query_queue()
    stats = queue.get_queue_stats()
    
    return success_response("Queue stats retrieved", stats)


@api_view(["POST"])
@handle_exception
def cancel_queue_job(request, job_id):
    """
    Cancel a pending job.
    """
    from base.services.agent.rag.queue_manager import get_query_queue
    
    queue = get_query_queue()
    cancelled = queue.cancel_job(job_id)
    
    if not cancelled:
        return error_response(
            "Job not found or cannot be cancelled (already processing/completed)",
            status.HTTP_400_BAD_REQUEST
        )
    
    return success_response("Job cancelled successfully", {"job_id": job_id})


# ─────────────────────────────── Feedback endpoints ───────────────────────────

@api_view(["POST"])
@handle_exception
def submit_feedback(request):
    """
    Submit thumbs-up (rating=1) or thumbs-down (rating=-1) feedback for a message.

    Request body (JSON):
        message_id  : str  – UUID of the ChatMessage being rated
        rating      : int  – 1 (positive) or -1 (negative)
        feedback_text: str – optional text explanation (thumbs-down)
        question    : str  – question text for reference
        answer      : str  – answer text for reference

    Returns the created/updated feedback record id.
    One record per (message_id, user) — submitting again updates the existing one.
    """
    data = request.data
    message_id = data.get("message_id", "").strip()
    rating = data.get("rating")

    if rating not in (1, -1):
        return error_response(
            "Invalid rating. Must be 1 (thumbs up) or -1 (thumbs down).",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if not message_id:
        return error_response(
            "message_id is required.",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    # Resolve optional FK
    chat_message = None
    try:
        chat_message = ChatMessage.objects.get(id=message_id)
    except (ChatMessage.DoesNotExist, Exception):
        pass  # message may not exist in DB if using streaming/queue path

    user = request.user if request.user.is_authenticated else None
    user_name = ""
    user_email = ""
    if user:
        user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or getattr(user, "username", "")
        user_email = getattr(user, "email", "") or ""

    feedback_obj, created = MessageFeedback.objects.update_or_create(
        message_id_str=message_id,
        user=user,
        defaults={
            "message": chat_message,
            "rating": rating,
            "feedback_text": data.get("feedback_text", "").strip() or None,
            "question": data.get("question", "").strip() or None,
            "answer": data.get("answer", "").strip() or None,
            "user_name": user_name or None,
            "user_email": user_email or None,
        },
    )

    logger.info(
        f"Feedback {'created' if created else 'updated'}: "
        f"message={message_id} rating={rating} user={user}"
    )

    return success_response(
        "Feedback submitted successfully",
        {
            "id": str(feedback_obj.id),
            "rating": feedback_obj.rating,
            "created": created,
        },
        status_code=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
    )


@api_view(["GET"])
@handle_exception
def list_feedback(request):
    """
    List all feedback records. SuperAdmin access only.

    Query params:
        limit  : int (default 100)
        offset : int (default 0)
        rating : 1 | -1 (optional filter)
    """
    from user.models import User as UserModel  # local import to avoid circular

    # Permission check — only superadmin group
    user = request.user
    if not (user.is_authenticated and (
        user.is_staff or user.is_superuser or
        user.groups.filter(name__iexact="superadmin").exists()
    )):
        return error_response(
            "Permission denied.",
            status_code=status.HTTP_403_FORBIDDEN,
        )

    try:
        limit = int(request.query_params.get("limit", 100))
        offset = int(request.query_params.get("offset", 0))
    except (ValueError, TypeError):
        limit, offset = 100, 0

    qs = MessageFeedback.objects.all()

    rating_filter = request.query_params.get("rating")
    if rating_filter is not None:
        try:
            qs = qs.filter(rating=int(rating_filter))
        except ValueError:
            pass

    total = qs.count()
    items = qs[offset: offset + limit]

    results = [
        {
            "id": str(fb.id),
            "message_id": fb.message_id_str,
            "rating": fb.rating,
            "feedback_type": "positive" if fb.rating == 1 else "negative",
            "feedback_text": fb.feedback_text or "",
            "question": fb.question or "",
            "answer": fb.answer or "",
            "user_name": fb.user_name or "Unknown",
            "user_email": fb.user_email or "Unknown",
            "timestamp": fb.created_at.isoformat(),
        }
        for fb in items
    ]

    return success_response(
        "Feedback list retrieved",
        {"count": total, "results": results},
    )
