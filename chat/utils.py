from base.services.audio.setup import allowed_file, audio_prompt_response
from django.http import JsonResponse
import os
from base.services.agent.rag.providers import RagProviders
from base.services.agent.rag.queue_manager import enqueue_query, get_query_queue, get_query_result


def ask_question(question, file_type=None, user_token=None, session_id=None, doc_id=None, async_processing=False, priority=5):
    """
    Process a question through the RAG pipeline.
    
    Args:
        question: User's question
        file_type: Optional document type filter
        user_token: User authentication token
        session_id: Chat session ID
        doc_id: Specific document to search
        async_processing: If True, enqueue for async processing and return job_id
        priority: Lower number = higher priority (1 = highest), only for async
        
    Returns:
        If async_processing=False: Answer dict (current behavior)
        If async_processing=True: Dict with job_id for polling
    """
    if async_processing:
        job_id = enqueue_query(
            question=question,
            file_type=file_type,
            user_token=user_token,
            session_id=session_id,
            doc_id=doc_id,
            priority=priority,
        )
        return {"job_id": job_id, "status": "queued"}
    
    # Original synchronous behavior
    rag = RagProviders()
    if file_type:
        answer = rag.generative_service.response(
            question=question,
            file_type=file_type,
            session_id=session_id,
            doc_id=doc_id,
        )
    else:
        answer = rag.generative_service.response(
            question=question,
            session_id=session_id,
            doc_id=doc_id,
        )
    return answer

def audio_processing(file_path):
    print(f"[AUDIO_PROCESSING] Starting with file: {file_path}")
    
    if not allowed_file(file_path):
        print(f"[AUDIO_PROCESSING] File extension not allowed: {file_path}")
        return JsonResponse({"error": "Invalid file format"}, status=400)

    if not os.path.exists(file_path):
        print(f"[AUDIO_PROCESSING] File not found: {file_path}")
        return JsonResponse({"error": "Uploaded audio file could not be read"}, status=400)

    print("[AUDIO_PROCESSING] File extension allowed")
    try:
        print("[AUDIO_PROCESSING] Calling audio_prompt_response")
        transcription_text = audio_prompt_response(file_path)

        if not transcription_text:
            raise ValueError(
                "No speech detected in the audio. Please speak more clearly and try again."
            )

        if transcription_text.startswith("Error"):
            raise Exception(
                f"Transcription failed or returned error: {transcription_text}"
            )

        print("Transcription: ", transcription_text)
        return transcription_text

    except Exception as e:
        error_message = str(e) or "Audio processing failed."
        lowered = error_message.lower()
        status_code = 400
        if any(keyword in lowered for keyword in ["unavailable", "model", "corrupted", "not installed"]):
            status_code = 503

        print(f"Error in audio processing: {error_message}")
        import traceback
        traceback.print_exc()
        return JsonResponse(
            {"error": error_message},
            status=status_code,
        )




