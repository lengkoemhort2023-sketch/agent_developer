import json
import logging
import re

from django.http import StreamingHttpResponse

from base.application.chat_queries import AnswerUserQueryHandler

logger = logging.getLogger(__name__)


def _is_no_info_response(response_data: dict) -> bool:
    """Detect the canonical no-relevant-information response."""
    if not isinstance(response_data, dict):
        return False

    no_info_messages = {
        "No relevant information found. Please try to ask in the Khmer language instead.",
        "រកមិនឃើញព័ត៌មានពាក់ព័ន្ធទេ។ សូមសាកល្បងសួរជាភាសាអង់គ្លេសជំនួសវិញ។",
        "No relevant information found.",
    }

    combined_answer = str(response_data.get("combined_answer", "") or "").strip()
    if combined_answer in no_info_messages:
        return True

    answers = response_data.get("answers", [])
    if isinstance(answers, list) and answers:
        first_answer = answers[0]
        if isinstance(first_answer, dict):
            answer_text = str(first_answer.get("text", "") or "").strip()
        else:
            answer_text = str(first_answer or "").strip()
        if answer_text in no_info_messages:
            return True

    message = str(response_data.get("message", "") or "").strip()
    return message in no_info_messages


def _retry_generate_suggestions(question: str, response_data: dict) -> list:
    """Fallback: directly retry suggestion generation when response_data has none."""
    try:
        from django.conf import settings as django_settings
        from base.services.agent.rag.config import FOLLOWUP_SUGGESTIONS_ENABLED, FOLLOWUP_SUGGESTIONS_COUNT

        if _is_no_info_response(response_data):
            logger.info("[RetrySuggestions] Skipping suggestion retry for no-info response")
            return []

        suggestions = response_data.get("suggestions", [])
        if suggestions and len(suggestions) > 0:
            return suggestions

        if not FOLLOWUP_SUGGESTIONS_ENABLED or FOLLOWUP_SUGGESTIONS_COUNT <= 0:
            return []

        combined_answer = response_data.get("combined_answer", "")
        answers = response_data.get("answers", [])

        top_answers = []
        for ans in (answers or [])[:5]:
            if isinstance(ans, dict):
                top_answers.append(ans)
            elif isinstance(ans, str):
                top_answers.append({"text": ans, "header": ""})

        if not top_answers and combined_answer:
            top_answers.append({"text": combined_answer[:3000], "header": ""})

        if not top_answers:
            return []

        from base.services.agent.rag.services.generative import ResponseGenerationService

        logger.info(
            f"[RetrySuggestions] Retrying suggestion generation with {len(top_answers)} answer(s)"
        )
        service = ResponseGenerationService()
        result = service._generate_followup_questions(
            question=question,
            top_answers=top_answers,
            count=FOLLOWUP_SUGGESTIONS_COUNT,
            language="en",
        )
        if result:
            logger.info(f"[RetrySuggestions] Generated {len(result)} suggestions on retry: {result}")
        else:
            logger.warning("[RetrySuggestions] Retry still returned empty suggestions")
        return result
    except Exception as e:
        logger.warning(f"[RetrySuggestions] Failed: {e}")
        return []


def tokenize_text(text: str):
    tokens = re.findall(r"\S+|\s+", text)
    return [token for token in tokens if token]


def stream_chat_message(request):
    """Streaming endpoint for chat messages using Server-Sent Events."""
    try:
        from .utils import ask_question

        answer_query_handler = AnswerUserQueryHandler(answer_fn=ask_question)
        question = request.GET.get("question", "").strip()
        session_id = request.GET.get("session_id")
        file_type = request.GET.get("file_type")
        doc_id = request.GET.get("doc_id")

        logger.info(
            f"Streaming request: question={question}..., file_type={file_type}, doc_id={doc_id}"
        )

        if not question:
            logger.error("No question provided for streaming")
            return StreamingHttpResponse(
                f"data: {json.dumps({'type': 'error', 'message': 'Question is required'})}\n\n",
                content_type="text/event-stream",
            )

        auth_header = request.META.get("HTTP_AUTHORIZATION", "")
        token = None

        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            logger.info("Token extracted from Authorization header")
        else:
            token = request.GET.get("token")
            if token:
                logger.warning(
                    "Token extracted from query parameter - this is deprecated and insecure"
                )

        if not token:
            logger.error("No authentication token provided for streaming")
            return StreamingHttpResponse(
                f"data: {json.dumps({'type': 'error', 'message': 'Authentication required'})}\n\n",
                content_type="text/event-stream",
            )

        from django.contrib.auth import get_user_model
        from rest_framework_simplejwt.tokens import AccessToken

        try:
            access_token = AccessToken(token)
            user_id = access_token["user_id"]
            User = get_user_model()
            user = User.objects.get(id=user_id)
            request.user = user
        except Exception as e:
            logger.error(f"Invalid authentication token: {e}")
            return StreamingHttpResponse(
                f"data: {json.dumps({'type': 'error', 'message': 'Invalid authentication token'})}\n\n",
                content_type="text/event-stream",
            )

        logger.info(
            f"Starting streaming response for question: {question}, file_type={file_type}, doc_id={doc_id}"
        )
        result = answer_query_handler.execute(
            user=request.user,
            question=question,
            user_token=token,
            session_id=session_id,
            file_type=file_type,
            doc_id=doc_id,
            input_type="text",
            missing_session_policy="create",
        )
        response_data = result.response_data
        session_id = str(result.session.id)
        logger.info(f"RAG response data keys: {list(response_data.keys()) if isinstance(response_data, dict) else type(response_data)}")

        # Ensure suggestions are populated; retry if empty
        suggestions = response_data.get("suggestions", [])
        if not _is_no_info_response(response_data) and (not suggestions or len(suggestions) == 0):
            fallback_suggestions = _retry_generate_suggestions(question, response_data)
            if fallback_suggestions:
                if isinstance(response_data, dict):
                    response_data["suggestions"] = fallback_suggestions
                logger.info(f"Applied {len(fallback_suggestions)} fallback suggestions to response_data")

        def generate_stream():
            try:
                combined_answer = response_data.get("combined_answer", "")
                answers = response_data.get("answers", [])

                if combined_answer:
                    full_text = combined_answer
                    logger.info(
                        f"Streaming combined answer with references, length: {len(full_text)}"
                    )
                elif answers and len(answers) > 0:
                    full_answer = answers[0]
                    full_text = full_answer.get("text", "")
                    logger.info(f"Streaming first answer, length: {len(full_text)}")
                else:
                    full_text = response_data.get(
                        "message", "No relevant information found."
                    )
                    logger.info("Streaming fallback message")

                try:
                    yield f"data: {json.dumps({'type': 'start'})}\n\n"
                except (GeneratorExit, IOError, OSError) as e:
                    logger.info(f"Client disconnected during streaming: {e}")
                    return

                has_markdown_table = "|" in full_text and full_text.count("|") > 4
                has_html_table = "<table" in full_text.lower()
                
                logger.info(f"Streaming content check: length={len(full_text)}, has_html_table={has_html_table}, has_markdown_table={has_markdown_table}")

                # has_html_table logic removed to fix streaming issues with div-wrapped tables
                # Treating HTML as regular text and streaming word-by-word works reliably
                # and avoids regex splitting issues or large chunk problems.
                
                if has_markdown_table and not has_html_table:
                    for line in full_text.split("\n"):
                        if line.strip():
                            yield (
                                f"data: {json.dumps({'type': 'answer', 'data': {'text': line + chr(10)}, 'is_partial': True})}\n\n"
                            )
                else:
                    # Generic word-based streaming for both plain text and HTML
                    # This ensures consistent delivery without complex regex splitting
                    words = re.findall(r"\S+\s*", full_text)
                    for i in range(0, len(words), 4):
                        chunk = "".join(words[i : i + 4])
                        if chunk:
                            yield (
                                f"data: {json.dumps({'type': 'answer', 'data': {'text': chunk}, 'is_partial': True})}\n\n"
                            )

                completion_data = {
                    "type": "answer",
                    "data": {"text": full_text},
                    "is_partial": False,
                    "session_id": session_id,
                    "answers": answers,
                    "combined_answer": combined_answer,
                    "document_references": response_data.get("document_references", []),
                    "suggestions": response_data.get("suggestions", []),
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except GeneratorExit:
                logger.info("Client disconnected - streaming cancelled by user")
            except (IOError, OSError) as e:
                logger.info(f"Client disconnected - streaming error: {e}")
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                try:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                except Exception:
                    pass

        response = StreamingHttpResponse(
            generate_stream(),
            content_type="text/event-stream",
        )
        response["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate"
        response["Pragma"] = "no-cache"
        response["Expires"] = "0"
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Headers"] = (
            "Cache-Control, Authorization, Content-Type"
        )
        response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        return response

    except Exception as e:
        logger.error(f"Error setting up streaming: {e}")
        return StreamingHttpResponse(
            f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n",
            content_type="text/event-stream",
        )
