import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from django.utils import timezone

from chat.models import ChatInput, ChatMessage, ChatSession


def serialize_bot_response(bot_response: Any) -> str:
    """Persist a normalized answer shape so sync and stream paths match."""
    try:
        if isinstance(bot_response, dict):
            answers = bot_response.get("answers", [])
            cleaned_answers = []
            for answer in answers:
                cleaned_answer = dict(answer)
                if "document_reference" in cleaned_answer and isinstance(
                    cleaned_answer["document_reference"], dict
                ):
                    cleaned_answer["document_reference"] = {
                        key: value
                        for key, value in cleaned_answer["document_reference"].items()
                        if key != "html_card"
                    }
                cleaned_answers.append(cleaned_answer)

            clean_refs = []
            if "document_references" in bot_response:
                clean_refs = [
                    {key: value for key, value in ref.items() if key != "html_card"}
                    for ref in bot_response["document_references"]
                ]

            per_answer_refs = []
            for answer in cleaned_answers:
                doc_ref = None
                if isinstance(answer, dict):
                    if "document_reference" in answer and isinstance(
                        answer.get("document_reference"), dict
                    ):
                        doc_ref = answer.get("document_reference")
                    elif "document_references" in answer and isinstance(
                        answer.get("document_references"), list
                    ):
                        refs = answer.get("document_references") or []
                        if refs and isinstance(refs[0], dict):
                            doc_ref = refs[0]
                if doc_ref:
                    per_answer_refs.append(
                        {
                            key: value
                            for key, value in doc_ref.items()
                            if key != "html_card"
                        }
                    )

            answer_to_save = {
                "answers": cleaned_answers,
                "combined_answer": bot_response.get("combined_answer", ""),
                "document_references": per_answer_refs if per_answer_refs else clean_refs,
                "document_references_aggregated": clean_refs,
                "language": bot_response.get("language", "en"),
                "suggestions": bot_response.get("suggestions", []),
            }
            return json.dumps(answer_to_save)
        return json.dumps(bot_response) if isinstance(bot_response, (list, dict)) else str(bot_response)
    except Exception:
        return json.dumps(bot_response) if isinstance(bot_response, (list, dict)) else str(bot_response)


@dataclass
class QueryExecutionResult:
    session: ChatSession
    message: ChatMessage
    chat_input: ChatInput
    response_data: Any
    question: str
    created_session: bool


class AnswerUserQueryHandler:
    """Shared orchestration for synchronous and streaming query execution."""

    def __init__(self, answer_fn: Callable[..., Any]):
        self.answer_fn = answer_fn

    def execute(
        self,
        *,
        user,
        question: str,
        user_token: Optional[str],
        session_id: Optional[str] = None,
        file_type: Optional[str] = None,
        doc_id: Optional[str] = None,
        input_type: str = "text",
        voice_file=None,
        missing_session_policy: str = "error",
    ) -> QueryExecutionResult:
        session, created_session = self._resolve_session(
            user=user,
            question=question,
            session_id=session_id,
            missing_session_policy=missing_session_policy,
        )

        chat_input = ChatInput.objects.create(
            user=user,
            input_type=input_type,
            content=question,
            voice_file=voice_file if input_type == "voice" else None,
        )

        bot_response = self.answer_fn(
            question=question,
            file_type=file_type,
            user_token=user_token,
            session_id=str(session.id),
            doc_id=doc_id,
        )

        message = ChatMessage.objects.create(
            session=session,
            question=question,
            answer=serialize_bot_response(bot_response),
            document_references=bot_response.get("document_references", [])
            if isinstance(bot_response, dict)
            else [],
            suggestions=bot_response.get("suggestions", [])
            if isinstance(bot_response, dict)
            else [],
        )

        chat_input.message = message
        chat_input.processed_at = timezone.now()
        chat_input.save(update_fields=["message", "processed_at"])

        if question and (not session.title or session.title == "New Chat"):
            session.title = question[:50]
            session.save(update_fields=["title", "updated_at"])

        return QueryExecutionResult(
            session=session,
            message=message,
            chat_input=chat_input,
            response_data=bot_response,
            question=question,
            created_session=created_session,
        )

    def _resolve_session(
        self,
        *,
        user,
        question: str,
        session_id: Optional[str],
        missing_session_policy: str,
    ) -> tuple[ChatSession, bool]:
        if not session_id:
            session = ChatSession.get_last_empty(user=user)
            if session:
                return session, False
            return ChatSession.objects.create(user=user, title=question[:50]), True

        try:
            return ChatSession.objects.get(id=session_id, user=user), False
        except ChatSession.DoesNotExist:
            if missing_session_policy == "create":
                return ChatSession.objects.create(user=user, title=question[:50]), True
            raise
