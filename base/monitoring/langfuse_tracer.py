"""
Langfuse tracing helpers for the RAG pipeline.

Provides safe wrappers around langfuse.decorators that gracefully degrade when
Langfuse is disabled (LANGFUSE_ENABLED=false) or the package is missing.

Coverage map (monitor_langfuse.md):
  A. LLM Performance   → observe decorators on llm_content / _synthesize_answer /
                          _generate_followup_questions / _clean_and_rephrase_with_llm
  B. RAG Retrieval     → observe decorator on get_related_docs
  C. Response Quality  → confidence score logged from AgenticRAG; user_feedback score
  D. Pipeline Flow     → observe decorator on response() as root trace; nested spans
  E. Failures          → exceptions recorded by @observe automatically
  F. Perf Breakdown    → phase timings stored in root trace metadata
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ENABLED: bool = os.environ.get("LANGFUSE_ENABLED", "false").lower() in {"true", "1", "yes"}

# ── Trace-id cache: question_id → langfuse trace_id (for post-hoc scoring) ───
_TRACE_CACHE: "OrderedDict[str, str]" = OrderedDict()
_TRACE_CACHE_MAX = 2000


# ── Public helpers ─────────────────────────────────────────────────────────────

def register_trace(question_id: str, trace_id: Optional[str]) -> None:
    """Store the Langfuse trace_id keyed by question_id for later feedback scoring."""
    if not trace_id:
        return
    _TRACE_CACHE[question_id] = trace_id
    while len(_TRACE_CACHE) > _TRACE_CACHE_MAX:
        _TRACE_CACHE.popitem(last=False)


def get_trace_id(question_id: str) -> Optional[str]:
    return _TRACE_CACHE.get(question_id)


def update_current_observation(**kwargs: Any) -> None:
    """Update the innermost active Langfuse span."""
    if not _ENABLED:
        return
    try:
        from langfuse.decorators import langfuse_context
        langfuse_context.update_current_observation(**kwargs)
    except Exception:
        pass


def update_current_trace(**kwargs: Any) -> None:
    """Update the root Langfuse trace (session_id, user_id, tags, metadata…)."""
    if not _ENABLED:
        return
    try:
        from langfuse.decorators import langfuse_context
        langfuse_context.update_current_trace(**kwargs)
    except Exception:
        pass


def get_current_trace_id() -> Optional[str]:
    if not _ENABLED:
        return None
    try:
        from langfuse.decorators import langfuse_context
        return langfuse_context.get_current_trace_id()
    except Exception:
        return None


def score_trace(trace_id: str, name: str, value: float, comment: Optional[str] = None) -> None:
    """Post a score to a completed Langfuse trace (used by user_feedback)."""
    if not _ENABLED or not trace_id:
        return
    try:
        from langfuse import Langfuse
        Langfuse().score(trace_id=trace_id, name=name, value=value, comment=comment)
    except Exception as exc:
        logger.debug("[Langfuse] score_trace failed (non-fatal): %s", exc)


def extract_token_usage(response: Any) -> Dict[str, int]:
    """
    Extract token counts from a LangChain / Ollama LLM response.

    Supports:
    - LangChain 0.2+ ``usage_metadata`` dict
    - Ollama ``response_metadata`` with ``prompt_eval_count`` / ``eval_count``
    """
    try:
        meta = getattr(response, "usage_metadata", None)
        if meta:
            if isinstance(meta, dict):
                inp = meta.get("input_tokens", 0) or 0
                out = meta.get("output_tokens", 0) or 0
            else:
                inp = getattr(meta, "input_tokens", 0) or 0
                out = getattr(meta, "output_tokens", 0) or 0
            return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

        rmeta: dict = getattr(response, "response_metadata", {}) or {}
        inp = rmeta.get("prompt_eval_count", 0) or 0
        out = rmeta.get("eval_count", 0) or 0
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
    except Exception:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Cost in USD. Defaults to $0 for self-hosted Ollama.
    Set LANGFUSE_COST_INPUT_PER_1K / LANGFUSE_COST_OUTPUT_PER_1K to override.
    """
    try:
        cost_in = float(os.environ.get("LANGFUSE_COST_INPUT_PER_1K", "0")) / 1000
        cost_out = float(os.environ.get("LANGFUSE_COST_OUTPUT_PER_1K", "0")) / 1000
        return input_tokens * cost_in + output_tokens * cost_out
    except Exception:
        return 0.0
