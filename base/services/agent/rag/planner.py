"""
Query Planner for the Agentic RAG pipeline.

Responsibilities:
  - Detect query intent (lookup / comparison / summarization / policy / multi_hop)
  - Generate focused sub-queries for complex questions
  - Build language-aware Qdrant filter hints
  - Fall back gracefully to a heuristic plan when the LLM is unavailable
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class QueryPlan:
    """Structured output of the planner, consumed by the retriever."""

    #: One of the INTENT_TYPES below
    intent: str = "lookup"

    #: Additional focused queries derived from the original question.
    #: Non-empty only for "comparison" and "multi_hop" intents.
    sub_queries: List[str] = field(default_factory=list)

    #: Detected language of the question ("km" | "en" | "multi")
    language: str = "en"

    #: Qdrant language filter hint, e.g. {"language": ["km", "multi"]}
    filters: Dict[str, object] = field(default_factory=dict)

    #: True when the question is likely asking about tabular data
    needs_table: bool = False


# ── Planner ──────────────────────────────────────────────────────────────────

class QueryPlanner:
    """
    Analyzes a user question and produces a :class:`QueryPlan`.

    Uses the configured LLM (gemma3:27b) to classify intent and generate
    sub-queries.  Falls back to a lightweight heuristic if the LLM fails or
    returns invalid JSON.
    """

    INTENT_TYPES = ["lookup", "comparison", "summarization", "policy", "multi_hop"]

    def __init__(self, llm) -> None:
        self.llm = llm

    # ── Public API ────────────────────────────────────────────────────────────

    def plan(self, question: str, language: str) -> QueryPlan:
        """
        Analyze *question* and return a :class:`QueryPlan`.

        Args:
            question: The user's original query (km or en).
            language:  Detected language code ("km" | "en" | "multi").

        Returns:
            A :class:`QueryPlan` with intent, sub_queries, and filters populated.
        """
        try:
            return self._llm_plan(question, language)
        except Exception as exc:
            logger.warning(f"[Planner] LLM planning failed ({exc}), using heuristic plan")
            return self._heuristic_plan(question, language)

    # ── Private: LLM-based plan ───────────────────────────────────────────────

    def _llm_plan(self, question: str, language: str) -> QueryPlan:
        from .config import MAX_SUBQUERIES  # avoid circular import at module level

        prompt = (
            "You are a query analysis assistant for an internal document Q&A system.\n"
            "Analyze the user question and respond with ONLY valid JSON — no prose, no markdown.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "intent": "<lookup|comparison|summarization|policy|multi_hop>",\n'
            '  "sub_queries": ["<focused sub-question>", ...],\n'
            '  "needs_table": <true|false>\n'
            "}\n\n"
            "Intent definitions:\n"
            "  lookup        — simple fact or definition retrieval (most questions)\n"
            "  comparison    — comparing two or more items, policies, or procedures\n"
            "  summarization — requesting a summary or overview of a document/section\n"
            "  policy        — asking about rules, procedures, regulations, or guidelines\n"
            "  multi_hop     — requires connecting facts from multiple separate sections\n\n"
            "Rules:\n"
            f"  - sub_queries: generate ONLY for 'comparison' or 'multi_hop' intent, else []\n"
            f"  - sub_queries: maximum {MAX_SUBQUERIES} items\n"
            "  - sub_queries: write in the SAME language as the question\n"
            "  - needs_table: true if the question is likely answered by a table or list\n"
            "  - Respond with ONLY the JSON object — no other text\n\n"
            f"Question: {question}\n\n"
            "JSON:"
        )

        msg = self.llm.invoke([HumanMessage(content=prompt)])
        raw = (msg.content or "").strip()

        # Strip any accidental markdown code fence
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

        # Extract the first {...} block
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in LLM planner response: {raw[:300]}")

        data = json.loads(json_match.group())

        intent = data.get("intent", "lookup")
        if intent not in self.INTENT_TYPES:
            logger.warning(f"[Planner] Unknown intent '{intent}', defaulting to 'lookup'")
            intent = "lookup"

        raw_subqueries = data.get("sub_queries", [])
        if not isinstance(raw_subqueries, list):
            raw_subqueries = []

        from .config import MAX_SUBQUERIES as _MAX
        sub_queries = [
            q.strip() for q in raw_subqueries
            if isinstance(q, str) and q.strip() and q.strip().lower() != question.strip().lower()
        ][:_MAX]

        needs_table = bool(data.get("needs_table", False))
        filters = self._language_filters(language)

        logger.info(
            f"[Planner] intent={intent} sub_queries={sub_queries} "
            f"needs_table={needs_table} filters={filters}"
        )
        return QueryPlan(
            intent=intent,
            sub_queries=sub_queries,
            language=language,
            filters=filters,
            needs_table=needs_table,
        )

    # ── Private: heuristic fallback plan ─────────────────────────────────────

    def _heuristic_plan(self, question: str, language: str) -> QueryPlan:
        """
        Lightweight keyword-based intent detection used when the LLM is unavailable.
        Always produces an empty sub_queries list.
        """
        q = question.lower()

        if any(w in q for w in ["compare", "difference", "vs", "versus",
                                  "ផ្ទឹម", "ខុសគ្នា", "ប្រៀបធៀប"]):
            intent = "comparison"
        elif any(w in q for w in ["summary", "summarize", "overview", "outline",
                                   "សង្ខេប", "ស្រង់", "ពិពណ៌នា"]):
            intent = "summarization"
        elif any(w in q for w in ["policy", "procedure", "regulation", "rule", "guideline",
                                   "គោលការណ៍", "នីតិវិធី", "លក្ខន្តិកៈ", "ច្បាប់"]):
            intent = "policy"
        else:
            intent = "lookup"

        logger.info(f"[Planner] Heuristic plan: intent={intent}")
        return QueryPlan(
            intent=intent,
            sub_queries=[],
            language=language,
            filters=self._language_filters(language),
            needs_table=False,
        )

    # ── Private: helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _language_filters(language: str) -> Dict[str, List[str]]:
        """Return the Qdrant language filter values for the detected language."""
        if language == "km":
            return {"language": ["km", "multi"]}
        if language == "en":
            return {"language": ["en", "multi"]}
        return {}   # "multi" or unknown → no language restriction
