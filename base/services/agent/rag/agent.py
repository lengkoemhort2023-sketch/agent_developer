"""
Agentic RAG Orchestrator.

Full pipeline
─────────────
1. Language gate    Detect query language; reject if not km / en / multi.
2. Plan             LLM-based intent classification + sub-query generation.
3. Multi-step retrieve
                    Pass A  (broad, top_k=40)
                    Pass B  (sub-queries, merged + deduped)
                    Pass C  (rerank → top 15)
4. Retry            If retrieval is empty, retry with a broader (no-subquery) plan.
5. Confidence gate  Compute retrieval confidence; return {} below threshold.
6. Convert          Translate ranked chunks into the dict format expected by
                    ResponseGenerationService.response() — identical to the
                    output of get_related_docs().

Integration with generative.py
────────────────────────────────
ResponseGenerationService.__init__() creates an AgenticRAG instance and stores
it as self.agentic_rag.  In response(), the single line:

    related_docs = self.get_related_docs(...)

is replaced with:

    related_docs = self.agentic_rag.get_docs(...)  # if enabled
    # else falls back to self.get_related_docs(...)

The return value is 100 % compatible with the existing downstream processing
(llm_content routing, follow-up question generation, history saving, etc.).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from base.monitoring.metrics import (
    rag_chunks_returned,
    rag_confidence_score,
    rag_request_duration_seconds,
    rag_requests_total,
    rag_retrieval_attempts_total,
)
from base.monitoring.langfuse_tracer import update_current_observation
from base.tracing import get_tracer
from langfuse.decorators import observe

from .config import (
    AGENTIC_RAG_ENABLED,
    CONFIDENCE_THRESHOLD_EN,
    CONFIDENCE_THRESHOLD_KM,
    MAX_RETRIES,
    SUPPORTED_LANGUAGES,
)
from .planner import QueryPlan, QueryPlanner
from .retriever import AgenticRetriever
from .tools import RAGTools

logger = logging.getLogger(__name__)


class AgenticRAG:
    """
    Agentic RAG orchestrator that replaces single-pass retrieval with a
    planned, multi-step, confidence-gated pipeline.

    The public entry point is :meth:`get_docs`, which returns a dict in the
    same format as ``ResponseGenerationService.get_related_docs()``.
    """

    def __init__(self, vector_service, llm) -> None:
        """
        Args:
            vector_service: Initialized :class:`VectorStoreService`.
            llm:            Initialized LangChain ChatOllama (or compatible).
        """
        self.llm = llm
        self.tools = RAGTools(vector_service, llm)
        self.planner = QueryPlanner(llm)
        self.retriever = AgenticRetriever(self.tools)
        logger.info("[AgenticRAG] Initialized (planner + tools + retriever)")

    # ── Public API ────────────────────────────────────────────────────────────

    @observe(name="rag.get_docs", capture_input=True)
    def get_docs(
        self,
        question: str,
        collection_name: Optional[str],
        query_language: Optional[str],
        doc_id: Optional[str] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Run the full agentic pipeline and return a combined_chunks dict.

        Args:
            question:        User's query (already refined by response()).
            collection_name: Qdrant collection name (= file_type), or None.
            query_language:  Qdrant language filter ("km" | "en" | None).
            doc_id:          Restrict retrieval to a single document UUID.
            language:        Detected language code ("km" | "en" | "multi").

        Returns:
            A ``combined_chunks`` dict identical to
            ``ResponseGenerationService.get_related_docs()``'s return value,
            or ``{}`` when confidence is too low / language is unsupported /
            no results are found.

        Combined_chunks format::

            {
                "<doc_source_uuid>": {
                    "file_name":     str,
                    "combined_text": str,
                    "chunks":        [<chunk_dicts compatible with response()>],
                    "max_score":     float,
                }
            }
        """
        if doc_id and collection_name:
            logger.info(
                f"[AgenticRAG] doc-scoped retrieval for doc_id={doc_id}; ignoring collection hint '{collection_name}' and searching across collections"
            )
            collection_name = None

        logger.info(
            f"[AgenticRAG] get_docs: question='{question[:80]}' "
            f"lang={language} collection={collection_name} doc_id={doc_id}"
        )

        _start = time.perf_counter()
        _tracer = get_tracer("docbot.rag")

        with _tracer.start_as_current_span("rag.get_docs") as span:
            span.set_attribute("rag.language", language)
            span.set_attribute("rag.question_len", len(question))
            span.set_attribute("rag.collection", collection_name or "all")

            # ── 1. Language gate ──────────────────────────────────────────────
            if language not in SUPPORTED_LANGUAGES:
                logger.warning(f"[AgenticRAG] Unsupported language '{language}' — rejecting")
                rag_requests_total.labels(language=language, status="language_rejected").inc()
                span.set_attribute("rag.status", "language_rejected")
                return {}

            # ── 2. Plan ───────────────────────────────────────────────────────
            with _tracer.start_as_current_span("rag.plan") as plan_span:
                plan: QueryPlan = self.planner.plan(question, language)
                plan_span.set_attribute("rag.intent", plan.intent or "")
                plan_span.set_attribute("rag.sub_queries_count", len(plan.sub_queries))
            logger.info(
                f"[AgenticRAG] Plan: intent={plan.intent}  "
                f"sub_queries={plan.sub_queries}  needs_table={plan.needs_table}"
            )

            # ── 3+4. Multi-step retrieval with retry ──────────────────────────
            top_chunks = self._retrieve_with_retry(
                question=question,
                plan=plan,
                collection_name=collection_name,
                doc_id=doc_id,
                language=language,
            )

            if not top_chunks:
                logger.warning("[AgenticRAG] No chunks after all retrieval attempts")
                rag_requests_total.labels(language=language, status="empty").inc()
                rag_request_duration_seconds.labels(language=language).observe(time.perf_counter() - _start)
                span.set_attribute("rag.status", "empty")
                return {}

            # ── 5. Confidence gate ────────────────────────────────────────────
            threshold = CONFIDENCE_THRESHOLD_KM if language == "km" else CONFIDENCE_THRESHOLD_EN
            with _tracer.start_as_current_span("rag.confidence") as conf_span:
                confidence = self.tools.confidence_score(question, top_chunks)
                conf_span.set_attribute("rag.confidence", round(confidence, 4))
                conf_span.set_attribute("rag.threshold", threshold)
            rag_confidence_score.labels(language=language).observe(confidence)

            if confidence < threshold:
                logger.warning(
                    f"[AgenticRAG] Confidence {confidence:.4f} < threshold {threshold:.4f} "
                    f"— returning empty (no-info path)"
                )
                rag_requests_total.labels(language=language, status="confidence_fail").inc()
                rag_request_duration_seconds.labels(language=language).observe(time.perf_counter() - _start)
                span.set_attribute("rag.status", "confidence_fail")
                update_current_observation(
                    metadata={
                        "status": "confidence_fail",
                        "confidence": round(confidence, 4),
                        "confidence_threshold": threshold,
                        "intent": plan.intent,
                        "sub_queries_count": len(plan.sub_queries),
                        "language": language,
                    },
                    level="WARNING",
                )
                return {}

            logger.info(f"[AgenticRAG] Confidence {confidence:.4f} ≥ {threshold:.4f} — proceeding")

            # ── 6. Convert to get_related_docs() format ───────────────────────
            result = self._to_combined_docs(top_chunks)
            total_chunks = sum(len(d["chunks"]) for d in result.values())
            rag_chunks_returned.labels(language=language).observe(total_chunks)
            rag_requests_total.labels(language=language, status="success").inc()
            rag_request_duration_seconds.labels(language=language).observe(time.perf_counter() - _start)
            span.set_attribute("rag.status", "success")
            span.set_attribute("rag.chunks_returned", total_chunks)
            span.set_attribute("rag.docs_returned", len(result))
            update_current_observation(
                metadata={
                    "status": "success",
                    "intent": plan.intent,
                    "sub_queries": plan.sub_queries,
                    "sub_queries_count": len(plan.sub_queries),
                    "confidence": round(confidence, 4),
                    "confidence_threshold": threshold,
                    "language": language,
                    "docs_returned": len(result),
                    "chunks_returned": total_chunks,
                    "top_docs": [
                        {"name": meta["file_name"], "score": round(meta["max_score"], 4)}
                        for meta in list(result.values())[:5]
                    ],
                },
            )
            return result

    # ── Private: retrieval with retry ─────────────────────────────────────────

    def _retrieve_with_retry(
        self,
        question: str,
        plan: QueryPlan,
        collection_name: Optional[str],
        doc_id: Optional[str],
        language: str = "en",
    ) -> List[dict]:
        """
        Call :meth:`AgenticRetriever.retrieve` up to ``MAX_RETRIES + 1`` times.

        On the first empty result, retries with a simplified plan
        (sub_queries cleared) for a broader search.
        """
        current_plan = plan
        attempts = MAX_RETRIES + 1

        for attempt in range(attempts):
            rag_retrieval_attempts_total.labels(language=language).inc()
            chunks = self.retriever.retrieve(
                question=question,
                plan=current_plan,
                collection_name=collection_name,
                doc_id=doc_id,
            )

            if chunks:
                if attempt > 0:
                    logger.info(f"[AgenticRAG] Retrieval succeeded on attempt {attempt + 1}")
                return chunks

            # Still empty — simplify plan for the next attempt
            if attempt < attempts - 1:
                logger.info(
                    f"[AgenticRAG] Empty retrieval on attempt {attempt + 1}, "
                    f"retrying with broader plan"
                )
                # Build a broader plan: keep language filters, drop sub-queries
                from dataclasses import replace as dc_replace
                current_plan = dc_replace(current_plan, sub_queries=[])

        logger.warning(f"[AgenticRAG] All {attempts} retrieval attempts returned 0 chunks")
        return []

    # ── Private: format conversion ────────────────────────────────────────────

    @staticmethod
    def _to_combined_docs(chunks: List[dict]) -> Dict[str, Any]:
        """
        Convert a flat list of reranked chunk dicts into the
        ``combined_chunks`` format produced by
        ``ResponseGenerationService.get_related_docs()``.

        This format is consumed directly by the downstream processing in
        ``response()`` including:
          - Score thresholding
          - Django active-document filter
          - llm_content routing (header == "Content")
          - Follow-up question generation
          - History saving

        Chunk fields preserved (exactly as response() expects):

        .. code-block:: python

            {
                "id":          int     # hash(chunk_id) % 10000
                "content":     str     # clean body text
                "page_content":str     # full markitdown text (may have "# Header" prefix)
                "page":        int     # page_number
                "chunk_index": int
                "header":      str     # section heading (key for llm_content routing)
                "file_name":   str
                "score":       float   # _rerank_score if available, else RRF score
                "has_table":   bool
            }
        """
        doc_map: Dict[str, Dict[str, Any]] = {}

        for chunk in chunks:
            source = chunk.get("source") or chunk.get("chunk_id", "unknown")

            if source not in doc_map:
                doc_map[source] = {
                    "file_name": chunk.get("file_name", "Document"),
                    "combined_text": "",
                    "chunks": [],
                    "max_score": 0.0,
                }

            rerank_score = float(chunk.get("_rerank_score") or 0.0)
            retrieval_score = float(chunk.get("score") or 0.0)
            # Keep the stronger signal for downstream user-visible filtering.
            chunk_score = max(rerank_score, retrieval_score)

            normalized = {
                "id": hash(chunk.get("chunk_id", "") or source) % 10000,
                "content": chunk.get("content") or chunk.get("page_content", ""),
                "page_content": chunk.get("page_content") or chunk.get("content", ""),
                "html": chunk.get("html", ""),
                "page": int(chunk.get("page_number") or 1),
                "chunk_index": int(chunk.get("chunk_index") or 0),
                "header": chunk.get("header", ""),
                "file_name": chunk.get("file_name", "Document"),
                "score": chunk_score,
                "has_table": bool(chunk.get("has_table", False)),
            }

            doc_map[source]["chunks"].append(normalized)
            doc_map[source]["max_score"] = max(
                doc_map[source]["max_score"], chunk_score
            )

        # Sort chunks within each document by position
        for doc in doc_map.values():
            doc["chunks"].sort(key=lambda c: (c["chunk_index"], c["id"]))
            doc["combined_text"] = "\n\n".join(
                f"{c['header']}\n{c['content']}" for c in doc["chunks"]
            )

        # Sort documents by max_score descending
        sorted_docs = dict(
            sorted(doc_map.items(), key=lambda kv: kv[1]["max_score"], reverse=True)
        )

        logger.info(
            f"[AgenticRAG] Returning {len(sorted_docs)} doc(s)  "
            + (
                f"top_score={next(iter(sorted_docs.values()))['max_score']:.4f}"
                if sorted_docs
                else "no docs"
            )
        )
        return sorted_docs
