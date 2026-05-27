"""
Tool layer for the Agentic RAG pipeline.

Strictly defined tools consumed by the retriever and agent:

  hybrid_search(query, top_k, collection_name, filters)
    → Run hybrid dense+sparse search via VectorStoreService.

  rerank(query, hits)
    → Re-score chunks with RRF score + keyword overlap, sort descending.

  get_chunk(chunk_id, collection_name)
    → Fetch one specific chunk by its chunk_id.

  summarize_table(chunk, question)
    → Extract a markdown table from page_content and answer with LLM.

  confidence_score(query, retrieved_chunks)
    → Compute a 0–1 confidence score for the retrieval quality.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Dict, List, Optional

import torch
from langchain_core.messages import HumanMessage

try:
    from FlagEmbedding import FlagReranker as _FlagReranker
    _RERANKER_AVAILABLE = True
except ImportError:
    _FlagReranker = None
    _RERANKER_AVAILABLE = False

# Global singleton cache for the reranker model
_GLOBAL_RERANKER = None
_GLOBAL_RERANKER_LOCK = None

try:
    import threading
    _GLOBAL_RERANKER_LOCK = threading.Lock()
except ImportError:
    pass

logger = logging.getLogger(__name__)
# Suppress the noisy tokenizer regex warning from transformers/tokenizers
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class RAGTools:
    """
    Thin tool wrappers that bridge the Agentic RAG pipeline to the underlying
    VectorStoreService (Qdrant) and LLM.

    All methods are synchronous to stay compatible with Django's WSGI stack.
    """

    def __init__(self, vector_service, llm) -> None:
        """
        Args:
            vector_service: An initialized :class:`VectorStoreService` instance.
            llm:            An initialized LangChain ChatOllama (or compatible) instance.
        """
        self.vs = vector_service
        self.llm = llm
        self._reranker = self._load_reranker()

    # ── Reranker loader ──────────────────────────────────────────────────────

    def _load_reranker(self):
        """
        Load the bge-reranker-v2-m3 cross-encoder model (singleton).

        Uses FP16 when a CUDA GPU is available (RTX 5090), otherwise FP32.
        Returns None on failure — the heuristic fallback is used instead.
        """
        global _GLOBAL_RERANKER, _GLOBAL_RERANKER_LOCK
        
        # Fast path if already loaded
        if _GLOBAL_RERANKER is not None:
            return _GLOBAL_RERANKER

        from .config import RERANKER_MODEL_PATH
        if not _RERANKER_AVAILABLE:
            logger.warning(
                "[Tools] FlagEmbedding not installed — reranker will use "
                "heuristic scoring. Install with: pip install FlagEmbedding"
            )
            return None

        # Slow path with lock
        if _GLOBAL_RERANKER_LOCK:
            with _GLOBAL_RERANKER_LOCK:
                if _GLOBAL_RERANKER is not None:
                    return _GLOBAL_RERANKER
                
                # FlagReranker auto-selects device (CUDA → MPS → CPU); overrides use_fp16=False on CPU
                reranker = None
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
                        warnings.filterwarnings("ignore", message=".*fix_mistral_regex.*")
                        warnings.filterwarnings("ignore", message=".*tokenizer you are loading.*")
                        reranker = _FlagReranker(RERANKER_MODEL_PATH, use_fp16=True)
                    logger.info(f"[Tools] bge-reranker-v2-m3 loaded from {RERANKER_MODEL_PATH} (device={reranker.device})")
                except Exception as exc:
                    logger.warning(
                        f"[Tools] Could not load bge-reranker-v2-m3 ({exc}). "
                        "Falling back to heuristic reranking."
                    )
                    return None

                _GLOBAL_RERANKER = reranker
                return _GLOBAL_RERANKER
        return None

    # ── Tool 1: hybrid_search ────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int,
        collection_name: Optional[str],
        filters: Dict,
    ) -> List[dict]:
        """
        Run hybrid dense+sparse search against one Qdrant collection (or all
        active collections when *collection_name* is None/empty).

        The ``is_active=True`` filter and language filter are applied at the
        Qdrant level inside ``VectorStoreService.query_collection_hybrid()``.

        Args:
            query:           User query string.
            top_k:           Maximum number of results per collection.
            collection_name: Target collection name, or None to search all.
            filters:         Optional extra filter hints, e.g.
                             ``{"language": ["km", "multi"]}``.

        Returns:
            List of standardized chunk dicts (see :meth:`_point_to_chunk`).
        """
        try:
            query_language = self._lang_from_filters(filters)
            source_filter = filters.get("source") if filters else None

            if collection_name:
                points = self.vs.query_collection_hybrid(
                    collection_name=collection_name,
                    query=query,
                    k=top_k,
                    query_language=query_language,
                    source_filter=source_filter,
                )
                return [self._point_to_chunk(p, collection_name) for p in points if p]

            # Search all active collections
            return self._search_all(query, top_k, query_language, source_filter=source_filter)

        except Exception as exc:
            logger.error(f"[Tools] hybrid_search error: {exc}")
            return []

    # ── Tool 2: rerank ────────────────────────────────────────────────────────

    def rerank(self, query: str, hits: List[dict]) -> List[dict]:
        """
        Re-score retrieved chunks using the bge-reranker-v2-m3 cross-encoder.

        Each (query, chunk_text) pair is scored by the cross-encoder.
        ``normalize=True`` applies sigmoid so every score is in [0, 1].

        Falls back to a lightweight heuristic when the model is unavailable:
            score = 0.70 × rrf_norm + 0.30 × keyword_overlap

        The ``_rerank_score`` key is added to every chunk dict so downstream
        consumers can inspect or log it.

        Args:
            query: Original user query.
            hits:  List of chunk dicts from :meth:`hybrid_search`.

        Returns:
            The same list sorted by ``_rerank_score`` descending.
        """
        if not hits:
            return []

        if self._reranker is not None:
            return self._rerank_with_model(query, hits)
        return self._rerank_heuristic(query, hits)

    def _rerank_with_model(self, query: str, hits: List[dict]) -> List[dict]:
        """Cross-encoder reranking using bge-reranker-v2-m3."""
        pairs = [
            [query, chunk.get("page_content") or chunk.get("content") or ""]
            for chunk in hits
        ]
        try:
            # normalize=True applies sigmoid → score in [0, 1]
            raw_scores = self._reranker.compute_score(pairs, normalize=True)
            # compute_score returns a float when len(pairs)==1, else a list
            if isinstance(raw_scores, float):
                raw_scores = [raw_scores]

            scored = [
                {**chunk, "_rerank_score": round(float(score), 6)}
                for chunk, score in zip(hits, raw_scores)
            ]
            scored.sort(key=lambda c: c["_rerank_score"], reverse=True)

            logger.info(
                f"[Tools] rerank (cross-encoder): {len(scored)} chunks → "
                f"top _rerank_score={scored[0]['_rerank_score']:.4f}"
            )
            return scored
        except Exception as exc:
            logger.warning(f"[Tools] Cross-encoder rerank failed ({exc}), using heuristic fallback.")
            return self._rerank_heuristic(query, hits)

    def _rerank_heuristic(self, query: str, hits: List[dict]) -> List[dict]:
        """Lightweight heuristic reranking: 0.70×RRF_norm + 0.30×keyword_overlap."""
        query_tokens = set(re.sub(r"\s+", " ", query.lower()).split())
        max_score = max((c.get("score") or 0) for c in hits) or 1.0

        scored: List[dict] = []
        for chunk in hits:
            rrf_norm = (chunk.get("score") or 0) / max_score
            text = (chunk.get("page_content") or chunk.get("content") or "").lower()
            text_tokens = set(re.sub(r"\s+", " ", text).split())
            overlap = min(len(query_tokens & text_tokens) / (len(query_tokens) + 1e-9), 1.0)
            final = round(0.70 * rrf_norm + 0.30 * overlap, 6)
            scored.append({**chunk, "_rerank_score": final})

        scored.sort(key=lambda c: c["_rerank_score"], reverse=True)
        if scored:
            logger.info(
                f"[Tools] rerank (heuristic): {len(scored)} chunks → "
                f"top _rerank_score={scored[0]['_rerank_score']:.4f}"
            )
        return scored

    # ── Tool 3: get_chunk ─────────────────────────────────────────────────────

    def get_chunk(self, chunk_id: str, collection_name: str) -> Optional[dict]:
        """
        Retrieve a specific chunk by its ``chunk_id`` field from Qdrant.

        Args:
            chunk_id:        The UUID stored in the ``chunk_id`` payload field.
            collection_name: Collection to search.

        Returns:
            A chunk dict, or None if not found.
        """
        try:
            from qdrant_client.http import models as qmodels

            scroll_result = self.vs.client.scroll(
                collection_name=collection_name,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="chunk_id",
                            match=qmodels.MatchValue(value=chunk_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            if scroll_result[0]:
                return self._point_to_chunk(scroll_result[0][0], collection_name)
            logger.warning(f"[Tools] get_chunk: chunk_id={chunk_id} not found in {collection_name}")
            return None

        except Exception as exc:
            logger.error(f"[Tools] get_chunk error: {exc}")
            return None

    # ── Tool 4: summarize_table ───────────────────────────────────────────────

    def summarize_table(self, chunk: dict, question: str) -> str:
        """
        Parse a markdown table embedded in ``chunk['page_content']`` and
        answer *question* using the LLM, restricted to that table only.

        The system stores table content as plain-text markdown inside
        ``page_content`` (``has_table=True`` chunks).  There is no separate
        ``table_json`` field — this tool extracts lines containing ``|``.

        Args:
            chunk:    A chunk dict (must have ``has_table=True`` for best results).
            question: The user's question about the table.

        Returns:
            LLM answer string, or the raw table text as fallback.
        """
        try:
            text = chunk.get("page_content") or chunk.get("content") or ""

            table_lines = [line for line in text.splitlines() if "|" in line]
            if not table_lines:
                logger.info("[Tools] summarize_table: no table lines found, returning raw text")
                return text

            table_text = "\n".join(table_lines)
            logger.info(
                f"[Tools] summarize_table: {len(table_lines)} table lines, "
                f"question='{question[:60]}'"
            )

            prompt = (
                "You are a data extraction assistant. "
                "Answer the question using ONLY the table below.\n\n"
                "Rules:\n"
                "  1. Use only information present in the table.\n"
                "  2. If the answer is not in the table, say so.\n"
                "  3. Be concise and direct.\n"
                "  4. Preserve numbers, names, and dates exactly as they appear.\n\n"
                f"Table:\n{table_text}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

            msg = self.llm.invoke([HumanMessage(content=prompt)])
            result = (msg.content or "").strip()
            return result if result else table_text

        except Exception as exc:
            logger.error(f"[Tools] summarize_table error: {exc}")
            return chunk.get("page_content") or ""

    # ── Tool 5: confidence_score ─────────────────────────────────────────────

    def confidence_score(self, query: str, retrieved_chunks: List[dict]) -> float:
        """
        Compute a [0, 1] confidence score indicating how well the retrieved
        chunks likely answer the query.

        Components (weighted sum):
          - ``rrf_max``     (weight 0.50): best RRF score, soft-capped at 0.05
          - ``overlap``     (weight 0.30): fraction of query tokens in the top chunk
          - ``rrf_avg``     (weight 0.20): average RRF score, soft-capped at 0.02

        Args:
            query:            The original user query.
            retrieved_chunks: List of chunk dicts (after reranking).

        Returns:
            Float in [0.0, 1.0].
        """
        if not retrieved_chunks:
            return 0.0

        scores = [c.get("score") or 0 for c in retrieved_chunks]
        max_s = max(scores)
        avg_s = sum(scores) / len(scores)

        query_tokens = set(re.sub(r"\s+", " ", query.lower()).split())
        top_text = (
            retrieved_chunks[0].get("page_content")
            or retrieved_chunks[0].get("content")
            or ""
        ).lower()
        top_tokens = set(re.sub(r"\s+", " ", top_text).split())

        overlap = len(query_tokens & top_tokens) / (len(query_tokens) + 1e-9) if query_tokens else 0.0
        overlap = min(overlap, 1.0)

        # Soft-normalise RRF scores: >0.05 → 1.0, 0 → 0.0
        rrf_max_norm = min(max_s / 0.05, 1.0)
        rrf_avg_norm = min(avg_s / 0.02, 1.0)

        confidence = (
            0.50 * rrf_max_norm
            + 0.30 * overlap
            + 0.20 * rrf_avg_norm
        )
        confidence = round(min(max(confidence, 0.0), 1.0), 4)

        logger.info(
            f"[Tools] confidence_score: max_rrf={max_s:.5f} avg_rrf={avg_s:.5f} "
            f"overlap={overlap:.2f} → confidence={confidence:.4f}"
        )
        return confidence

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _search_all(
        self, query: str, top_k: int, query_language: Optional[str], source_filter: Optional[str] = None
    ) -> List[dict]:
        """Search all non-system collections and merge results by score."""
        try:
            skip = {"inactive", "history", "history_inactive"}
            collections_resp = self.vs.client.get_collections()
            all_chunks: List[dict] = []

            for col in collections_resp.collections:
                if col.name in skip:
                    continue
                try:
                    points = self.vs.query_collection_hybrid(
                        collection_name=col.name,
                        query=query,
                        k=top_k,
                        query_language=query_language,
                        source_filter=source_filter,
                    )
                    all_chunks.extend(
                        self._point_to_chunk(p, col.name) for p in points if p
                    )
                except Exception as exc:
                    logger.warning(f"[Tools] _search_all: collection '{col.name}' failed: {exc}")

            all_chunks.sort(key=lambda c: c.get("score") or 0, reverse=True)
            logger.info(f"[Tools] _search_all: {len(all_chunks)} total hits across all collections")
            return all_chunks[:top_k]

        except Exception as exc:
            logger.error(f"[Tools] _search_all error: {exc}")
            return []

    @staticmethod
    def _point_to_chunk(point, collection_name: str) -> dict:
        """
        Convert a raw Qdrant hit (ScoredPoint or dict) to a standardized chunk
        dict using the exact payload field names defined in the Qdrant schema.
        """
        payload = getattr(point, "payload", None)
        if payload is None and isinstance(point, dict):
            payload = point.get("payload", {})
        payload = payload or {}

        score = getattr(point, "score", None)
        if score is None and isinstance(point, dict):
            score = point.get("score", 0.0)
        score = float(score or 0.0)

        return {
            # ── identity ──────────────────────────────────────────────────────
            "chunk_id": payload.get("chunk_id") or str(getattr(point, "id", "")),
            "source": str(payload.get("source", "")),       # document UUID

            # ── text content ─────────────────────────────────────────────────
            # page_content: full markitdown text (may have "# Header\nbody")
            # content:      clean body only
            "page_content": payload.get("page_content") or payload.get("content", ""),
            "content": payload.get("content") or payload.get("page_content", ""),
            "html": payload.get("html", ""),

            # ── metadata ─────────────────────────────────────────────────────
            "file_name": payload.get("file_name", "Document"),
            "page_number": int(payload.get("page_number") or 1),
            "header": payload.get("header", ""),
            "header_level": int(payload.get("header_level") or 1),
            "language": payload.get("language", ""),
            "has_table": bool(payload.get("has_table", False)),
            "chunk_index": int(payload.get("chunk_index") or 0),
            "is_active": bool(payload.get("is_active", True)),
            "extraction_method": payload.get("extraction_method", ""),

            # ── retrieval score ──────────────────────────────────────────────
            "score": score,
            "collection": collection_name,
        }

    @staticmethod
    def _lang_from_filters(filters: Dict) -> Optional[str]:
        """Extract the primary (non-multi) language from a filters dict."""
        lang_list = filters.get("language") if filters else None
        if not lang_list:
            return None
        non_multi = [l for l in lang_list if l != "multi"]
        return non_multi[0] if non_multi else None
