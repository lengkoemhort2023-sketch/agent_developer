"""
Multi-step Agentic Retriever.

Execution order
───────────────
Pass A  Broad hybrid search
        Query: original question
        top_k: TOP_K_BROAD (default 40)
        Purpose: cast a wide semantic net

Pass B  Sub-query expansion  (skipped when plan.sub_queries is empty)
        Query: each planner sub-query independently
        top_k: TOP_K_SUBQUERY (default 20) per sub-query
        Results merged and de-duplicated with Pass A results

Pass C  Rerank + prune
        All unique chunks from A+B are re-scored with
        ``RAGTools.rerank()`` (RRF-norm + keyword overlap)
        Only the top TOP_K_RERANK (default 15) are returned

The output is a list of chunk dicts ready for the confidence gate and
downstream answer synthesis.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .planner import QueryPlan
from .tools import RAGTools

logger = logging.getLogger(__name__)


class AgenticRetriever:
    """
    Executes the three-pass multi-step retrieval strategy.

    This class is stateless between calls — every call to :meth:`retrieve`
    is independent.
    """

    def __init__(self, tools: RAGTools) -> None:
        self.tools = tools

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        question: str,
        plan: QueryPlan,
        collection_name: Optional[str],
        doc_id: Optional[str] = None,
    ) -> List[dict]:
        """
        Execute the three-pass retrieval and return reranked chunks.

        Args:
            question:        Original user query.
            plan:            :class:`QueryPlan` produced by the planner.
            collection_name: Qdrant collection to search (None = all collections).
            doc_id:          When set, restricts retrieval to a single document
                             (used when the user is querying a specific file).

        Returns:
            Up to ``TOP_K_RERANK`` chunk dicts sorted by ``_rerank_score`` desc.
        """
        from .config import TOP_K_BROAD, TOP_K_SUBQUERY, TOP_K_RERANK

        # Merge doc_id restriction into filters so it is pushed into Qdrant queries
        # (Pass A and Pass B) rather than applied as a lossy Python post-filter.
        filters = dict(plan.filters)
        if doc_id:
            filters["source"] = str(doc_id)
            logger.info(f"[Retriever] doc_id mention active – restricting search to source={doc_id}")

        # ── Pass A: broad search ──────────────────────────────────────────────
        logger.info(f"[Retriever] Pass A  top_k={TOP_K_BROAD}")
        pass_a = self.tools.hybrid_search(
            query=question,
            top_k=TOP_K_BROAD,
            collection_name=collection_name,
            filters=filters,
        )
        logger.info(f"[Retriever] Pass A  → {len(pass_a)} chunks")

        # ── Pass B: sub-query expansion ───────────────────────────────────────
        pass_b: List[dict] = []
        if plan.sub_queries:
            logger.info(
                f"[Retriever] Pass B  {len(plan.sub_queries)} sub-queries "
                f"top_k={TOP_K_SUBQUERY} each"
            )
            for sq in plan.sub_queries:
                sq_chunks = self.tools.hybrid_search(
                    query=sq,
                    top_k=TOP_K_SUBQUERY,
                    collection_name=collection_name,
                    filters=filters,
                )
                logger.info(
                    f"[Retriever] Pass B  sub-query='{sq[:60]}' → {len(sq_chunks)} chunks"
                )
                pass_b.extend(sq_chunks)

        # ── Merge + deduplicate ───────────────────────────────────────────────
        all_chunks = self._merge_dedup(pass_a, pass_b)
        logger.info(f"[Retriever] After merge+dedup: {len(all_chunks)} unique chunks")

        if not all_chunks:
            return []

        # ── Pass C: rerank + prune ────────────────────────────────────────────
        logger.info(f"[Retriever] Pass C  rerank → keep top {TOP_K_RERANK}")
        reranked = self.tools.rerank(question, all_chunks)
        top_chunks = reranked[:TOP_K_RERANK]

        if top_chunks:
            logger.info(
                f"[Retriever] Final: {len(top_chunks)} chunks  "
                f"top_rerank={top_chunks[0].get('_rerank_score', 0):.4f}  "
                f"top_rrf={top_chunks[0].get('score', 0):.5f}"
            )
        else:
            logger.warning("[Retriever] Final: 0 chunks after reranking")

        return top_chunks

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _merge_dedup(*chunk_lists: List[dict]) -> List[dict]:
        """
        Merge any number of chunk lists into one, keeping each chunk_id only
        once (the occurrence with the highest RRF score is retained).

        Falls back to using the first 50 characters of ``page_content`` as a
        de-duplication key when ``chunk_id`` is absent.
        """
        seen: Dict[str, dict] = {}

        for chunk_list in chunk_lists:
            for chunk in chunk_list:
                key = (
                    chunk.get("chunk_id")
                    or (chunk.get("page_content") or "")[:50]
                    or id(chunk)
                )
                existing = seen.get(key)  # type: ignore[arg-type]
                if existing is None or (chunk.get("score") or 0) > (existing.get("score") or 0):
                    seen[key] = chunk  # type: ignore[index]

        return list(seen.values())
