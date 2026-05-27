"""Configuration for the RAG system."""

from __future__ import annotations

import os
from pathlib import Path


def env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).lower() in {"1", "true", "yes", "on"}


def env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    return int(value) if value not in (None, "") else default


def env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    return float(value) if value not in (None, "") else default


PROJECT_ROOT = Path(__file__).resolve().parents[4]


SUPPORTED_EXTENSIONS = {
    "text": [".txt", ".md"],
    "word": [".docx", ".doc"],
    "spreadsheet": [".xlsx", ".xls"],
    "presentation": [".pptx", ".ppt"],
    "csv": [".csv"],
    "tsv": [".tsv"],
}


# Path to the local bge-reranker-v2-m3 model (cross-encoder).
RERANKER_MODEL_PATH = os.environ.get(
    "RERANKER_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "bge-reranker-v2-m3"),
)


# Master switch - set to False to fall back to the classic single-pass retrieval.
AGENTIC_RAG_ENABLED = env_bool("AGENTIC_RAG_ENABLED", True)


# Retrieval passes.
TOP_K_BROAD = env_int("AGENTIC_TOP_K_BROAD", 40)
TOP_K_SUBQUERY = env_int("AGENTIC_TOP_K_SUBQUERY", 20)
TOP_K_RERANK = env_int("AGENTIC_TOP_K_RERANK", 15)
RETRIEVAL_K = env_int("AGENTIC_RETRIEVAL_K", 20)


# Confidence gating.
CONFIDENCE_THRESHOLD_KM = env_float("AGENTIC_CONFIDENCE_THRESHOLD_KM", 0.57)
CONFIDENCE_THRESHOLD_EN = env_float("AGENTIC_CONFIDENCE_THRESHOLD_EN", 0.57)
MIN_KEYWORD_OVERLAP = env_float("AGENTIC_MIN_KEYWORD_OVERLAP", 0.05)


# Planner.
MAX_SUBQUERIES = env_int("AGENTIC_MAX_SUBQUERIES", 3)
MAX_RETRIES = env_int("AGENTIC_MAX_RETRIES", 2)


# Self-verification.
VERIFICATION_ENABLED = env_bool("AGENTIC_VERIFICATION_ENABLED", True)


# Query path controls.
QUERY_VALIDATION_ENABLED = env_bool("AGENTIC_QUERY_VALIDATION_ENABLED", True)
# Keep follow-up questions enabled by default for drop-in folder replacement.
FOLLOWUP_SUGGESTIONS_ENABLED = env_bool("AGENTIC_FOLLOWUPS_ENABLED", True)
FOLLOWUP_SUGGESTIONS_COUNT = max(0, env_int("AGENTIC_FOLLOWUPS_COUNT", 3))


# Language support.
SUPPORTED_LANGUAGES = ["km", "en", "multi"]


# Query queue.
QUEUE_MAX_WORKERS = env_int("AGENTIC_QUEUE_MAX_WORKERS", 4)
QUEUE_MAX_QUEUE_SIZE = env_int("AGENTIC_QUEUE_MAX_QUEUE_SIZE", 100)
QUEUE_TIMEOUT_SECONDS = env_int("AGENTIC_QUEUE_TIMEOUT_SECONDS", 5)
QUEUE_DEFAULT_PRIORITY = env_int("AGENTIC_QUEUE_DEFAULT_PRIORITY", 5)


# Model Configuration
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemma4:e2b-mlx")

# Prompts

PLANNER_PROMPT = """You are an expert retrieval planner for an enterprise RAG system.
Your goal is to analyze the user's question and create a step-by-step retrieval plan.

Input:
- User Question: {question}
- Current Date: {current_date}

Analysis Required:
1. Identify the intent: "lookup", "comparison", "summarization", "policy_extraction", "multi_hop".
2. Identify key entities and terms.
3. Determine if the question requires multiple steps or sub-queries.
4. Determine necessary filters (e.g., specific document types, dates).

Output JSON format:
{{
    "intent": "string",
    "reasoning": "string",
    "sub_queries": ["string", "string"],  // Max 3 sub-queries. If simple lookup, just 1 (the original).
    "filters": {{
        "file_name": ["string"], // optional
        "source": ["string"]     // optional
    }}
}}
"""

CONTENT_SYNTHESIS_PROMPT = """You are a helpful assistant.
Your goal is to extract the answer from the following context based on the user question.
If the answer is not found in the context, use the context to answer as best as you can.
DO NOT say "not found" if the context itself contains the information requested (e.g. a list, table, or content).
You MUST answer in {language}.
- If {language} is "Khmer", the ENTIRE response MUST be in Khmer script.

Context:
{context}

Question:
{question}

Answer:"""

SYNTHESIS_PROMPT = """You are an expert enterprise assistant.

INSTRUCTIONS:
1. Answer the user's question based ONLY on the provided context chunks below.
2. If the answer is not found in the context, say "Information not found in indexed documents".
3. Do NOT use external knowledge.
4. LANGUAGE REQUIREMENT: You MUST answer in {language}.
   - If {language} is "Khmer", the ENTIRE response MUST be in Khmer script.
   - Do NOT use English unless the specific technical term is only known in English.
   - If the context is in English but {language} is Khmer, TRANSLATE the answer to Khmer.
5. Do NOT repeat the question. Start directly with the answer.

CONTEXT:
{context}

QUESTION:
{question}
"""

VERIFICATION_PROMPT = """Verify if the answer is supported by the context and in the correct language ({language}).

Question: {question}
Context: {context}
Answer: {answer}

Output JSON:
{{"valid": boolean, "reason": "string"}}
"""
