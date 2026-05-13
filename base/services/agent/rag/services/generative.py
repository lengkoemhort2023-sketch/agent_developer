from .vector_store import VectorStoreService
from langchain_ollama import ChatOllama
from .document_loader import DocumentLoaderService
from django.conf import settings
from sentence_transformers import util
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document as LangChainDocument
from langchain_qdrant import QdrantVectorStore
import os
import uuid
import time
import numpy as np
import logging
import json
from typing import List, Dict, Any, Optional
import re
import psutil
import markdown as py_markdown


def cleanup_bold_colon(text: str) -> str:
    """
    Clean up markdown bold colon patterns that appear in table cells.
    Examples:
      '**text**:**' -> 'text:'
      '**text** :**' -> 'text:'
      '**:**' -> ':'
      '** :**' -> ':'
      '**?**' -> '?' (Khmer colon)
      '**?**:' -> '?' (Khmer colon with regular colon)
    """
    if not text:
        return text
    # Replace **:** or ** :** with just :
    text = re.sub(r'\*\*\s*:\*\*', ':', text)
    text = re.sub(r'\*\*\s*:\s*\*\*', ':', text)
    # Replace **text**:** or **text** :** with text:
    text = re.sub(r'\*\*([^*]+)\*\*\s*:\*\*', r'\1:', text)
    text = re.sub(r'\*\*([^*]+)\*\*\s*:\s*\*\*', r'\1:', text)
    # Normalize malformed "quad-bold" wrappers: ****text**** -> **text**
    text = re.sub(r'\*{4}([^*]+)\*{4}', r'**\1**', text)

    # Replace bold Khmer punctuation only (":"/"៖"), without touching normal bold text.
    text = re.sub(r'\*\*\s*[:៖]\s*\*\*', '៖', text)
    text = re.sub(r'\*\*\s*[:៖]\s*\*\*\s*:', '៖', text)
    return text
import gc

qdrant_host = settings.QDRANT_HOST
qdrant_port = settings.QDRANT_PORT
from .token_logger import TokenLogger

# Add logging configuration
logger = logging.getLogger(__name__)

LEGACY_DISPLAY_THRESHOLD_KM = 0.57
LEGACY_DISPLAY_THRESHOLD_EN = 0.57


def strip_mention_prefix(text: str) -> str:
    """Remove any leading frontend document mention prefixes before retrieval."""
    if not text:
        return text

    cleaned = text.strip()
    while True:
        updated = re.sub(r'^\s*\(@[^)]*\)\s*\|\s*', '', cleaned).strip()
        if updated == cleaned:
            return cleaned
        cleaned = updated


def normalize_user_query(text: str) -> str:
    """Normalize user query text (mentions, hidden chars, whitespace)."""
    if not text:
        return ""

    cleaned = strip_mention_prefix(text)
    # Remove zero-width/control formatting chars that often appear in copied headers.
    cleaned = re.sub(r'[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]', '', cleaned)
    # Normalize tabs/newlines/non-breaking spaces into regular spaces.
    cleaned = cleaned.replace('\u00A0', ' ').replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')
    # Normalize token separators that often appear in pasted identifiers.
    cleaned = cleaned.replace('_', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

# Production optimization: Limit numpy threads to prevent memory bloat
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

class MemoryManager:
    """Monitor and manage system memory for production"""
    # Lower thresholds for production - be more proactive
    MEMORY_THRESHOLD = 50  # percentage - start cleanup at 50%
    CRITICAL_THRESHOLD = 75  # percentage - critical level at 75%
    
    # Singleton instance to track cleanup state
    _instance = None
    _last_cleanup_time = 0
    _cleanup_cooldown = 10  # seconds between cleanups

    @staticmethod
    def check_memory():
        """Check current memory usage"""
        ram = psutil.virtual_memory()
        return ram.percent

    @staticmethod
    def check_gpu_memory():
        """Check GPU memory usage if available"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                gpu_memory_cached = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
                return gpu_memory, gpu_memory_cached
            return 0, 0
        except Exception:
            return 0, 0

    @staticmethod
    def force_cleanup():
        """Force comprehensive garbage collection and memory cleanup"""
        try:
            # Force Python garbage collection
            gc.collect()
            
            # Clear Python object cache
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()
            
            # Clear module caches
            import importlib
            importlib.invalidate_caches()
            
            # Clear string interning cache
            import sys
            if hasattr(sys, 'intern'):
                sys.intern('')
            
            return True
        except Exception as e:
            logging.warning(f"Memory cleanup warning: {e}")
            return False

    @staticmethod
    def force_gpu_cleanup():
        """Force GPU memory cleanup"""
        try:
            import torch
            if torch.cuda.is_available():
                # Empty CUDA cache
                torch.cuda.empty_cache()
                
                # Clear CUDA memory reserved
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
                
                # Clear any cached models
                torch.cuda.synchronize()
                
                return True
        except Exception as e:
            logging.warning(f"GPU cleanup warning: {e}")
            return False

    @staticmethod
    def should_cleanup(force: bool = False):
        """Check if memory cleanup is needed with cooldown protection"""
        import time
        
        # Allow forced cleanup regardless of cooldown
        if force:
            return MemoryManager.check_memory() > MemoryManager.MEMORY_THRESHOLD
        
        # Check cooldown
        time_since_last = time.time() - MemoryManager._last_cleanup_time
        if time_since_last < MemoryManager._cleanup_cooldown:
            return False
        
        return MemoryManager.check_memory() > MemoryManager.MEMORY_THRESHOLD

    @staticmethod
    def should_critical_cleanup():
        """Check if critical memory cleanup is needed"""
        return MemoryManager.check_memory() > MemoryManager.CRITICAL_THRESHOLD
    
    @staticmethod
    def record_cleanup():
        """Record that cleanup was performed"""
        import time
        MemoryManager._last_cleanup_time = time.time()

    @staticmethod
    def get_memory_status():
        """Get comprehensive memory status"""
        ram = psutil.virtual_memory()
        gpu_used, gpu_cached = MemoryManager.check_gpu_memory()
        
        return {
            'ram_percent': ram.percent,
            'ram_used_gb': ram.used / (1024 ** 3),
            'ram_total_gb': ram.total / (1024 ** 3),
            'gpu_memory_used_gb': gpu_used,
            'gpu_memory_cached_gb': gpu_cached,
            'needs_cleanup': MemoryManager.should_cleanup(),
            'needs_critical_cleanup': MemoryManager.should_critical_cleanup()
        }
    
    @staticmethod
    def smart_cleanup():
        """
        Smart cleanup based on memory level.
        Call this before memory-intensive operations, not after.
        Returns True if cleanup was performed.
        """
        current_mem = MemoryManager.check_memory()
        
        # Always clean if critical
        if current_mem > MemoryManager.CRITICAL_THRESHOLD:
            MemoryManager.force_cleanup()
            MemoryManager.force_gpu_cleanup()
            MemoryManager.record_cleanup()
            return True
        
        # Clean if above threshold and cooldown has passed
        if MemoryManager.should_cleanup(force=False):
            MemoryManager.force_cleanup()
            # Only clean GPU if using CUDA
            gpu_used, _ = MemoryManager.check_gpu_memory()
            if gpu_used > 1:  # Only clean if using >1GB GPU memory
                MemoryManager.force_gpu_cleanup()
            MemoryManager.record_cleanup()
            return True
        
        return False

class ResponseGenerationService:
    """ RAG Service for question answering using retrieved chunks"""

    _DEFAULT_LLM_CONTENT_PROMPT_TEMPLATE = (
        "You are an expert Q&A assistant. "
        "Your task is to answer the user's question based *strictly* on the provided context.\n\n"
        "**Instructions:**\n"
        "1.  **Locate the relevant section**: Scan the context and identify ONLY the part "
        "that directly answers the question. The context may contain many sections — "
        "do NOT return everything, only the matching section.\n"
        "2.  **Return that section verbatim**: Once you have identified the relevant "
        "section, copy it EXACTLY as it appears — including all list items, table rows, "
        "and formatting. Do NOT summarise, shorten, paraphrase, or reformat it.\n"
        "3.  **Use ONLY the provided context**: If the answer is not in the context, "
        "respond with \"I am sorry, but the answer to your question is not available "
        "in the provided context.\"\n"
        "4.  **Do not use prior knowledge**: Your response must be grounded entirely in "
        "the text provided.\n"
        "5.  **No padding**: Do not add introductions, conclusions, or any text that is "
        "not part of the extracted answer itself.\n\n"
        "**Context:**\n{body}\n\n"
        "**User Question:**\n{question}\n\n"
        "**Answer:**"
    )

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.llm_content_prompt_template = self._DEFAULT_LLM_CONTENT_PROMPT_TEMPLATE
            try:
                self.vector_service = VectorStoreService()
            except Exception as e:
                logger.error(f"Error initializing VectorStoreService: {e}")
                self.vector_service = None
            self.document_loader = DocumentLoaderService()
            self.token_logger = TokenLogger("logdata/token_usage.csv")


            # Add logging configuration
            logging.basicConfig(level=logging.INFO)

            # Database customized
            self.client = QdrantClient(host=qdrant_host, port=int(qdrant_port))

            self.history_collection_name = "history"
            self.inactive_history_collection_name = "history_inactive"

            # Create history collection if it doesn't exist
            try:
                self.client.get_collection(self.history_collection_name)
            except Exception:
                # Collection doesn't exist, create it
                try:
                    if self.vector_service and getattr(self.vector_service, 'embedding', None):
                        test_embedding = self.vector_service.embedding.embed_query("test")
                        vector_size = len(test_embedding)
                        self.client.create_collection(
                            collection_name=self.history_collection_name,
                            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                        )
                    else:
                        logger.warning("Embedding not available - skipping history collection creation for now.")
                except Exception as e:
                    logger.warning(f"Could not create history collection now: {e}. Will attempt later when embeddings are available.")

            # Create inactive history collection if it doesn't exist
            try:
                self.client.get_collection(self.inactive_history_collection_name)
            except Exception:
                # Collection doesn't exist, create it
                try:
                    if self.vector_service and getattr(self.vector_service, 'embedding', None):
                        test_embedding = self.vector_service.embedding.embed_query("test")
                        vector_size = len(test_embedding)
                        self.client.create_collection(
                            collection_name=self.inactive_history_collection_name,
                            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                        )
                    else:
                        logger.warning("Embedding not available - skipping inactive history collection creation for now.")
                except Exception as e:
                    logger.warning(f"Could not create inactive history collection now: {e}. Will attempt later when embeddings are available.")

            # Note: For history, we'll use direct Qdrant operations

            llm_num_ctx = max(1024, int(getattr(settings, "OLLAMA_NUM_CTX", 16384) or 16384))
            llm_num_predict = max(
                128,
                int(getattr(settings, "OLLAMA_NUM_PREDICT", 1024) or 1024),
            )
            llm_temperature = float(getattr(settings, "OLLAMA_TEMPERATURE", 0.1) or 0.1)
            self.llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=llm_temperature,
                num_ctx=llm_num_ctx,
                num_predict=llm_num_predict,
            )
            logger.info(
                f"Ollama LLM initialized successfully with model: {settings.OLLAMA_MODEL}, "
                f"num_ctx={llm_num_ctx}, num_predict={llm_num_predict}, temperature={llm_temperature}"
            )

            # Initialize AgenticRAG (replaces single-pass get_related_docs)
            self.agentic_rag = None
            try:
                from ..config import AGENTIC_RAG_ENABLED
                if AGENTIC_RAG_ENABLED and self.vector_service:
                    from ..agent import AgenticRAG
                    self.agentic_rag = AgenticRAG(self.vector_service, self.llm)
                    logger.info("[AgenticRAG] Ready")
                else:
                    logger.info("[AgenticRAG] Disabled - using classic get_related_docs()")
            except Exception as _rag_init_err:
                logger.error(f"[AgenticRAG] Init failed, falling back to classic retrieval: {_rag_init_err}")
                self.agentic_rag = None


    def _get_conversation_context(self, session_id: str, max_history: int = 3) -> str:
        """
        Get recent conversation history for context.
        Returns formatted string of recent Q&A pairs.
        """
        try:
            if not session_id:
                return ""
            
            from chat.models import ChatMessage, ChatSession
            
            # Get session
            session = ChatSession.objects.filter(id=session_id).first()
            if not session:
                return ""
            
            # Get last N messages (most recent first)
            recent_messages = ChatMessage.objects.filter(
                session=session
            ).order_by('-created_at')[:max_history]
            
            if not recent_messages:
                return ""
            
            # Format as conversation history (reverse to chronological order)
            history_parts = []
            for msg in reversed(recent_messages):
                history_parts.append(f"User: {msg.question}")
                # Truncate long answers
                answer_preview = msg.answer[:200] if len(msg.answer) > 200 else msg.answer
                history_parts.append(f"Assistant: {answer_preview}")
            
            context = "\n".join(history_parts)
            logger.info(f"Retrieved {len(recent_messages)} messages for conversation context")
            return context
            
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return ""

    @staticmethod
    def _contains_thai_characters(text: str) -> bool:
        return bool(text and re.search(r"[\u0E00-\u0E7F]", text))

    def _sanitize_followup_questions(self, suggestions: list, count: int) -> list:
        cleaned = []
        blocked = 0

        for suggestion in suggestions or []:
            question_text = str(suggestion).strip().strip("\"'")
            if not question_text:
                continue
            if self._contains_thai_characters(question_text):
                blocked += 1
                continue
            if len(question_text) <= 3:
                continue
            cleaned.append(question_text)
            if len(cleaned) >= count:
                break

        if blocked:
            logger.info(f"Blocked {blocked} Thai follow-up suggestion(s)")

        return cleaned

    def _generate_followup_questions(self, question: str, top_answers: list, count: int, language: str) -> list:
        """
        Generate follow-up questions using LLM from chunk header + content (up to 200 words).
        Each candidate is formatted as:  [Header]\n<first 200 words of content>
        The LLM uses this material to generate relevant follow-up questions in the
        same language as the original query.
        """
        try:
            if not question or count <= 0:
                return []

            if self._contains_thai_characters(question):
                logger.info("Skipping follow-up generation because the user question contains Thai characters")
                return []

            # Detect language directly from the question for accuracy.
            # 'multi' means mixed/Khmer-dominant treat it as Khmer (same as outer scope).
            detected = self.document_loader.detect_languages(question)
            km = detected in ('km', 'multi')
            lang_name = "Khmer" if km else "English"

            # -- Build context blocks: header + 50-word snippet ---------------
            # Using 50 words instead of 200 to avoid exhausting num_predict tokens
            # (the model has OLLAMA_NUM_PREDICT=1024 and needs room for output)
            context_blocks = []
            for ans in (top_answers or [])[:max(count + 2, 6)]:
                if not isinstance(ans, dict):
                    continue

                header = (ans.get('header') or '').strip()
                text = (ans.get('text') or '').strip()

                words = text.split()
                snippet = ' '.join(words[:50])

                if header or snippet:
                    block = f"[{header}]\n{snippet}" if header else snippet
                    context_blocks.append(block)

            if not context_blocks:
                logger.warning("No context blocks for follow-up generation")
                return []

            context_text = "\n\n---\n\n".join(context_blocks)

            def _llm_call(prompt_lang: str, attempt_label: str) -> str:
                """Make an LLM call and return cleaned response text."""
                prompt = (
                    f"You are a helpful Q&A assistant. Based on the document sections below, "
                    f"generate exactly {count} short follow-up questions in {prompt_lang} "
                    f"that a user might want to ask next after asking:\n"
                    f"\"{question}\"\n\n"
                    f"Document sections:\n{context_text}\n\n"
                    f"Rules:\n"
                    f"- Every question MUST be written in {prompt_lang} only.\n"
                    f"- Thai language is strictly forbidden in the output.\n"
                    f"- Do NOT use any Thai characters.\n"
                    f"- Each question must be under 15 words.\n"
                    f"- Questions must be directly based on the sections above.\n"
                    f"- Output ONLY a JSON array of strings: [\"Q1\", \"Q2\", ...].\n"
                    f"- No numbering, no explanation, no extra text.\n\n"
                    f"If {prompt_lang} is Thai, automatically use English instead.\n\n"
                    f"JSON array:"
                )
                logger.info(
                    f"Generating {count} follow-up questions via LLM "
                    f"(lang={prompt_lang}, attempt={attempt_label})"
                )
                resp = self.llm.invoke([HumanMessage(content=prompt)])
                raw = (resp.content or "").strip()
                # Strip <think>...</think> blocks
                raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE).strip()
                # Strip markdown fences
                raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
                raw = re.sub(r'\s*```$', '', raw).strip()
                logger.debug(
                    f"Follow-up LLM raw output (attempt={attempt_label}): "
                    f"{'<EMPTY>' if not raw else raw[:300]}"
                )
                return raw

            # Attempt 1: original detected language
            resp_text = _llm_call(lang_name, "1-detected-lang")

            # Attempt 2: if empty and language was Khmer, retry with English
            if not resp_text and km:
                logger.info("LLM returned empty for Khmer; retrying with English")
                resp_text = _llm_call("English", "2-english-fallback")

            # JSON array parse find the LAST valid JSON array in the response
            # (using greedy search so we skip any partial arrays in preamble text)
            try:
                # Find all [...] candidates (non-greedy, then validate each)
                candidates = list(re.finditer(r'\[.*?\]', resp_text, re.DOTALL))
                # Try from last to first; the LLM output array is usually at the end
                for m in reversed(candidates):
                    try:
                        parsed = json.loads(m.group())
                        if isinstance(parsed, list) and parsed:
                            result = self._sanitize_followup_questions(parsed, count)
                            if result:
                                logger.info(f"Follow-up questions generated: {result}")
                                return result[:count]
                    except Exception:
                        continue
            except Exception:
                pass

            # Line-by-line fallback: skip lines that look like think-block content
            cleaned = []
            for line in resp_text.splitlines():
                line = line.strip().strip("[]\"',")
                line = re.sub(r'^[\-\d\.\)\s]+', '', line).strip()
                # Skip very short lines or lines that look like reasoning prose
                if line and len(line) > 10:
                    if not line.endswith('?'):
                        line += '?'
                    cleaned.append(line)
                    if len(cleaned) >= count:
                        break

            cleaned = self._sanitize_followup_questions(cleaned, count)
            if cleaned:
                return cleaned[:count]

            # Last resort: try splitting by common delimiters in case LLM output is dense
            if not cleaned:
                parts = re.split(r'[?\n]+', resp_text)
                for part in parts:
                    part = part.strip().strip("[]\"',")
                    part = re.sub(r'^[\-\d\.\)\s]+', '', part).strip()
                    if part and len(part) > 10:
                        if not part.endswith('?'):
                            part += '?'
                        cleaned.append(part)
                        if len(cleaned) >= count:
                            break
                cleaned = self._sanitize_followup_questions(cleaned, count)
                if cleaned:
                    return cleaned[:count]

            # Ultra-last-resort: split by comma (LLM sometimes outputs comma-separated values
            # instead of proper JSON or line-by-line)
            if not cleaned:
                parts = resp_text.split(",")
                for part in parts:
                    part = part.strip().strip("[]\"'`\n\r")
                    part = re.sub(r'^[\-\d\.\)\s]+', '', part).strip()
                    if part and len(part) > 10:
                        if not part.endswith('?'):
                            part += '?'
                        cleaned.append(part)
                        if len(cleaned) >= count:
                            break
                cleaned = self._sanitize_followup_questions(cleaned, count)
                if cleaned:
                    return cleaned[:count]

            logger.warning("LLM follow-up parse failed returning empty suggestions")

            # Final fallback: generate suggestions from top_answers headers
            # This works even when the LLM fails (returns empty/unparseable)
            fallback_headers = []
            for ans in (top_answers or [])[:count + 1]:
                if not isinstance(ans, dict):
                    continue
                header = (ans.get('header') or '').strip()
                text = (ans.get('text') or '').strip()
                candidate = header or text[:80]
                if candidate and len(candidate) > 10:
                    fallback_headers.append(candidate)
                if len(fallback_headers) >= count:
                    break

            if fallback_headers:
                logger.info(
                    f"Using header-based fallback suggestions ({len(fallback_headers)} items): "
                    f"{fallback_headers}"
                )
                return fallback_headers[:count]

            return []

        except Exception as exc:
            logger.warning(f"_generate_followup_questions failed: {exc}")
            return []

    def verify_query(self, user_input: str) -> tuple:
        """
        Verifies if the user query is a correct/meaningful query before Qdrant search.
        Returns: (is_valid: bool, reason: str)
        Uses LLM to evaluate query quality for both Khmer and English.
        """
        try:
            detected_lang = self.document_loader.detect_languages(user_input)
            
            # Accept likely section-header queries (short/title-like) to avoid
            # false rejection when users paste document headers as queries.
            normalized = re.sub(r'^\s*\([^)]*\)\s*\|\s*', '', (user_input or '').strip())
            normalized = re.sub(r'\s+', ' ', normalized).strip(" -:|")
            words = [w for w in re.split(r'\s+', normalized) if w]
            if normalized and len(words) <= 7:
                alpha_words = [w for w in words if re.search(r'[A-Za-z]', w)]
                title_like_words = [w for w in alpha_words if w[:1].isupper()]
                # Example: "Physical Security and Environmental Controls"
                if alpha_words and len(title_like_words) >= max(2, len(alpha_words) - 1):
                    logger.info(
                        f"Query verification ({detected_lang}): '{user_input}' -> True "
                        f"(header-like short query, auto-accepted)"
                    )
                    return (True, "Header-like short query, auto-accepted")

            verification_prompt = f"""Evaluate whether the user query is related to ANY of the following departments:

Core Banking System, Contact Center, Credit Business, Executive, Finance, Human Resources, Internal Audit, IT Infrastructure & Operation, Marketing and Communication, Product Development, Admin and Procurement, Risk Management, Research, Training and Development, Treasury, IT Security, Bancassurance, Management Information System, Legal and Compliance, Deposit and Service, Business, IT Project Management, Software Research and Development, Credit Underwriting, Operations, Digital Banking & Card Payment, Business Development, Supply Chain Financing Business, Credit Control, Research and Development, Agent and Digital Banking, Business Intelligent, Legal, Compliance

Rules:
- Approve if the query is clearly related to ANY of these departments (even indirectly)
- Approve both English and Khmer queries
- Reject if the query is unrelated (e.g., general chat, random topics, personal questions)
- Reject if meaningless or gibberish

Query: "{user_input}"

Answer ONLY:
yes or no"""

            response = self.llm.invoke([HumanMessage(content=verification_prompt)])
            response_text = response.content.strip().lower()

            is_valid = "yes" in response_text or "valid" in response_text
            reason = f"AMK_LLM validation: {response_text}"

            logger.info(f"Query verification ({detected_lang}): '{user_input}' -> {is_valid} ({response_text})")

            return (is_valid, reason)
            
        except Exception as e:
            logger.error(f"Error verifying query: {e}")
            # On error, accept query (fail open)
            return (True, f"Verification error, accepting query: {str(e)}")
    
    def refine_query(self, user_input: str) -> str:
        """
        Validates if the query is meaningful.
        Now uses verify_query() for proper evaluation.
        Returns empty string if invalid, original query if valid.
        """
        is_valid, reason = self.verify_query(user_input)
        
        if is_valid:
            logger.info(f"Query accepted: {reason}")
            return user_input
        else:
            logger.warning(f"Query rejected: {reason}")
            return ""

    def _resolve_doc_retrieval_collection(self, doc_id: Optional[str]) -> Optional[str]:
        """Resolve the Qdrant collection that stores a specific document."""
        if not doc_id:
            return None

        try:
            from document.models import Document as DocModel

            doc = (
                DocModel.objects.select_related("type")
                .filter(id=doc_id)
                .only("id", "type__name")
                .first()
            )
            if doc and getattr(doc, "type", None) and getattr(doc.type, "name", None):
                collection_name = str(doc.type.name)
                logger.info(
                    f"Resolved doc_id={doc_id} to collection '{collection_name}' from document metadata"
                )
                return collection_name
        except Exception as e:
            logger.warning(f"Failed to resolve collection from document metadata for doc_id={doc_id}: {e}")

        try:
            skip = {"inactive", self.history_collection_name, self.inactive_history_collection_name}
            collections_resp = self.client.get_collections()
            for col in collections_resp.collections:
                if col.name in skip:
                    continue

                probe = self.client.scroll(
                    collection_name=col.name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=str(doc_id)),
                            ),
                            models.FieldCondition(
                                key="is_active",
                                match=models.MatchValue(value=True),
                            ),
                        ]
                    ),
                    limit=1,
                    with_payload=False,
                    with_vectors=False,
                )
                if probe[0]:
                    logger.info(
                        f"Resolved doc_id={doc_id} to collection '{col.name}' by probing Qdrant"
                    )
                    return col.name
        except Exception as e:
            logger.warning(f"Failed to resolve collection from Qdrant for doc_id={doc_id}: {e}")

        logger.warning(f"Could not resolve a collection for doc_id={doc_id}")
        return None

    def get_related_docs(self, question: str, file_type=None, query_language: str = None, k: int = 5, doc_id: str = None):
        """
        Get related documents with query expansion for short queries.
        Short queries (<=5 words) are expanded using LLM for better semantic matching.
        
        Args:
            doc_id: Optional document ID to filter results to a specific document
        """
        # -- AGENTIC RAG HOOK --------------------------------------------------
        resolved_doc_collection = self._resolve_doc_retrieval_collection(doc_id) if doc_id else None
        retrieval_collection = (
            resolved_doc_collection
            if doc_id
            else (str(file_type) if file_type else None)
        )
        if doc_id:
            logger.info(
                f"Doc-scoped retrieval active for doc_id={doc_id}; "
                f"resolved_collection={retrieval_collection!r}, request_file_type={file_type!r}"
            )
        if self.agentic_rag and not doc_id:
            try:
                # Detect language for the agent
                detected = self.document_loader.detect_languages(question)
                agent_lang = "km" if detected in ("km", "multi") else "en"
                
                # Delegate to AgenticRAG
                return self.agentic_rag.get_docs(
                    question=question,
                    collection_name=retrieval_collection,
                    query_language=query_language,
                    doc_id=doc_id,
                    language=agent_lang
                )
            except Exception as e:
                logger.error(f"[AgenticRAG] Failed during execution, falling back to standard: {e}")
                # Fall through to standard logic below
        # ----------------------------------------------------------------------

        # Skip LLM expansion - use original query for testing
        search_query = question
        
        try:
            logger.info(f"Direct Qdrant search: query='{search_query[:50]}...', k={k}, file_type={file_type}, doc_id={doc_id}")
            
            # Use direct Qdrant search for document retrieval to preserve all metadata
            if retrieval_collection:
                # Search specific collection
                collection_name = retrieval_collection
                if not self.client.collection_exists(collection_name):
                    logger.warning(f"Collection '{collection_name}' does not exist!")
                    return {}
                
                results = self._search_single_collection_direct(
                    collection_name,
                    search_query,
                    k,
                    query_language,
                    source_filter=str(doc_id) if doc_id else None,
                )
                
            else:
                # Search all collections using existing method
                results = self.search_all_collections_direct(search_query, k, query_language)

            if not results:
                logger.warning("No results found in QdrantVectorStore search")
                return {}

            # Filter by specific document ID if provided
            if doc_id:
                logger.info(f"Filtering results to specific document: {doc_id}")
                # Debug: log all source values in results
                source_values = set()
                for result, score in results[:10]:  # Log first 10
                    source_val = result.metadata.get('source')
                    source_values.add(str(source_val))
                logger.debug(f"Source values in results (first 10): {source_values}")
                
                filtered_results = []
                for result, score in results:
                    source_doc_id = result.metadata.get('source')
                    if str(source_doc_id) == str(doc_id):
                        filtered_results.append((result, score))
                results = filtered_results
                logger.info(f"After doc_id filter: {len(results)} results")

            # Guard against stale vectors that reference documents already deleted/inactive
            # in the relational DB. These stale chunks can emit dead image URLs.
            source_ids = {
                str(result.metadata.get("source", "")).strip()
                for result, _score in results
                if str(result.metadata.get("source", "")).strip()
            }
            if source_ids:
                try:
                    from document.models import Document as DocModel

                    active_source_ids = {
                        str(value)
                        for value in DocModel.objects.filter(
                            id__in=list(source_ids),
                            is_active=True,
                        ).values_list("id", flat=True)
                    }

                    filtered_active_results = []
                    dropped_chunks = 0
                    for result, score in results:
                        source_doc_id = str(result.metadata.get("source", "")).strip()
                        if source_doc_id and source_doc_id not in active_source_ids:
                            dropped_chunks += 1
                            continue
                        filtered_active_results.append((result, score))

                    if dropped_chunks:
                        logger.info(
                            "Dropped %s stale chunks from non-active documents (sources=%s)",
                            dropped_chunks,
                            sorted(source_ids),
                        )
                    results = filtered_active_results
                except Exception as e:
                    logger.warning("Failed active-document filtering for retrieval results: %s", e)

            # Take top 5 results (score filtering removed for now)
            results = results[:5]

            # Process results similar to original method
            top_docs_id = []
            doc_chunks = {}
            doc_scores = {}

            for result, score in results:
                chunk_id = result.metadata.get('chunk_id', 'unknown')
                doc_id = result.metadata.get('source', chunk_id)
                logger.info(f"Processing chunk_id: {chunk_id}, doc_id: {doc_id}, score: {score:.4f}")

                if doc_id not in top_docs_id:
                    top_docs_id.append(doc_id)
                    doc_chunks[doc_id] = []
                    doc_scores[doc_id] = []

                # Use chunk_id as unique identifier
                chunk_number = hash(chunk_id) % 10000

                # Check for different page number keys
                page_num = result.metadata.get('page_number') or result.metadata.get('page') or 1
                page_start = result.metadata.get('page_start') or page_num
                page_end = result.metadata.get('page_end') or page_num
                chunk_index = result.metadata.get('chunk_index', 0)
                header = result.metadata.get('header', '')
                header_level = result.metadata.get('header_level', 1)

                # Use page_content first (full content), fallback to content (may be truncated)
                raw_content = result.metadata.get("page_content") or result.metadata.get("content") or result.page_content or ""
                
                logger.debug(f"get_related_docs: header='{header}', content length={len(raw_content)}")
                
                doc_chunks[doc_id].append({
                    "id": chunk_number,
                    "content": raw_content,
                    "page_content": raw_content,  # Store full content separately
                    "html": result.metadata.get("html", ""),
                    "page_image_url": result.metadata.get("page_image_url", ""),
                    "page_image_path": result.metadata.get("page_image_path", ""),
                    "inline_images": result.metadata.get("inline_images", []),
                    "inline_image_urls": result.metadata.get("inline_image_urls", []),
                    "inline_image_url": result.metadata.get("inline_image_url", ""),
                    "inline_image_path": result.metadata.get("inline_image_path", ""),
                    "inline_image_count": result.metadata.get("inline_image_count", 0),
                    "image_directory_path": result.metadata.get("image_directory_path", ""),
                    "image_directory_url": result.metadata.get("image_directory_url", ""),
                    "image_manifest_path": result.metadata.get("image_manifest_path", ""),
                    "image_manifest_url": result.metadata.get("image_manifest_url", ""),
                    "page": int(page_num),
                    "page_start": int(page_start),
                    "page_end": int(page_end),
                    "page_confidence": result.metadata.get("page_confidence", ""),
                    "page_method": result.metadata.get("page_method", ""),
                    "chunk_index": chunk_index,
                    "header": header,
                    "header_level": header_level,
                    "file_name": result.metadata.get('file_name', 'Document'),
                    "score": score,
                    "has_table": result.metadata.get('has_table', False),
                })
                doc_scores[doc_id].append(score)

            # Aggregate scores per document (use max score as document score)
            combined_chunks = {}
            for doc_id in top_docs_id:
                # Sort chunks by chunk_index to preserve document structure
                doc_chunks[doc_id].sort(key=lambda x: (x['chunk_index'], x['id']))

                # Combine chunks with header markers
                combined_text_with_pages = "\n\n".join(
                    f"{chunk['header']}\n{(chunk.get('html') if chunk.get('html') else chunk['content'])}"
                    for chunk in doc_chunks[doc_id]
                )

                # Calculate document score (max of chunk scores)
                max_score = max(doc_scores[doc_id]) if doc_scores[doc_id] else 0.0

                combined_chunks[doc_id] = {
                    "file_name": doc_chunks[doc_id][0]["file_name"],
                    "combined_text": combined_text_with_pages,
                    "chunks": doc_chunks[doc_id],
                    "max_score": max_score
                }

            # Sort documents by max_score descending
            sorted_doc_ids = sorted(
                combined_chunks.keys(),
                key=lambda doc_id: combined_chunks[doc_id]['max_score'],
                reverse=True
            )

            # Reorder combined_chunks by score
            combined_chunks = {doc_id: combined_chunks[doc_id] for doc_id in sorted_doc_ids}

            logger.info(f"Combined chunks keys (doc_ids): {list(combined_chunks.keys())}")
            logger.info(f"Retrieved {len(combined_chunks)} documents, top score: {combined_chunks[sorted_doc_ids[0]]['max_score']:.4f}" if combined_chunks else "No documents retrieved")

            return combined_chunks

        except Exception as e:
            logger.error(f"Error in get_related_docs: {e}")
            # Fallback to original hybrid search
            return self._fallback_hybrid_search(question, file_type, query_language, k)

    def search_all_collections_direct(self, query: str, k: int, query_language: str = None):
        """Search across all collections using direct Qdrant client to preserve all metadata"""
        try:
            # Get all collections from Qdrant
            collections_response = self.client.get_collections()
            collection_names = [col.name for col in collections_response.collections]
            
            # Filter out history collections
            collection_names = [name for name in collection_names 
                              if name not in [self.history_collection_name, self.inactive_history_collection_name]]
            
            logger.debug(f"Available collections for search: {collection_names}")
            
            if not collection_names:
                logger.warning("No collections available for search")
                return []
        except Exception as e:
            logger.error(f"Error getting collections from Qdrant: {e}")
            return []

        all_results = []

        for collection_name in collection_names:
            try:
                logger.debug(f"Searching collection: {collection_name}")
                # Use direct Qdrant search to preserve all payload fields
                direct_results = self._search_single_collection_direct(collection_name, query, k, query_language)
                logger.debug(f"Collection {collection_name} returned {len(direct_results)} results")
                all_results.extend(direct_results)
            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {e}")
                continue

        # Sort by score and take top k
        all_results.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Total results from all collections: {len(all_results)}")
        return all_results[:k]  # Return tuples (doc, score) to match other search methods

    def _search_single_collection_direct(
        self,
        collection_name: str,
        query: str,
        k: int,
        query_language: str = None,
        source_filter: str = None,
    ):
        """Search a single collection using direct Qdrant client to preserve all payload fields (especially 'source').

        Uses hybrid prefetch+fusion via VectorStoreService and does not fallback to dense-only search.
        """
        try:
            points = self.vector_service.query_collection_hybrid(
                collection_name,
                query,
                k,
                query_language,
                source_filter=source_filter,
            )
            results = []
            for hit in points:
                payload = getattr(hit, 'payload', None)
                if payload is None and isinstance(hit, dict):
                    payload = hit.get('payload', {})
                payload = payload or {}

                score = getattr(hit, 'score', None)
                if score is None and isinstance(hit, dict):
                    score = hit.get('score', 0.0)
                score = score or 0.0

                chunk_id = getattr(hit, 'id', None) or (hit.get('id') if isinstance(hit, dict) else None) or payload.get('chunk_id')

                # Check both 'page_content' and 'content' fields for compatibility
                # manual_consolidated_khmer_markitdown uses 'content', others use 'page_content'
                page_content = payload.get("page_content") or payload.get("content", "")

                doc = LangChainDocument(
                    page_content=page_content,
                    metadata={
                        "chunk_id": payload.get("chunk_id", str(chunk_id)),
                        "score": score,
                        "language": payload.get("language"),
                        "source": payload.get("source"),
                        "file_name": payload.get("file_name"),
                        "page_number": payload.get("page_number"),
                        "page_start": payload.get("page_start"),
                        "page_end": payload.get("page_end"),
                        "page_confidence": payload.get("page_confidence"),
                        "page_method": payload.get("page_method"),
                        "file_type": collection_name,
                        "chunk_index": payload.get("chunk_index", 0),
                        "header": payload.get("header", ""),
                        "html": payload.get("html", ""),
                    }
                )
                results.append((doc, score))

            logger.info(f"Hybrid direct search in {collection_name}: found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Hybrid direct search failed for {collection_name}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []


    def _fallback_hybrid_search(self, question: str, file_type, query_language, k: int):
        """Fallback to direct Qdrant search if direct search fails"""
        logger.warning("Falling back to direct Qdrant search")
        try:
            # Use direct Qdrant search as fallback
            return self.get_related_docs(question, file_type, query_language, k)
        except Exception as e:
            logger.error(f"Fallback direct search also failed: {e}")
            return {}

    def reuse_history(self, question: str, question_embedded, language_name, session_id: str = None):
        # If we don't have embeddings available, skip history reuse gracefully
        if question_embedded is None:
            logger.info("No question embedding available - skipping history reuse.")
            return []

        # Get all history points
        scroll_filter = None
        if session_id:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    )
                ]
            )
        scroll_result = self.client.scroll(
            collection_name=self.history_collection_name,
            scroll_filter=scroll_filter,
            limit=100
        )
        points = scroll_result[0]

        if not points:
            logger.info("No similar questions found in history.[1]")
            return []

        try:
            # Create similar_questions as LangChain documents
            similar_questions = []
            embeddings = []
            for point in points:
                payload = point.payload
                doc = LangChainDocument(
                    page_content=payload.get("question", ""),
                    metadata=payload
                )
                doc.id = point.id  # Add id attribute
                similar_questions.append(doc)
                embeddings.append(point.vector)

            embeddings_tensor = np.array(embeddings, dtype=np.float32)
            question_tensor = np.array(question_embedded, dtype=np.float32)

            # Using cosine similarity for history matching
            cos_sin_score = util.cos_sim(question_tensor, embeddings_tensor)
            cos_scores = cos_sin_score[0].cpu().numpy() if hasattr(cos_sin_score[0], 'cpu') else cos_sin_score[0].numpy()
            mask = cos_scores > 0.40
            related_questions = np.array(similar_questions)[mask]
            feedback_docs = self.user_feedback_compute(related_questions)

            compute_relation = util.semantic_search(
                question_tensor, embeddings_tensor, top_k=3, score_function=util.cos_sim
            )

            questions = []
            compute_relation_90 = []
            # Track complete answer data from history for restoration on page refresh
            history_answer_data = None

            first_history_candidate = None
            per_answer_history_candidate = None
            for item in compute_relation[0]:
                if item['score'] > 0.40:
                    answer_str = similar_questions[item['corpus_id']].metadata.get("answer", "")
                    try:
                        answer_data = json.loads(answer_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Attempting to fix malformed data in history: {answer_str}")
                        # Attempt to fix single quotes to double quotes for backward compatibility
                        answer_str = answer_str.replace("'", '"')
                        try:
                            answer_data = json.loads(answer_str)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to fix and parse data from history metadata: {e}, content: {answer_str}")
                            continue # Skip this malformed entry

                    # Handle both new format (with answers array) and old format (string/dict without answers array)
                    if isinstance(answer_data, dict) and "answers" in answer_data:
                        answer_list = answer_data["answers"]
                    else:
                        # Migrate old format: convert string/single answer to answers array
                        try:
                            if isinstance(answer_data, str):
                                # Old format: plain text answer
                                answer_list = [{"text": answer_data, "doc_id": "", "file_name": ""}]
                                answer_data = {"answers": answer_list}
                                logger.info("Migrated old string format to answers array")
                            elif isinstance(answer_data, dict) and "text" in answer_data:
                                # Old format: dict with text field instead of answers
                                answer_list = [answer_data]
                                answer_data = {"answers": answer_list}
                                logger.info("Migrated old dict format to answers array")
                            else:
                                logger.warning(f"Unknown history format, skipping: {answer_str}")
                                continue
                        except Exception as e:
                            logger.warning(f"Failed to migrate old history format: {e}")
                            continue
                    
                    if answer_list and len(answer_list) > 0 and feedback_docs.get(answer_list[0].get("doc_id"), True):
                        compute_relation_90.append(item)
                        questions.append(similar_questions[item['corpus_id']].page_content)
                        logger.info(similar_questions[item['corpus_id']].page_content)

                        # Remember the first valid history candidate (fallback)
                        if not first_history_candidate:
                            first_history_candidate = answer_data
                            logger.debug(
                                f"Candidate history found with {len(answer_data.get('document_references', []))} document references"
                            )

                        # Prefer history entries that contain per-answer document_reference(s)
                        has_per_answer_ref = False
                        try:
                            for ans in answer_list:
                                if isinstance(ans, dict) and ("document_reference" in ans or "document_references" in ans):
                                    has_per_answer_ref = True
                                    break
                        except Exception:
                            has_per_answer_ref = False

                        if has_per_answer_ref:
                            per_answer_history_candidate = answer_data
                            logger.info("Found history entry containing per-answer document_reference; preferring this one")
                            break

            # Prefer per-answer history if found, otherwise fall back to first valid candidate
            if per_answer_history_candidate:
                history_answer_data = per_answer_history_candidate
                doc_refs_count = len(history_answer_data.get('document_references', [])) if isinstance(history_answer_data, dict) else 0
                logger.info(f"Restored complete per-answer history data from history with {doc_refs_count} document references")
            elif first_history_candidate and not history_answer_data:
                history_answer_data = first_history_candidate
                doc_refs_count = len(history_answer_data.get('document_references', [])) if isinstance(history_answer_data, dict) else 0
                logger.info(f"Restored complete answer data from history with {doc_refs_count} document references")

             # Question verification is disabled - history reuse is disabled
            # if len(questions) < 5:
            #     logger.info("Not enough questions checking.")
            #     return []
            #
            # # Fixed prompt formatting
            # prompt = self.verify_question.format(
            #     question=question,
            #     similar_questions=questions
            # )
            #
            # response = self.llm.invoke([HumanMessage(content=prompt)])
            # self.token_logger.log_token_usage(
            #     model=settings.OLLAMA_MODEL,
            #     operation="verify_question",
            #     usage_metadata=response.usage_metadata,
            #     meta_json={"question": question}
            # )
            #
            # response_content = response.content.strip()
            # try:
            #     idxs = ast.literal_eval(response_content)
            # except (ValueError, SyntaxError):
            #     logger.error(f"Failed to parse response indices: {response_content}")
            #     return []
            #
            # if not idxs or len(idxs) != 5:
            #     logger.info("No similar questions found in history.[2]")
            #     return []
            # Return complete answer data with document_references for page refresh restoration
            if history_answer_data:
                # Ensure language is included (set it if missing)
                if "language" not in history_answer_data:
                    history_answer_data["language"] = language_name
                return history_answer_data
            return []

        except Exception as e:
            logger.error(f"Error in reuse_history: {e}")
            return []

    def safe_historical_handling(self, question: str, answer: str, question_id: str, session_id: str = None) -> bool:
        """ Safely handle historical data with proper error handling"""
        try:
            doc_ids = []
            try:
                answer_json = json.loads(answer)
                if isinstance(answer_json, dict) and "answers" in answer_json:
                    answer_lst = answer_json["answers"]
                    doc_ids = [ans.get('doc_id') for ans in answer_lst if ans.get('doc_id')]
                else:
                    doc_ids = [] # Handle cases where it's a string but not the expected JSON format
            except Exception as e:
                logger.error(f"Error parsing answer for doc_ids in safe_historical_handling: {str(e)}")
                doc_ids = []

            # Convert list to comma-separated string
            doc_ids_str = ','.join(doc_ids) if doc_ids else ""

            # Embed the question
            embedding = self.vector_service.embedding.embed_query(question)

            payload = {
                "id": str(question_id),
                "question": question,
                "answer": str(answer),  # Keep as string representation
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "related_doc": doc_ids_str,
                "session_id": str(session_id) if session_id else None,
                # Preserve any document_references present in the answer JSON or provided separately
                "document_references": (json.loads(answer).get("document_references") if isinstance(answer, str) and answer.strip().startswith("{") else None)
            }

            point = models.PointStruct(
                id=str(question_id),
                vector=embedding,
                payload=payload
            )

            # Add to the history collection
            self.client.upsert(collection_name=self.history_collection_name, points=[point])
            logger.info(f"Successfully added to history: question_id={question_id}, session_id={session_id}")
            return True

        except Exception as e:
            logger.error(f"Error in safe_historical_handling: {e}")
            return False

    def deactivate_history(self, file_id: str, session_id: str = None):
        """ Move history answer to inactive collection"""

        scroll_filter = None
        if session_id:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    )
                ]
            )

        scroll_result = self.client.scroll(
            collection_name=self.history_collection_name,
            scroll_filter=scroll_filter,
            limit=10000,
            with_vectors=True
        )
        points = scroll_result[0]

        ans_to_move = []
        ids_to_move = []

        for point in points:
            payload = point.payload
            related_doc_ids = payload.get('related_doc', '').split(',')

            if file_id in related_doc_ids:
                payload_copy = payload.copy()
                payload_copy['is_active'] = False
                point_to_move = models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=payload_copy
                )
                ans_to_move.append(point_to_move)
                ids_to_move.append(point.id)

        if not ans_to_move:
            logger.info("No history found")
            return False

        # move to inactive
        self.client.upsert(collection_name=self.inactive_history_collection_name, points=ans_to_move)

        # Verify
        verify_scroll = self.client.scroll(
            collection_name=self.inactive_history_collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="id",
                        match=models.MatchValue(value=ids_to_move[0])  # Check first one
                    )
                ]
            ),
            limit=1
        )
        if not verify_scroll[0]:
            logger.warning("Warning: History entry was not moved to inactive collection")
            return False

        self.client.delete(
            collection_name=self.history_collection_name,
            points_selector=models.PointIdsList(points=ids_to_move)
        )
        return True

    def activate_history(self, file_id : str, session_id: str = None):
        """ Move history answer back to collection"""

        scroll_filter = None
        if session_id:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id)
                    )
                ]
            )

        scroll_result = self.client.scroll(
            collection_name=self.inactive_history_collection_name,
            scroll_filter=scroll_filter,
            limit=10000,
            with_vectors=True
        )
        points = scroll_result[0]

        ans_to_move = []
        ids_to_move = []

        for point in points:
            payload = point.payload
            related_doc_ids = payload.get('related_doc', '').split(',')

            if file_id in related_doc_ids:
                payload_copy = payload.copy()
                payload_copy['is_active'] = True
                point_to_move = models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=payload_copy
                )
                ans_to_move.append(point_to_move)
                ids_to_move.append(point.id)

        if ans_to_move:
            self.client.upsert(collection_name=self.history_collection_name, points=ans_to_move)

            self.client.delete(
                collection_name=self.inactive_history_collection_name,
                points_selector=models.PointIdsList(points=ids_to_move)
            )

        return True

    def user_feedback_compute(self, most_related_questions):
        doc_dict = {}
        for item in most_related_questions:
            metadata = item.metadata
            ans_str = metadata["answer"]
            try:
                ans_json = json.loads(ans_str)
            except json.JSONDecodeError:
                logger.warning(f"Attempting to fix malformed data in history for feedback: {ans_str}")
                # Attempt to fix single quotes to double quotes for backward compatibility
                ans_str = ans_str.replace("'", '"')
                try:
                    ans_json = json.loads(ans_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to fix and parse data from history metadata for feedback: {e}, content: {ans_str}")
                    continue # Skip this malformed entry

            if isinstance(ans_json, dict) and "answers" in ans_json and ans_json["answers"]:
                ans = ans_json["answers"][0]
                if ans["doc_id"] not in doc_dict:
                    doc_dict[ans["doc_id"]] = {
                        "1":0,
                        "0":0,
                        "-1":0
                    }
            else:
                logger.warning(f"Unexpected answer format in history metadata for feedback: {ans_str}")
                continue
            is_relevant = ans.get("is_relevant", 0)  # Default to 0 (neutral) if key doesn't exist
            doc_dict[ans["doc_id"]][str(is_relevant)] += 1
        result = {}
        for doc_id, counts in doc_dict.items():
            if (counts["1"] + counts["-1"]>5) and (counts["1"] < counts["-1"]):
                result[doc_id] = False
            else:
                result[doc_id] = True
        return result

    def _get_no_info_response(self, question_id: str, language_name: str, simplified_output: bool):
        no_info_message = "រកមិនឃើញព័ត៌មានពាក់ព័ន្ធទេ។ សូមសាកល្បងសួរជាភាសាអង់គ្លេសជំនួសវិញ។" if language_name == "Khmer" else "No relevant information found. Please try to ask in the Khmer language instead."
        if simplified_output:
            return {
                "answers": [no_info_message],
                "combined_answer": no_info_message,
                "language": language_name,
                "document_references": []
            }
        return {
            "answers": [{
                "question_id": question_id,
                "text": no_info_message,
                "file_name": "",
                "doc_id": "",
                "id": 0,
                "is_relevant": 0
            }],
            "language": language_name,
            "document_references": []
        }

    def _get_user_display_score_threshold(self, detected_language: str, use_agentic_retrieval: bool) -> float:
        if detected_language == "km":
            return LEGACY_DISPLAY_THRESHOLD_KM
        return LEGACY_DISPLAY_THRESHOLD_EN

    @staticmethod
    def _extract_display_score(item: dict) -> float:
        if not isinstance(item, dict):
            return 0.0

        candidates = [item.get("score"), item.get("max_score")]
        if isinstance(item.get("document_reference"), dict):
            candidates.append(item["document_reference"].get("max_score"))

        parsed_scores: List[float] = []
        for value in candidates:
            try:
                parsed_scores.append(float(value))
            except (TypeError, ValueError):
                continue

        if not parsed_scores:
            return 0.0
        return max(parsed_scores)

    def _filter_user_visible_results(
        self,
        answers: List[dict],
        document_references: List[dict],
        detected_language: str,
        use_agentic_retrieval: bool,
    ) -> tuple[List[dict], List[dict], float]:
        score_threshold = self._get_user_display_score_threshold(
            detected_language,
            use_agentic_retrieval,
        )

        filtered_answers = [
            answer for answer in answers
            if self._extract_display_score(answer) >= score_threshold
        ]
        visible_doc_ids = {
            answer.get("doc_id")
            for answer in filtered_answers
            if isinstance(answer, dict) and answer.get("doc_id")
        }
        filtered_document_references = [
            ref for ref in document_references
            if self._extract_display_score(ref) >= score_threshold
            and (not visible_doc_ids or ref.get("doc_id") in visible_doc_ids)
        ]

        logger.info(
            f"Applied user-visible score threshold {score_threshold:.2f} "
            f"for language={detected_language} agentic={use_agentic_retrieval}: "
            f"answers {len(answers)}->{len(filtered_answers)}, "
            f"document_references {len(document_references)}->{len(filtered_document_references)}"
        )

        return filtered_answers, filtered_document_references, score_threshold

    def _format_response_output(self, text: str) -> str:
        """Post-process response text - minimal cleanup."""
        if not text or text.strip() == "NOT_FOUND":
            return text

        # Fix malformed markup patterns like bRPO/b -> RPO
        text = re.sub(r'b([^/]+)/b', r'**\\1**', text)

        # Clean up multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()

    def _is_html_table(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r'<\s*(table|tr|td|th)\b', text, flags=re.IGNORECASE))

    @staticmethod
    def _looks_like_html(text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r'<\s*(div|p|table|ul|ol|li|img|h[1-6]|br|strong|em)\b', text, flags=re.IGNORECASE))

    def _to_display_html(self, text: str) -> str:
        value = (text or "").strip()
        if not value:
            return ""
        if self._looks_like_html(value):
            return value
        try:
            return py_markdown.markdown(
                value,
                extensions=["tables", "sane_lists", "nl2br"],
                output_format="html5",
            )
        except Exception:
            logger.warning("Failed converting markdown answer to HTML; returning raw text")
            return value

    @staticmethod
    def _dedupe_markdown_images(text: str) -> str:
        if not text:
            return ""
        seen: set[str] = set()
        pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

        def replacement(match: re.Match[str]) -> str:
            src = (match.group(2) or "").strip()
            if not src:
                return match.group(0)
            key = src.lower()
            if key in seen:
                return ""
            seen.add(key)
            return match.group(0)

        output = pattern.sub(replacement, text)
        return re.sub(r"\n{3,}", "\n\n", output).strip()

    def _dedupe_images_in_display_html(self, text: str) -> str:
        value = (text or "").strip()
        if not value:
            return ""

        value = self._dedupe_markdown_images(value)
        if not self._looks_like_html(value):
            return value

        seen: set[str] = set()
        img_pattern = re.compile(
            r'<img\b[^>]*\bsrc\s*=\s*["\']([^"\']+)["\'][^>]*>',
            flags=re.IGNORECASE,
        )

        def replacement(match: re.Match[str]) -> str:
            src = (match.group(1) or "").strip()
            if not src:
                return match.group(0)
            key = src.lower()
            if key in seen:
                return ""
            seen.add(key)
            return match.group(0)

        deduped = img_pattern.sub(replacement, value)
        deduped = re.sub(r"<p>\s*</p>", "", deduped, flags=re.IGNORECASE)
        deduped = re.sub(r"\n{3,}", "\n\n", deduped)
        return deduped.strip()

    def _strip_images_from_reference_html(self, text: str) -> str:
        value = self._dedupe_images_in_display_html(text)
        if not value:
            return ""

        # Remove markdown image syntax if present.
        value = re.sub(r'!\[[^\]]*\]\(([^)]+)\)', '', value)
        # Remove HTML <img ...> tags for source-document cards.
        value = re.sub(r'<img\b[^>]*>', '', value, flags=re.IGNORECASE)
        value = re.sub(r"<p>\s*</p>", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()

    @staticmethod
    def _has_structured_markdown(text: str) -> bool:
        value = (text or "").strip()
        if not value:
            return False
        has_table = bool(re.search(r"(?:^|\n)\|.*\|(?:\n|$)", value))
        has_list = bool(re.search(r"(?:^|\n)\s*(?:[-*+]\s+|\d+\.\s+)", value))
        return has_table or has_list

    def _strip_chunk_header_from_body(self, text: str, header: str) -> str:
        body = (text or "").strip()
        header = (header or "").strip()
        if not body:
            return ""
        if header:
            body = re.sub(r'^#+\s*' + re.escape(header) + r'\s*\n*', '', body, count=1)
            if body.lstrip().startswith(header):
                body = body.lstrip()[len(header):].lstrip('\n').lstrip()
        return body.strip()

    @staticmethod
    def _contains_preamble_keyword(text: str) -> bool:
        lowered = (text or "").lower()
        return any(
            k in lowered
            for k in (
                "preamble",
                "preface",
                "abbreviation",
                "abbreviations",
                "abbreviated",
                "acronym",
                "acronyms",
                "short form",
                "short forms",
                "preamble line",
                "preamble lines",
                "preamble_lines",
                "បុព្វកថា",
                "អក្សរកាត់",
            )
        )

    @staticmethod
    def _contains_abbreviation_keyword(text: str) -> bool:
        lowered = (text or "").lower()
        return any(
            k in lowered
            for k in (
                "abbreviation",
                "abbreviations",
                "abbreviated",
                "acronym",
                "acronyms",
                "short form",
                "short forms",
                "អក្សរកាត់",
            )
        )

    @staticmethod
    def _is_no_answer_text(text: str) -> bool:
        lowered = (text or "").lower()
        no_answer_phrases = (
            "not available",
            "i am sorry",
            "cannot find",
            "not found",
            "not in the provided context",
            "does not contain",
        )
        return any(phrase in lowered for phrase in no_answer_phrases)

    def _determine_query_focus(self, question: str, headers: List[str]) -> str:
        """
        Decide answer formatting strategy:
          - header   : header + content
          - content  : content + header
          - preamble : llm_content extraction
        """
        normalized_q = normalize_user_query(question)
        if not normalized_q:
            return "content"

        if self._contains_preamble_keyword(normalized_q):
            return "preamble"

        q_lower = normalized_q.lower()
        has_meaningful_chars = bool(re.search(r"[a-z0-9\u1780-\u17ff]", q_lower))
        if not has_meaningful_chars:
            # Some clients/files can degrade Khmer text into "????...".
            # In that case, prefer preamble route if any retrieved header hints preamble/abbreviations.
            for header in headers:
                if self._contains_preamble_keyword(header or ""):
                    return "preamble"
            return "content"

        q_tokens = set(re.findall(r"[a-z0-9]+", q_lower))

        for header in headers:
            h = (header or "").strip()
            if not h or h.lower() == "content":
                continue
            h_lower = h.lower()
            if len(normalized_q) >= 4 and (q_lower in h_lower or h_lower in q_lower):
                return "header"
            h_tokens = set(re.findall(r"[a-z0-9]+", h_lower))
            if q_tokens and h_tokens and len(q_tokens & h_tokens) >= max(2, len(q_tokens) // 2):
                return "header"

        return "content"

    def _select_llm_content_chunk(self, chunks: List[Dict]) -> Dict:
        """
        Pick the best chunk for llm_content, prioritizing legacy behavior:
        1) header == "Content"
        2) preamble-like header/body
        3) first available chunk
        """
        if not chunks:
            return {}

        for chunk in chunks:
            if (chunk.get("header") or "").strip() == "Content":
                return chunk

        for chunk in chunks:
            header = (chunk.get("header") or "")
            text = (chunk.get("text") or "")
            if self._contains_preamble_keyword(header) or self._contains_preamble_keyword(text):
                return chunk

        return chunks[0]

    def _select_preamble_chunk(self, chunks: List[Dict], question: str = "") -> Dict:
        """
        Select the most likely preamble chunk with a deterministic fallback.

        Priority:
        1) Chunk whose header/text contains a preamble keyword.
        2) Earliest chunk in document order (page, chunk_index), preferring header "Content".
        """
        if not chunks:
            return {}

        abbreviation_query = self._contains_abbreviation_keyword(question or "")

        abbreviation_hits = [
            chunk for chunk in chunks
            if self._contains_abbreviation_keyword(chunk.get("header", ""))
            or self._contains_abbreviation_keyword(chunk.get("text", ""))
        ]
        if abbreviation_query and abbreviation_hits:
            pool = abbreviation_hits
        else:
            keyword_hits = [
            chunk for chunk in chunks
            if self._contains_preamble_keyword(chunk.get("header", ""))
            or self._contains_preamble_keyword(chunk.get("text", ""))
            ]
            pool = keyword_hits or chunks

        def chunk_sort_key(chunk: Dict) -> tuple:
            ref = chunk.get("reference", {}) if isinstance(chunk, dict) else {}
            header = (chunk.get("header") or "").strip() if isinstance(chunk, dict) else ""
            page = ref.get("page", 10**9)
            chunk_index = chunk.get("chunk_index", 10**9) if isinstance(chunk, dict) else 10**9
            score = float(chunk.get("score", 0.0) or 0.0) if isinstance(chunk, dict) else 0.0
            text = (chunk.get("text") or "") if isinstance(chunk, dict) else ""
            has_table = 1 if self._has_structured_markdown(text) else 0
            # Prefer Content header, then earliest page/index, then higher score.
            # For abbreviation queries, prefer table-like chunks first.
            table_rank = 0 if (abbreviation_query and has_table) else 1
            return (table_rank, 0 if header == "Content" else 1, int(page), int(chunk_index), -score)

        return sorted(pool, key=chunk_sort_key)[0]

    @staticmethod
    def _normalize_star_separated_table_list(text: str) -> str:
        """
        Convert single-line star-separated table content into markdown list lines.
        Example:
          'a * b * c' -> '- a\\n- b\\n- c'
        """
        value = (text or "").strip()
        if not value or "*" not in value:
            return value
        # Skip already-structured HTML fragments.
        if "<" in value and ">" in value:
            return value

        # Split only on standalone '*' separators (not '**bold**').
        collapsed = re.sub(r"\s+", " ", value).strip()
        parts = [
            part.strip()
            for part in re.split(r"\s+(?<!\*)\*(?!\*)\s+", collapsed)
            if part.strip()
        ]
        cleaned_parts = []
        for part in parts:
            cleaned = re.sub(r"^\s*(?:[-•*]\s+)+", "", part).strip()
            if cleaned:
                cleaned_parts.append(cleaned)
        parts = cleaned_parts

        # Require multiple separators to avoid converting formulas like "A * B".
        if len(parts) < 3:
            return value

        if parts[0].endswith((":", "៖")) and len(parts) > 2:
            return parts[0] + "\n" + "\n".join(f"- {item}" for item in parts[1:])
        return "\n".join(f"- {item}" for item in parts)

    def _render_table_cell_inline_markdown(self, cell_text: str) -> str:
        """Render inline markdown for a table header/cell value."""
        value = cleanup_bold_colon((cell_text or "").strip())
        normalized_value = self._normalize_star_separated_table_list(value)
        force_dash_prefix = normalized_value != value
        value = normalized_value
        if not value:
            return ""

        try:
            rendered = py_markdown.markdown(
                value,
                extensions=["sane_lists"],
                output_format="html5",
            ).strip()
        except Exception:
            return value

        if force_dash_prefix and "<li" in rendered:
            # Only inject a visible dash for plain-text list items.
            rendered = re.sub(r"<li>\s*(?![-•<])", "<li>- ", rendered)

        paragraph_match = re.fullmatch(r"<p>(.*)</p>", rendered, flags=re.DOTALL)
        if paragraph_match:
            return paragraph_match.group(1).strip()
        return rendered

    def _convert_markdown_table_to_html(self, text: str) -> str:
        """Convert markdown pipe tables to HTML tables. More robust conversion."""
        try:
            lines = text.splitlines()
            result_lines = []
            i = 0

            while i < len(lines):
                line = lines[i]

                if '|' in line and line.strip():
                    table_lines = []
                    while i < len(lines) and '|' in lines[i]:
                        table_lines.append(lines[i])
                        i += 1

                    if len(table_lines) >= 1:
                        separator_idx = -1
                        for idx, tl in enumerate(table_lines):
                            stripped = tl.strip().replace('|', '').replace(' ', '').replace('-', '').replace(':', '')
                            if not stripped and ('-' in tl or ':' in tl):
                                separator_idx = idx
                                break

                        rows = []
                        header_cells = []

                        if separator_idx >= 0:
                            if separator_idx == 0:
                                body_lines = table_lines[separator_idx + 1:]
                                if body_lines:
                                    first_row_cells = [c.strip() for c in body_lines[0].strip().strip('|').split('|')]
                                    header_cells = [""] * len(first_row_cells)
                            else:
                                header_line = table_lines[0]
                                body_lines = table_lines[separator_idx + 1:]
                                header_cells = [c.strip() for c in header_line.strip().strip('|').split('|')]

                            for bl in body_lines:
                                cells = [c.strip() for c in bl.strip().strip('|').split('|')]
                                if cells and any(c for c in cells):
                                    rows.append(cells)
                        else:
                            header_cells = [c.strip() for c in table_lines[0].strip().strip('|').split('|')]
                            body_lines = table_lines[1:]

                            for bl in body_lines:
                                cells = [c.strip() for c in bl.strip().strip('|').split('|')]
                                if cells and any(c for c in cells):
                                    rows.append(cells)

                        html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.875rem;">\n'

                        if header_cells and any(h for h in header_cells):
                            html += '<thead style="background-color: var(--primary); color: white;"><tr>\n'
                            for h in header_cells:
                                rendered_header = self._render_table_cell_inline_markdown(h)
                                html += f'<th style="border: 1px solid #dee2e6; padding: 0.75rem 1rem; text-align: left; font-weight: 600; white-space: normal; word-break: normal; overflow-wrap: break-word;">{rendered_header}</th>\n'
                            html += '</tr></thead>\n'

                        num_cols = len(header_cells) if header_cells else (len(rows[0]) if rows else 0)
                        html += '<tbody>\n'
                        for row in rows:
                            if num_cols:
                                row = (row + [''] * num_cols)[:num_cols]
                            html += '<tr>\n'
                            for cell in row:
                                rendered_cell = self._render_table_cell_inline_markdown(cell)
                                html += f'<td style="border: 1px solid #dee2e6; padding: 0.75rem 1rem; background: white; color: #1a1a1a; font-weight: 500; white-space: normal; word-break: normal; overflow-wrap: break-word;">{rendered_cell}</td>\n'
                            html += '</tr>\n'

                        html += '</tbody>\n</table></div>'
                        result_lines.append(html)
                    else:
                        result_lines.extend(table_lines)
                else:
                    result_lines.append(line)
                    i += 1

            return '\n'.join(result_lines)
        except Exception as e:
            logger.error(f"Error converting markdown table to HTML: {e}")
            return text

    def llm_content(self, chunk_data: dict, question: str, language_name: str) -> str:
        """
        Called when the top-1 retrieved chunk needs strict in-context extraction.

        The payload stores two fields:
          - 'content'      : clean body text (no markdown header prefix)
          - 'page_content' : "# Content  <title>\n<body>" (markitdown format)

        Flow:
          1. Use 'content' directly if present; otherwise strip the leading
             "# Content " line from 'page_content'.
          2. Send body + question to LLM with a strict grounded-answer prompt.
          3. Return the LLM answer, or an empty result if the answer is not
             grounded in the provided context.
          4. Convert markdown tables to HTML if present (matches production logic
             so table cells containing sequences like **/** render literally
             instead of being mis-parsed by the frontend markdown renderer).

        Chunks whose header != 'Content' never reach this function response()
        returns them as "**header**\n\nbody" directly (old method).
        """
        try:
            # Use the clean 'content' field first (no "# Content" prefix).
            # Fall back to 'page_content' and strip the header line manually.
            body = (chunk_data.get('content') or '').strip()

            if not body:
                raw = (chunk_data.get('page_content') or '').strip()
                if not raw:
                    logger.warning("llm_content: chunk has no text, returning empty")
                    return ''
                # Strip the first markdown header line injected by markitdown
                # e.g, "# Content\nbody"?  "body"
                lines = raw.splitlines()
                body_lines = []
                header_removed = False
                for line in lines:
                    if not header_removed and re.match(r'^#{1,6}\s', line.strip()):
                        header_removed = True
                        continue
                    body_lines.append(line)
                body = '\n'.join(body_lines).strip() or raw

            logger.info(f"llm_content: body_length={len(body)}, question='{question[:80]}'")

            prompt = (
                self.llm_content_prompt_template
                .replace("{body}", body)
                .replace("{question}", question)
            )

            logger.info("llm_content: invoking LLM")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = (response.content or "").strip()

            if result:
                logger.info(f"llm_content: got answer, length={len(result)}")

                if self._is_no_answer_text(result):
                    logger.info("llm_content: LLM reported no grounded answer in context")
                    return ""

                if '|' in result:
                    is_table = any(
                        re.match(r'^[\s|:-]+$', line.strip()) and '-' in line
                        for line in result.splitlines()
                        if line.strip()
                    )

                    if is_table:
                        logger.info("llm_content: markdown table detected, converting to HTML")
                        try:
                            converted = self._convert_markdown_table_to_html(result)
                            logger.info(f"llm_content: table conversion successful, result length={len(converted)}")
                            return converted
                        except Exception as e:
                            logger.warning(f"llm_content: table conversion failed: {e}, returning original markdown")
                            return result

                return result

            logger.warning("llm_content: LLM returned empty, treating as no grounded answer")
            return ""

        except Exception as e:
            logger.error(f"llm_content error: {e}")
            return ""

    def _synthesize_answer(self, question: str, chunks: List[Dict], language: str, is_content_header: bool = False) -> str:
        """Synthesize an answer from multiple chunks with citations."""
        try:
            from ..config import SYNTHESIS_PROMPT, CONTENT_SYNTHESIS_PROMPT
            
            context_parts = []
            for i, chunk in enumerate(chunks):
                ref = chunk.get("reference", {})
                # Format: [section]
                # User requested removal of document title and page number from context headers
                section = chunk.get('header', 'Content')
                
                header_line = f"[{section}]"
                content = chunk.get("text", "")
                context_parts.append(f"{header_line}\n{content}")
                
            context_str = "\n\n".join(context_parts)
            
            if is_content_header:
                # Use the permissive prompt for Table of Contents/Content requests
                # This prompt mimics the legacy llm_content behavior
                prompt = CONTENT_SYNTHESIS_PROMPT.format(
                    context=context_str,
                    question=question,
                    language=language
                )
                logger.info(f"_synthesize_answer: using CONTENT_SYNTHESIS_PROMPT for {len(chunks)} chunks")
            else:
                prompt = SYNTHESIS_PROMPT.format(
                    context=context_str,
                    question=question,
                    language=language
                )
                # Append strict language enforcement
                if "Khmer" in language or language == "km":
                     prompt += "\nCRITICAL: You MUST answer in Khmer script. Do not use English unless the specific technical term is only known in English."
                
                logger.info(f"_synthesize_answer: invoking LLM with {len(chunks)} chunks")

            msg = self.llm.invoke([HumanMessage(content=prompt)])
            result = (msg.content or "").strip()

            # Check for markdown table and convert to HTML if needed (matches production).
            if '|' in result:
                is_table = any(
                    re.match(r'^[\s|:-]+$', line.strip()) and '-' in line
                    for line in result.splitlines()
                    if line.strip()
                )

                if is_table:
                    logger.info("_synthesize_answer: markdown table detected, converting to HTML")
                    try:
                        if hasattr(self, '_convert_markdown_table_to_html'):
                            converted = self._convert_markdown_table_to_html(result)
                            logger.info(f"_synthesize_answer: table conversion successful, result length={len(converted)}")
                            return converted
                    except Exception as e:
                        logger.warning(f"_synthesize_answer: table conversion failed: {e}, returning original markdown")
                        return result

            return result

        except Exception as e:
            logger.error(f"_synthesize_answer failed: {e}")
            # Fallback to first chunk content
            if chunks:
                return chunks[0].get("text", "")
            return ""

    def response(self, question, question_id=None, file_type=None, simplified_output: bool = False, session_id=None, doc_id: str = None):
        if not question_id:
            question_id = str(uuid.uuid4())
        logger.info(f"Starting response generation for question_id: {question_id}")

        original_question = question or ""
        raw_question = strip_mention_prefix(original_question)

        question = normalize_user_query(raw_question)
        if original_question != question:
            logger.info(
                f"Sanitized mentioned question for retrieval: '{original_question[:120]}' -> '{question[:120]}'"
            )
        llm_question = (raw_question or question or "").strip()

        # Track start time for performance evaluation
        start_time = time.time()

        # Check memory status at the beginning
        memory_status = MemoryManager.get_memory_status()
        logger.info(f"Memory status at start: {memory_status}")

        # Force cleanup if critical memory level
        if memory_status['needs_critical_cleanup']:
            logger.warning("CRITICAL: Memory usage too high, forcing cleanup before processing")
            MemoryManager.force_cleanup()
            MemoryManager.force_gpu_cleanup()

        # Validate input query
        if not question or not isinstance(question, str) or not question.strip():
            logger.warning(f"Invalid query: empty or None")
            self.token_logger.finish_query_session(
                self.token_logger.start_query_session(query_id=question_id, question=question)
            )
            return self._get_no_info_response(question_id, "English", simplified_output)

        # Validate query language before retrieval
        try:
            detected_lang = self.document_loader.detect_languages(question.strip())
            if detected_lang not in ['km', 'en', 'multi']:
                logger.warning(f"Query detected as unsupported language: {detected_lang}. Query: {question}")
                self.token_logger.finish_query_session(
                    self.token_logger.start_query_session(query_id=question_id, question=question)
                )
                return self._get_no_info_response(question_id, "English", simplified_output)
        except Exception as e:
            logger.error(f"Error validating query language: {e}")
            # Continue processing on error

        query_session_id = self.token_logger.start_query_session(
            query_id=question_id,
            question=question,
        )

        # Ensure cleanup happens even if an exception occurs
        try:
            from ..config import (
                FOLLOWUP_SUGGESTIONS_COUNT,
                FOLLOWUP_SUGGESTIONS_ENABLED,
                QUERY_VALIDATION_ENABLED,
                RETRIEVAL_K,
            )

            # Step 0: Get conversation context and reformulate query if needed
            logger.info(f"Step 0: Getting conversation context...")
            conversation_context = self._get_conversation_context(session_id, max_history=3)
            
            if conversation_context:
                # Skip LLM reformulation - use original question for testing
                logger.info("Skipping query reformulation - using original question")
            
            # Step 1: Use the original normalized query directly
            logger.info("Step 1: Query refinement removed - using original query")
            retrieval_query = question

            if QUERY_VALIDATION_ENABLED:
                is_valid_query, validation_reason = self.verify_query(retrieval_query)
                if not is_valid_query:
                    logger.warning(
                        f"Query validation rejected input. Query: {retrieval_query}, "
                        f"reason: {validation_reason}"
                    )
                    self.token_logger.finish_query_session(query_session_id)
                    try:
                        detected_lang = self.document_loader.detect_languages(retrieval_query)
                        response_lang = "Khmer" if detected_lang in ("km", "multi") else "English"
                    except Exception:
                        response_lang = "English"
                    return self._get_no_info_response(question_id, response_lang, simplified_output)

            # Validate that the retrieval query is still meaningful
            if not retrieval_query or not retrieval_query.strip() or len(retrieval_query.strip()) < 2:
                logger.warning(f"Query is empty/invalid after normalization. Original: {question}, Retrieval: {retrieval_query}")
                self.token_logger.finish_query_session(
                    self.token_logger.start_query_session(query_id=question_id, question=question)
                )
                # Detect language of original query for appropriate response
                try:
                    detected_lang = self.document_loader.detect_languages(question)
                    response_lang = "Khmer" if detected_lang in ('km', 'multi') else "English"
                except:
                    response_lang = "English"
                return self._get_no_info_response(question_id, response_lang, simplified_output)

            k = max(5, RETRIEVAL_K)

            try:
                detected_language = self.document_loader.detect_languages(retrieval_query)
                if detected_language == 'km':
                    language_name = "Khmer"
                    query_language = 'km'
                elif detected_language == 'multi':
                    language_name = "Khmer"
                    query_language = None
                else:
                    language_name = "English"
                    query_language = 'en'
                logger.info(f"Detected language: {language_name} ({detected_language}), normalized query_language: {query_language} for question: {question}")
            except Exception as e:
                logger.error(f"Error detecting language: {e}")
                detected_language = 'en'
                language_name = "English"
                query_language = 'en'

            try:
                logger.info(f"Detected_language: {language_name}, {detected_language}")
                question_embedded = None
                if self.vector_service and getattr(self.vector_service, 'embedding', None):
                    try:
                        question_embedded = self.vector_service.embedding.embed_query(retrieval_query)
                    except Exception as e:
                        logger.warning(f"Failed to embed query: {e}")
                        question_embedded = None
                else:
                    logger.warning("Embedding service not available - skipping embedding/reuse_history.")

                # Memory cleanup after embedding
                if MemoryManager.should_cleanup():
                    MemoryManager.force_cleanup()
                    logger.info(f"Memory cleanup performed after embedding. Status: {MemoryManager.get_memory_status()}")

                # Enable history reuse to restore PDF references on page refresh
                check_history = self.reuse_history(retrieval_query, question_embedded, language_name, session_id)
                if check_history:
                    logger.info(
                        f"History reused successfully with {len(check_history.get('document_references', []))} document references"
                    )
                    # Return complete history with document_references preserved
                    return check_history

                logger.info("Continue processing...")
                retrieval_collection = None if doc_id else file_type
                use_agentic_retrieval = bool(self.agentic_rag and not doc_id)
                if use_agentic_retrieval:
                    logger.info("[AgenticRAG] Using agentic multi-step retrieval")
                    related_docs = self.agentic_rag.get_docs(
                        question=retrieval_query,
                        collection_name=retrieval_collection,
                        query_language=query_language,
                        doc_id=doc_id,
                        language=detected_language,
                    )
                else:
                    if doc_id:
                        logger.info(
                            f"Using deterministic doc-scoped retrieval for doc_id={doc_id}"
                        )
                    related_docs = self.get_related_docs(
                        retrieval_query,
                        retrieval_collection,
                        query_language=query_language,
                        k=k,
                        doc_id=doc_id,
                    )
                logger.info(f"Related documents found: {len(related_docs) if related_docs else 0}")


                # Check memory after document retrieval
                memory_status = MemoryManager.get_memory_status()
                logger.info(f"Memory status after document retrieval: {memory_status}")
                
                if memory_status['needs_cleanup']:
                    logger.info("Performing memory cleanup after document retrieval")
                    MemoryManager.force_cleanup()
                    MemoryManager.force_gpu_cleanup()

                if not related_docs:
                    logger.warning("No related documents found")
                    self.token_logger.finish_query_session(query_session_id)
                    return self._get_no_info_response(question_id, language_name, simplified_output)

                # Apply language-dependent score threshold to Qdrant results.
                # NOTE: Skip this filter when AgenticRAG is active it already
                # applies its own confidence gate using RRF scores internally.
                # The rerank (_rerank_score) values returned by AgenticRAG are
                # cross-encoder scores (typically 0.05) and must NOT be
                # compared against similarity-based thresholds (0.50-0.65).
                if use_agentic_retrieval or doc_id:
                    reason = "confidence gate already applied internally" if use_agentic_retrieval else f"doc-scoped retrieval active for doc_id={doc_id}"
                    logger.info(
                        f"Skipping legacy score threshold ({reason})"
                    )
                    filtered_related_docs = related_docs
                else:
                    try:
                        if detected_language == 'km':
                            score_threshold = 0.57
                        else:
                            score_threshold = 0.57
                        logger.info(f"Applying score threshold: {score_threshold} for language {detected_language}")

                        # Filter documents whose max_score meets or exceeds the threshold
                        filtered_related_docs = {doc_id: doc for doc_id, doc in related_docs.items() if float(doc.get('max_score', 0.0)) >= score_threshold} if isinstance(related_docs, dict) else related_docs
                    except Exception as e:
                        logger.error(f"Error applying score threshold: {e}")
                        filtered_related_docs = related_docs

                # Use filtered documents going forward
                related_docs = filtered_related_docs
                relevant_doc_ids = list(related_docs.keys())
                
                # Filter out inactive documents
                from document.models import Document as DocModel
                # Convert UUIDs to strings for consistent comparison
                active_doc_ids = set(
                    str(doc_id) for doc_id in DocModel.objects.filter(
                        id__in=relevant_doc_ids,
                        is_active=True
                    ).values_list('id', flat=True)
                )
                
                # Remove inactive documents from related_docs
                inactive_count = len(relevant_doc_ids) - len(active_doc_ids)
                if inactive_count > 0:
                    inactive_doc_ids = set(relevant_doc_ids) - active_doc_ids
                    logger.warning(f"Filtering out {inactive_count} inactive/deleted documents from search results")
                    logger.warning(f"Inactive document IDs: {list(inactive_doc_ids)}")
                    
                    # Log detailed status of each document
                    for doc_id in relevant_doc_ids:
                        try:
                            doc = DocModel.objects.get(id=doc_id)
                            logger.warning(f"Document {doc_id}: is_active={doc.is_active}, status={getattr(doc, 'status', 'N/A')}, name={getattr(doc, 'name', 'N/A')}")
                        except DocModel.DoesNotExist:
                            logger.warning(f"Document {doc_id}: NOT FOUND in database")
                    
                    related_docs = {doc_id: doc_data for doc_id, doc_data in related_docs.items() if doc_id in active_doc_ids}
                    relevant_doc_ids = list(related_docs.keys())
                
                if not relevant_doc_ids:
                    logger.warning("No active documents found in search results.")
                    self.token_logger.finish_query_session(query_session_id)
                    return self._get_no_info_response(question_id, language_name, simplified_output)

                # Collect all chunks organized by document with their references
                all_sections = []  # Store each doc's chunks + reference together

                # STEP-BY-STEP RANKING: Sort documents by relevance score (max_score)
                # This ensures top PDFs are shown first, followed by other related items
                sorted_doc_ids = sorted(
                    relevant_doc_ids,
                    key=lambda doc_id: float(related_docs[doc_id].get('max_score', 0.0)),
                    reverse=True
                )
                
                # Show all documents ranked by relevance (top 3 first, then others)
                logger.info(f"Processing {len(sorted_doc_ids)} documents in ranked order: {[(d, related_docs[d].get('max_score')) for d in sorted_doc_ids]}")

                for doc_id in sorted_doc_ids:
                    try:
                        file_name = related_docs[doc_id].get("file_name") or "Untitled Document"
                        if file_name == "Document":
                            file_name = "PDF Document"

                        # Process each chunk individually
                        chunks_for_doc = related_docs[doc_id]["chunks"]
                        
                        for chunk_idx, chunk in enumerate(chunks_for_doc):
                            # Use page_content for full content (fallback to content if not available)
                            chunk_content = chunk.get("page_content") or chunk.get("content", "")
                            
                            logger.debug(
                                f"chunk_idx={chunk_idx}, content length={len(chunk_content)}, "
                                f"has page_content={'page_content' in chunk}"
                            )

                            # Skip empty chunks
                            if not chunk_content or len(chunk_content.strip()) < 20:
                                logger.info(f"Chunk {chunk_idx} in document {doc_id} has insufficient content, skipping")
                                continue

                            # Each chunk gets its own reference - store as structured data
                            page_num = chunk.get("page", 1)
                            page_start = int(chunk.get("page_start", page_num) or page_num)
                            page_end = int(chunk.get("page_end", page_num) or page_num)
                            if page_end < page_start:
                                page_end = page_start
                            pages_range = list(range(page_start, page_end + 1)) if page_end > page_start else [page_start]
                            chunk_count = len(chunks_for_doc)
                            max_score = related_docs[doc_id].get("max_score", 0.0)
                            
                            # Format chunk as HTML table row data for storage
                            # Table will be assembled in the answers array
                            chunk_score = float(chunk.get("score", max_score))
                            table_row_data = {
                                "content": chunk_content,
                                "document": file_name,
                                "page": page_num,
                                "pages": pages_range,
                                "score": f"{chunk_score:.2f}",
                                "doc_id": doc_id
                            }
                            
                            # Store chunk with structured reference data for frontend
                            chunk_header = chunk.get("header", "")
                            chunk_data = {
                                "text": chunk_content,  # Plain text for table
                                "table_row": table_row_data,  # Table row data
                                "header": chunk_header,  # Section header for LLM processing
                                "html": chunk.get("html", ""),
                                "inline_images": chunk.get("inline_images", []),
                                "inline_image_urls": chunk.get("inline_image_urls", []),
                                "inline_image_url": chunk.get("inline_image_url", ""),
                                "inline_image_count": chunk.get("inline_image_count", 0),
                                "page_image_url": chunk.get("page_image_url", ""),
                                "image_directory_url": chunk.get("image_directory_url", ""),
                                "image_manifest_url": chunk.get("image_manifest_url", ""),
                                "header_level": chunk.get("header_level", 1),
                                "chunk_index": chunk.get("chunk_index", 0),
                                "reference": {
                                    "file_name": file_name,
                                    "doc_id": doc_id,
                                    "file_url": f"/v/{doc_id}",
                                    "page": page_num,
                                    "pages": pages_range,
                                    "page_start": page_start,
                                    "page_end": page_end,
                                    "page_confidence": chunk.get("page_confidence", ""),
                                    "page_method": chunk.get("page_method", ""),
                                    "chunks_count": chunk_count,
                                    "max_score": f"{max_score:.2f}",
                                    "score": chunk_score,
                                    "has_table": chunk.get("has_table", False),
                                    "html": chunk.get("html", ""),
                                    "inline_images": chunk.get("inline_images", []),
                                    "inline_image_urls": chunk.get("inline_image_urls", []),
                                    "inline_image_url": chunk.get("inline_image_url", ""),
                                    "inline_image_count": chunk.get("inline_image_count", 0),
                                    "page_image_url": chunk.get("page_image_url", ""),
                                    "image_directory_url": chunk.get("image_directory_url", ""),
                                    "image_manifest_url": chunk.get("image_manifest_url", ""),
                                    "image_directory_path": chunk.get("image_directory_path", ""),
                                    "image_manifest_path": chunk.get("image_manifest_path", ""),
                                },
                                "score": chunk_score,
                                "has_table": chunk.get("has_table", False)
                            }
                            
                            all_sections.append(chunk_data)
                            logger.debug(f"Added chunk {chunk_idx} from doc_id: {doc_id}, page: {page_num}")

                    except Exception as e:
                        logger.error(f"Error processing document {doc_id}: {e}")
                        continue

                # Check if we have any sections before creating response
                if not all_sections:
                    logger.warning("No valid chunk content generated")
                    self.token_logger.finish_query_session(query_session_id)
                    return self._get_no_info_response(question_id, language_name, simplified_output)

                # Sort by chunk-level relevance score (descending) and keep top 5
                all_sections_sorted = sorted(all_sections, key=lambda x: float(x.get("score", 0)), reverse=True)
                all_sections_top5 = all_sections_sorted[:5]  # Keep top 5 chunks

                if not all_sections_top5:
                    logger.warning("No chunks found after sorting")
                    self.token_logger.finish_query_session(query_session_id)
                    return self._get_no_info_response(question_id, language_name, simplified_output)

                # Group chunks by doc_id to get all content from each document
                doc_chunks_map = {}
                for sec in all_sections_top5:
                    doc_id = sec.get("reference", {}).get("doc_id")
                    if doc_id not in doc_chunks_map:
                        doc_chunks_map[doc_id] = []
                    doc_chunks_map[doc_id].append(sec)

                # Get top document's chunks (all of them)
                top_doc_id = all_sections_top5[0].get("reference", {}).get("doc_id")
                all_top_chunks = doc_chunks_map.get(top_doc_id, [])
                logger.info(f"Using all {len(all_top_chunks)} chunks from top doc: {top_doc_id}")

                extracted_section = ""
                follow_up_chunks = []
                answer_chunks = []
                selected_chunk_text = ""
                selected_chunk_header = ""
                selected_reference = all_sections_top5[0].get("reference", {})
                header_candidates = [
                    (sec.get("header") or "").strip() for sec in all_sections_top5
                ]
                query_focus = self._determine_query_focus(llm_question, header_candidates)

                if self.agentic_rag:
                    # Agentic synthesis: Use top chunks from ALL documents (multi-hop support)
                    context_chunks = all_sections_top5
                    answer_chunks = context_chunks[:1]
                    follow_up_chunks = all_sections_top5[1:]  # Use other top chunks for follow-ups

                    logger.info(f"[AgenticRAG] Synthesizing answer from {len(context_chunks)} chunks")

                    # Check first chunk header to decide strategy (matches production).
                    top_chunk = context_chunks[0] if context_chunks else {}
                    top_header = (top_chunk.get("header") or "").strip()
                    top_ref = top_chunk.get("reference") if isinstance(top_chunk, dict) else None
                    if top_ref:
                        selected_reference = top_ref
                    top_text = (top_chunk.get("text") or "").strip() if isinstance(top_chunk, dict) else ""
                    if not selected_chunk_text and top_text:
                        selected_chunk_text = top_text
                        selected_chunk_header = top_header

                    if query_focus == "preamble":
                        preamble_chunk = self._select_preamble_chunk(context_chunks, llm_question)
                        answer_chunks = [preamble_chunk] if preamble_chunk else context_chunks[:1]
                        follow_up_chunks = [c for c in context_chunks if c not in answer_chunks][:4]

                        selected_chunk = answer_chunks[0] if answer_chunks else {}
                        if selected_chunk.get("reference"):
                            selected_reference = selected_chunk.get("reference", {})

                        selected_header = (selected_chunk.get("header") or "").strip()
                        selected_text = (selected_chunk.get("text") or "").strip()
                        selected_body = self._strip_chunk_header_from_body(selected_text, selected_header)
                        source_text = selected_body or selected_text

                        if source_text:
                            llm_processed_text = self.llm_content(
                                chunk_data={
                                    "content": source_text,
                                    "page_content": selected_text,
                                    "header": selected_header,
                                },
                                question=question,
                                language_name=language_name,
                            )
                            if llm_processed_text and not self._is_no_answer_text(llm_processed_text):
                                extracted_section = llm_processed_text.strip()
                            else:
                                extracted_section = source_text

                            if not selected_chunk_text:
                                selected_chunk_text = selected_text
                                selected_chunk_header = selected_header
                    elif top_header == "Content":
                        # Filter to Content-only chunks and run the permissive synthesis prompt.
                        content_chunks = [
                            c for c in context_chunks
                            if (c.get("header") or "").strip() == "Content"
                        ]
                        logger.info(
                            f"[AgenticRAG] Filtered {len(context_chunks)} chunks to "
                            f"{len(content_chunks)} 'Content' chunks"
                        )
                        if content_chunks:
                            answer_chunks = content_chunks[:1]

                        raw_answer = self._synthesize_answer(
                            question,
                            content_chunks,
                            language_name,
                            is_content_header=True,
                        )
                        extracted_section = (raw_answer or "").strip()
                    else:
                        logger.info(
                            f"[AgenticRAG] Header '{top_header}' != 'Content' — using verbatim formatting (old logic)"
                        )
                        combined_parts = []
                        for chunk in context_chunks[:1]:
                            h = (chunk.get("header") or "").strip()
                            raw_text = (chunk.get("text") or "").strip()
                            body = self._strip_chunk_header_from_body(raw_text, h)
                            if h and h != "Content":
                                combined_parts.append(f"**{h}**\n\n{body.strip()}")
                            else:
                                combined_parts.append(body or raw_text)
                        extracted_section = "\n\n---\n\n".join(
                            p for p in combined_parts if p and p.strip()
                        )
                    
                else:
                    # Legacy: Restrict to single top document
                    # Split: top 1 chunk for answer, remaining 4 for follow-up questions
                    if query_focus == "preamble":
                        preamble_chunk = self._select_preamble_chunk(all_top_chunks, llm_question)
                        answer_chunks = [preamble_chunk] if preamble_chunk else all_top_chunks[:1]
                    else:
                        answer_chunks = all_top_chunks[:1]  # default: 1 chunk for answer

                    # If user asks with a pasted header/title (short header-like query),
                    # and top chunk is header-only, include extra chunks from same doc.
                    normalized_q = normalize_user_query(question).strip(" -:|")
                    q_words = [w for w in re.split(r'\s+', normalized_q) if w]
                    header_like_query = False
                    if normalized_q and len(q_words) <= 7:
                        alpha_words = [w for w in q_words if re.search(r'[A-Za-z]', w)]
                        title_like_words = [w for w in alpha_words if w[:1].isupper()]
                        header_like_query = bool(
                            alpha_words and len(title_like_words) >= max(2, len(alpha_words) - 1)
                        )

                    if header_like_query and answer_chunks:
                        top_chunk = answer_chunks[0]
                        top_header = (top_chunk.get("header") or "").strip()
                        top_text = (top_chunk.get("text") or "").strip()

                        top_body = re.sub(r'^#+\s*' + re.escape(top_header) + r'\s*\n*', '', top_text, count=1) if top_header else top_text
                        if top_header and top_body.lstrip().startswith(top_header):
                            top_body = top_body.lstrip()[len(top_header):].lstrip('\n').lstrip()

                        # Header is present but body is too thin -> pull a couple more chunks.
                        if top_header and len(top_body.strip()) < 40:
                            extra_chunks = []
                            for c in all_top_chunks[1:]:
                                c_text = (c.get("text") or "").strip()
                                if len(c_text) >= 40:
                                    extra_chunks.append(c)
                                if len(extra_chunks) >= 2:
                                    break
                            if extra_chunks:
                                answer_chunks = [top_chunk] + extra_chunks
                                logger.info(
                                    f"Header-like query detected; expanded answer chunks "
                                    f"from 1 to {len(answer_chunks)}"
                                )

                    follow_up_chunks = [c for c in all_top_chunks if c not in answer_chunks][:4]

                    # Build answer from the top chunk (header-driven routing; matches production).
                    chunks_for_render = answer_chunks

                    combined_content_parts = []
                    for chunk in chunks_for_render:
                        if chunk.get("reference"):
                            selected_reference = chunk.get("reference", {})

                        chunk_text = (chunk.get("text") or "").strip()
                        chunk_header = (chunk.get("header") or "").strip()
                        if not selected_chunk_text and chunk_text:
                            selected_chunk_text = chunk_text
                            selected_chunk_header = chunk_header
                        logger.info(f"response: processing chunk header='{chunk_header}'")
                        chunk_body = self._strip_chunk_header_from_body(chunk_text, chunk_header)

                        if (chunk_header == "Content" or query_focus == "preamble") and question:
                            source_text = chunk_body or chunk_text
                            logger.info(f"response: invoking llm_content for question='{question[:100]}'")
                            llm_processed_text = self.llm_content(
                                chunk_data={
                                    "content": source_text,
                                    "page_content": chunk_text,
                                    "header": chunk_header
                                },
                                question=question,
                                language_name=language_name
                            )
                            if llm_processed_text and not self._is_no_answer_text(llm_processed_text):
                                chunk_text = llm_processed_text.strip()
                                logger.info(f"response: llm_content returned {len(chunk_text)} chars")
                            else:
                                logger.info("response: llm_content found no grounded answer; skipping chunk text fallback")
                                chunk_text = ""

                        if chunk_text:
                            if chunk_header and chunk_header != "Content":
                                combined_content_parts.append(
                                    f"**{chunk_header}**\n\n{chunk_body}"
                                )
                            else:
                                combined_content_parts.append(chunk_text)

                    # Join content
                    extracted_section = "\n\n---\n\n".join(combined_content_parts)
                    logger.info(f"Combined {len(combined_content_parts)} chunks into answer, {len(follow_up_chunks)} chunks for follow-ups")

                # Create answer with combined content
                answers = []
                ref = selected_reference or all_sections_top5[0]["reference"]
                doc_id = ref["doc_id"]
                page = ref.get("page", 1)
                answer_pages = ref.get("pages") if isinstance(ref.get("pages"), list) and ref.get("pages") else [page]
                score = ref.get("score") or ref.get("max_score", 0.5)

                if extracted_section and extracted_section.strip():
                    # Keep answer text as markdown/source structure so frontend markdown
                    # renderer preserves bullets, new lines, and table layout like VS Code.
                    display_section = self._dedupe_images_in_display_html(extracted_section)
                    display_section_html = self._dedupe_images_in_display_html(
                        self._to_display_html(display_section)
                    )
                    reference_source = self._strip_chunk_header_from_body(
                        selected_chunk_text,
                        selected_chunk_header,
                    ) if selected_chunk_text else extracted_section
                    reference_html = self._strip_images_from_reference_html(
                        self._to_display_html(reference_source or selected_chunk_text or extracted_section)
                    )
                    answers.append({
                        "question_id": question_id,
                        "text": display_section,
                        "file_name": ref["file_name"],
                        "doc_id": doc_id,
                        "page": page,
                        "pages": answer_pages,
                        "score": score,
                        "id": 0,
                        "is_relevant": 0,
                        "document_reference": {
                            "doc_id": doc_id,
                            "file_name": ref["file_name"],
                            "file_url": f"/v/{doc_id}",
                            "page": page,
                            "pages": answer_pages,
                            "max_score": score,
                            "html": reference_html,
                        },
                        "html": display_section_html
                    })
                    logger.info(f"Created answer with extracted section ({len(extracted_section)} chars)")
                else:
                    logger.warning("No relevant section extracted from top chunk")
                    self.token_logger.finish_query_session(query_session_id)
                    return self._get_no_info_response(question_id, language_name, simplified_output)

                # Track unique document for document_references
                unique_docs = {
                    doc_id: {
                        "doc_id": doc_id,
                        "file_name": ref["file_name"],
                        "pages": set(answer_pages),
                        "scores": [ref.get("score", 0.5)],
                        "html": reference_html,
                        "inline_images": ref.get("inline_images", []),
                        "inline_image_urls": ref.get("inline_image_urls", []),
                        "inline_image_url": ref.get("inline_image_url", ""),
                        "inline_image_path": ref.get("inline_image_path", ""),
                        "inline_image_count": ref.get("inline_image_count", 0),
                        "page_image_url": ref.get("page_image_url", ""),
                        "page_image_path": ref.get("page_image_path", ""),
                        "image_directory_path": ref.get("image_directory_path", ""),
                        "image_directory_url": ref.get("image_directory_url", ""),
                        "image_manifest_path": ref.get("image_manifest_path", ""),
                        "image_manifest_url": ref.get("image_manifest_url", ""),
                    }
                }

                # Candidate pool for follow-up questions.
                if follow_up_chunks:
                    all_sections = follow_up_chunks
                else:
                    all_sections = [
                        s for s in all_sections_top5
                        if s.get("reference", {}).get("doc_id") != top_doc_id
                           or s not in answer_chunks
                    ]
                    logger.info(
                        f"follow_up_chunks empty; using {len(all_sections)} sections "
                        f"from all_sections_top5 as followup pool"
                    )

                logger.info(f"Created 1 answer from extracted section ({len(extracted_section)} chars)")
                
                # Check if we have any answers with relevant information
                if not answers or len(answers) == 0:
                    logger.warning("No relevant information found in any chunks after LLM filtering")
                    self.token_logger.finish_query_session(query_session_id)
                    return self._get_no_info_response(question_id, language_name, simplified_output)

                # Build document_references array as clickable PDF cards
                # Only include documents that actually have answers (not filtered out)
                document_references = []
                docs_with_answers = set(answer["doc_id"] for answer in answers)
                
                # Generate follow-up suggestions early (BEFORE exception handler path) to ensure they're always available
                # This fixes the issue where suggestions would be missing when llm_content processing errors occur
                suggestions_early = []
                if FOLLOWUP_SUGGESTIONS_ENABLED and FOLLOWUP_SUGGESTIONS_COUNT > 0:
                    logger.info(f"[DEBUG SUGGESTIONS] ENABLED={FOLLOWUP_SUGGESTIONS_ENABLED}, COUNT={FOLLOWUP_SUGGESTIONS_COUNT}")
                    logger.info(f"[DEBUG SUGGESTIONS] all_sections length: {len(all_sections) if isinstance(all_sections, list) else 'NOT A LIST'}")
                    logger.info(f"[DEBUG SUGGESTIONS] all_sections_top5 length: {len(all_sections_top5) if isinstance(all_sections_top5, list) else 'NOT A LIST'}")
                    logger.info(f"Preparing follow-up suggestions pool from {len(all_sections)} sections")
                    chunk_candidates_early = []
                    try:
                        top_chunks = all_sections if isinstance(all_sections, list) else []
                        logger.info(f"[DEBUG SUGGESTIONS] top_chunks from all_sections: {len(top_chunks)}")
                        if not top_chunks and isinstance(all_sections_top5, list):
                            top_chunks = all_sections_top5
                            logger.info(f"[DEBUG SUGGESTIONS] Fallback to all_sections_top5, using {len(top_chunks)} chunks")
                        for chunk in top_chunks[:6]:  # Take up to 6 chunks for candidates
                            try:
                                ref = chunk.get('reference') if isinstance(chunk, dict) else None
                                chunk_candidates_early.append({
                                    'header': chunk.get('header', ''),
                                    'file_name': ref.get('file_name') if ref else '',
                                    'page': ref.get('page') if ref else None,
                                    'pages': [ref.get('page')] if ref and ref.get('page') else [],
                                    'text': chunk.get('text') if isinstance(chunk, dict) else str(chunk)
                                })
                            except Exception as e:
                                logger.warning(f"Error processing chunk for early followup candidates: {e}")
                    except Exception as e:
                        logger.warning(f"Error building early chunk_candidates for followups: {e}")
                        chunk_candidates_early = []

                    if chunk_candidates_early:
                        logger.info(f"[DEBUG SUGGESTIONS] chunk_candidates_early has {len(chunk_candidates_early)} items, calling LLM")
                        try:
                            suggestions_early = self._generate_followup_questions(
                                question,
                                chunk_candidates_early[:4],
                                FOLLOWUP_SUGGESTIONS_COUNT,
                                detected_language,
                            )
                            logger.info(f"Early generated {len(suggestions_early)} follow-up suggestions")
                            logger.info(f"[DEBUG EARLY] suggestions_early content: {suggestions_early}")
                        except Exception as e:
                            logger.warning(f"Early follow-up generation failed: {e}, will retry later")
                            suggestions_early = []
                    else:
                        logger.warning(f"[DEBUG SUGGESTIONS] chunk_candidates_early is EMPTY - no suggestions generated!")
                else:
                    logger.info("Follow-up suggestions disabled for fast mode")

                for doc_id in sorted_doc_ids:  # Use same sort order as chunks
                    if doc_id in unique_docs and doc_id in docs_with_answers:
                        doc_info = unique_docs[doc_id]
                        pages_list = sorted(list(doc_info["pages"]))
                        max_score = max(doc_info["scores"]) if doc_info["scores"] else 0.5
                        chunks_count = len([a for a in answers if a["doc_id"] == doc_id])

                        if pages_list:
                            page_display_str = f"page {', '.join(map(str, pages_list))}"
                        else:
                            page_display_str = "N/A"
                        pdf_card_html = f'''<div class="flex flex-col p-4 rounded-xl border border-gray-200 bg-white hover:bg-primary/5 hover:border-primary/30 transition-all duration-200 cursor-pointer group/doc" onclick="window.location='/v/{doc_id}'">
<div class="flex items-start gap-3 w-full">
<div class="p-2 bg-primary/10 rounded-lg flex-shrink-0 group-hover/doc:bg-primary/20 transition-all">
<img alt="PDF" loading="lazy" width="24" height="24" decoding="async" data-nimg="1" class="w-5 h-5" src="/PDFicon.svg" style="color: transparent;">
</div>
<div class="flex-1 min-w-0">
<p class="text-sm font-semibold text-gray-900 truncate group-hover/doc:text-primary transition-colors">{doc_info["file_name"]}</p>
<p class="text-xs text-gray-500 mt-1"><span class="font-semibold text-primary">{page_display_str} • </span>{chunks_count} chunks • Score: {max_score:.2f}</p>
</div>
<div class="text-primary opacity-0 group-hover/doc:opacity-100 transition-opacity text-lg">?</div>
</div>
</div>'''

                        document_references.append({
                            "doc_id": doc_id,
                            "file_name": doc_info["file_name"],
                            "file_url": f"/v/{doc_id}",
                            "chunks_count": chunks_count,
                            "max_score": max_score,
                            "pages": pages_list,
                            "page_display": page_display_str,
                            "html": doc_info.get("html", ""),
                            "inline_images": doc_info.get("inline_images", []),
                            "inline_image_urls": doc_info.get("inline_image_urls", []),
                            "inline_image_url": doc_info.get("inline_image_url", ""),
                            "inline_image_path": doc_info.get("inline_image_path", ""),
                            "inline_image_count": doc_info.get("inline_image_count", 0),
                            "page_image_url": doc_info.get("page_image_url", ""),
                            "page_image_path": doc_info.get("page_image_path", ""),
                            "image_directory_path": doc_info.get("image_directory_path", ""),
                            "image_directory_url": doc_info.get("image_directory_url", ""),
                            "image_manifest_path": doc_info.get("image_manifest_path", ""),
                            "image_manifest_url": doc_info.get("image_manifest_url", ""),
                            "html_card": pdf_card_html
                        })

                logger.info(f"Generated {len(answers)} answers from {len(document_references)} documents (filtered from {len(sorted_doc_ids)} total retrieved)")
                logger.info(f"Documents with relevant answers: {[ref['file_name'] for ref in document_references]}")

                answers, document_references, display_score_threshold = self._filter_user_visible_results(
                    answers=answers,
                    document_references=document_references,
                    detected_language=detected_language,
                    use_agentic_retrieval=use_agentic_retrieval,
                )
                if not answers:
                    logger.warning(
                        f"No answers remain after applying user-visible score threshold "
                        f"{display_score_threshold:.2f} for language={detected_language}"
                    )
                    self.token_logger.finish_query_session(query_session_id)
                    return self._get_no_info_response(question_id, language_name, simplified_output)

                # Check memory after document processing
                memory_status = MemoryManager.get_memory_status()
                logger.info(f"Memory status after document processing: {memory_status}")
                
                if memory_status['needs_cleanup']:
                    logger.info("Performing final memory cleanup after document processing")
                    MemoryManager.force_cleanup()
                    MemoryManager.force_gpu_cleanup()

                completed_session = self.token_logger.finish_query_session(query_session_id)

                if answers:
                    # Skip LLM processing - chunks already have inline references
                    # Save to history with session_id if provided
                    # Remove html_card from document_references before saving (keep only data fields)
                    clean_document_references = [
                        {k: v for k, v in ref.items() if k != "html_card"}
                        for ref in document_references
                    ]
                    
                    # Preserve per-answer references as top-level to ensure refresh matches streaming display
                    per_answer_references = []
                    for ans in answers:
                        dr = None
                        if isinstance(ans, dict):
                            if 'document_reference' in ans and isinstance(ans.get('document_reference'), dict):
                                dr = ans.get('document_reference')
                            elif 'document_references' in ans and isinstance(ans.get('document_references'), list) and ans.get('document_references'):
                                first_dr = ans.get('document_references')[0]
                                if isinstance(first_dr, dict):
                                    dr = first_dr
                        if dr:
                            per_answer_references.append({k: v for k, v in dr.items() if k != "html_card"})
                        else:
                            # Fallback: construct minimal ref from common fields
                            doc_id = ans.get('doc_id') if isinstance(ans, dict) else None
                            file_name = ans.get('file_name') if isinstance(ans, dict) else None
                            page = ans.get('page') if isinstance(ans, dict) else None
                            score = ans.get('score') or ans.get('max_score', 0.5)
                            if doc_id or file_name:
                                per_answer_references.append({
                                    'doc_id': doc_id,
                                    'file_name': file_name or '',
                                    'file_url': ans.get('file_url') if isinstance(ans, dict) and ans.get('file_url') else (f"/v/{doc_id}" if doc_id else None),
                                    'page': page,
                                    'max_score': score,
                                    'html': ans.get('html', '') if isinstance(ans, dict) else '',
                                    'inline_images': ans.get('inline_images', []) if isinstance(ans, dict) else [],
                                    'inline_image_urls': ans.get('inline_image_urls', []) if isinstance(ans, dict) else [],
                                    'inline_image_url': ans.get('inline_image_url', '') if isinstance(ans, dict) else '',
                                    'inline_image_path': ans.get('inline_image_path', '') if isinstance(ans, dict) else '',
                                    'inline_image_count': ans.get('inline_image_count', 0) if isinstance(ans, dict) else 0,
                                    'page_image_url': ans.get('page_image_url', '') if isinstance(ans, dict) else '',
                                    'page_image_path': ans.get('page_image_path', '') if isinstance(ans, dict) else '',
                                    'image_directory_path': ans.get('image_directory_path', '') if isinstance(ans, dict) else '',
                                    'image_directory_url': ans.get('image_directory_url', '') if isinstance(ans, dict) else '',
                                    'image_manifest_path': ans.get('image_manifest_path', '') if isinstance(ans, dict) else '',
                                    'image_manifest_url': ans.get('image_manifest_url', '') if isinstance(ans, dict) else '',
                                })

                    # Prepare history and response selection (limit answers & generate followups)
                    try:
                        # Logic: Always show 1 answer (top chunk), optionally generate follow-up questions
                        top_n = 1  # Always show 1 answer
                        suggestions_count = (
                            FOLLOWUP_SUGGESTIONS_COUNT if FOLLOWUP_SUGGESTIONS_ENABLED else 0
                        )

                        selected_answers = answers[:top_n]
                        selected_per_answer_references = per_answer_references[:top_n] if per_answer_references else []

                        # Aggregate only the selected documents for the top answers
                        sel_doc_ids = set(a.get('doc_id') for a in selected_answers if isinstance(a, dict) and a.get('doc_id'))
                        selected_clean_document_references = [ref for ref in clean_document_references if ref.get('doc_id') in sel_doc_ids] if clean_document_references else []

                        # Build chunk-level follow-up candidates from the top relevant chunks (prefer chunks NOT in selected answers)
                        suggestions = []
                        if suggestions_count > 0:
                            logger.info(f"[DEBUG MAIN] Building chunk candidates for main path, suggestions_count={suggestions_count}")
                            chunk_candidates = []
                            try:
                                top_chunks = all_sections if isinstance(all_sections, list) else []
                                logger.info(f"[DEBUG MAIN] top_chunks from all_sections: {len(top_chunks)}")
                                # If still empty, pull from the full all_sections_top5 pool
                                if not top_chunks and isinstance(all_sections_top5, list):
                                    top_chunks = all_sections_top5
                                    logger.info(f"[DEBUG MAIN] Fallback to all_sections_top5: {len(top_chunks)} chunks")
                                logger.info(f"[DEBUG MAIN] Processing {len(top_chunks)} chunks, prefer outside sel_doc_ids={sel_doc_ids}")
                                for chunk in top_chunks:
                                    try:
                                        ref = chunk.get('reference') if isinstance(chunk, dict) else None
                                        doc_id = ref.get('doc_id') if ref else None
                                        # prefer chunks from other docs first
                                        if doc_id and doc_id in sel_doc_ids:
                                            continue
                                        chunk_candidates.append({
                                            'header': chunk.get('header', ''),
                                            'file_name': ref.get('file_name') if ref else '',
                                            'page': ref.get('page') if ref else None,
                                            'pages': [ref.get('page')] if ref and ref.get('page') else [],
                                            'text': chunk.get('text') if isinstance(chunk, dict) else str(chunk)
                                        })
                                    except Exception as e:
                                        logger.warning(f"Error processing chunk for followup candidates: {e}")
                                # If not enough candidates, include chunks from selected docs to fill
                                if len(chunk_candidates) < suggestions_count:
                                    for chunk in top_chunks:
                                        try:
                                            ref = chunk.get('reference') if isinstance(chunk, dict) else None
                                            candidate = {
                                                'header': chunk.get('header', ''),
                                                'file_name': ref.get('file_name') if ref else '',
                                                'page': ref.get('page') if ref else None,
                                                'pages': [ref.get('page')] if ref and ref.get('page') else [],
                                                'text': chunk.get('text') if isinstance(chunk, dict) else str(chunk)
                                            }
                                            if candidate not in chunk_candidates:
                                                chunk_candidates.append(candidate)
                                        except Exception:
                                            pass
                                        if len(chunk_candidates) >= suggestions_count:
                                            break
                            except Exception as e:
                                logger.warning(f"Error building chunk_candidates for followups: {e}")
                                chunk_candidates = []
                            
                            logger.info(f"[DEBUG MAIN] Final chunk_candidates: {len(chunk_candidates)} items")
                            if not chunk_candidates:
                                logger.warning(f"[DEBUG MAIN] chunk_candidates is EMPTY - no main path suggestions!")

                            # Generate follow-up suggestions using LLM
                            try:
                                suggestions = self._generate_followup_questions(
                                    question,
                                    chunk_candidates[:4],
                                    suggestions_count,
                                    detected_language,
                                )
                                logger.info(f"Generated {len(suggestions)} follow-up suggestions (main path)")
                                logger.info(f"[DEBUG MAIN] suggestions content: {suggestions}")
                            except Exception as e:
                                logger.info(f"Follow-up generation failed (main path): {e}")
                                # Fallback to early-generated suggestions if available
                                if suggestions_early:
                                    suggestions = suggestions_early
                                    logger.info(
                                        f"Using early-generated suggestions ({len(suggestions_early)} items) as fallback"
                                    )
                                else:
                                    suggestions = []
                                logger.info(f"[DEBUG MAIN FALLBACK] suggestions after fallback: {suggestions}")

                            # If main generation returned empty, try early-generated suggestions
                            if not suggestions and suggestions_early:
                                suggestions = suggestions_early
                                logger.info(
                                    f"Using early-generated suggestions ({len(suggestions_early)} items) "
                                    f"as main path returned empty"
                                )

                        # Build history payload using selected answers and suggestions
                        history_data = {
                            "answers": selected_answers,
                            "document_references": selected_per_answer_references,
                            "document_references_aggregated": selected_clean_document_references,
                            "suggestions": suggestions
                        }
                        if session_id:
                            history_data["session_id"] = str(session_id)

                        # Save history including suggestions
                        try:
                            history_data["language"] = language_name
                            success = self.safe_historical_handling(question, json.dumps(history_data), question_id, session_id)
                            if not success:
                                logger.warning("Failed to save to history")
                        except Exception as e:
                            logger.error(f"Error saving history: {e}")

                        # Build combined_answer with proper formatting (answers joined with separators)
                        combined_answer_parts = []
                        for i, ans in enumerate(selected_answers, start=1):
                            ans_text = ans.get("text", "") if isinstance(ans, dict) else str(ans)
                            if ans_text:
                                # Only normalize legacy markdown noise for non-HTML answers.
                                if not self._looks_like_html(ans_text):
                                    ans_text = cleanup_bold_colon(ans_text)
                                combined_answer_parts.append(ans_text)
                                # Debug: Check newlines in answer text
                                logger.debug(
                                    f"Answer {i} has {ans_text.count(chr(10))} newlines, length={len(ans_text)}"
                                )
                                if '|' in ans_text:
                                    logger.debug(
                                        f"Answer {i} contains table pipes. First 300 chars: {repr(ans_text[:300])}"
                                    )
                        combined_answer = "\n\n---\n\n".join(combined_answer_parts) if combined_answer_parts else ""
                        
                        # Debug: Check final combined_answer
                        logger.debug(f"combined_answer has {combined_answer.count(chr(10))} newlines total")

                        # Prepare final response payload
                        if simplified_output:
                            return {
                                "answers": [ans.get("text", "") for ans in selected_answers],
                                "combined_answer": combined_answer,
                                "language": language_name,
                                "suggestions": suggestions
                            }

                        logger.info(f"[DEBUG RETURN] Returning response with {len(suggestions) if isinstance(suggestions, list) else 0} suggestions")
                        return {
                            "answers": selected_answers,
                            "combined_answer": combined_answer,
                            "document_references": selected_clean_document_references,
                            "language": language_name,
                            "suggestions": suggestions
                        }
                    except Exception as e:
                        logger.error(f"Error preparing response selection: {e}")
                        logger.info(f"[DEBUG] suggestions at error: {suggestions if 'suggestions' in locals() else 'NOT DEFINED'}")
                        # Fallback to original full payload in case of error
                        try:
                            fallback_suggestions = (
                                suggestions
                                if 'suggestions' in locals() and isinstance(suggestions, list)
                                else (suggestions_early if suggestions_early else [])
                            )
                            history_data = {
                                "answers": answers,
                                "document_references": per_answer_references,
                                "document_references_aggregated": clean_document_references,
                                "suggestions": fallback_suggestions,
                            }
                            if session_id:
                                history_data["session_id"] = str(session_id)
                            history_data["language"] = language_name
                            self.safe_historical_handling(question, json.dumps(history_data), question_id, session_id)
                        except Exception:
                            pass
                        # Build combined_answer for fallback
                        fallback_combined_parts = []
                        for i, ans in enumerate(answers, start=1):
                            ans_text = ans.get("text", "") if isinstance(ans, dict) else str(ans)
                            if ans_text:
                                if not self._looks_like_html(ans_text):
                                    ans_text = cleanup_bold_colon(ans_text)
                                fallback_combined_parts.append(ans_text)
                        fallback_combined = "\n\n---\n\n".join(fallback_combined_parts) if fallback_combined_parts else ""
                        # Return the original full response as last resort
                        # NOTE: suggestions may not have been generated yet in this error path,
                        # so fallback to early-generated suggestions or empty list
                        fallback_suggestions = suggestions if 'suggestions' in locals() and isinstance(suggestions, list) else (suggestions_early if suggestions_early else [])
                        logger.info(f"Error fallback: using {len(fallback_suggestions)} suggestions (early={len(suggestions_early)}, main={len(suggestions) if 'suggestions' in locals() else 0})")
                        return {
                            "answers": answers,
                            "combined_answer": fallback_combined,
                            "document_references": clean_document_references,
                            "language": language_name,
                            "suggestions": fallback_suggestions
                        }
                else:
                    return self._get_no_info_response(question_id, language_name, simplified_output)
            except Exception as e:
                logger.error(f"Error in response method: {e}")
                self.token_logger.finish_query_session(query_session_id)
                return self._get_no_info_response(question_id, language_name, simplified_output)

        except Exception as e:
            logger.error(f"Error in response method: {e}")
            self.token_logger.finish_query_session(query_session_id)
            return self._get_no_info_response(question_id, language_name, simplified_output)

    def delete_history(self, file_id:str, session_id: str = None):
        """Permanentlly delete history question related to a specific file"""

        try:
            # Delete from active history
            scroll_filter = None
            if session_id:
                scroll_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="session_id",
                            match=models.MatchValue(value=session_id)
                        )
                    ]
                )

            scroll_result = self.client.scroll(
                collection_name=self.history_collection_name,
                scroll_filter=scroll_filter,
                limit=10000
            )
            points = scroll_result[0]
            ids_to_delete = []

            for point in points:
                payload = point.payload
                related_doc_ids = payload.get('related_doc', '').split(',')
                if file_id in related_doc_ids:
                    ids_to_delete.append(point.id)

            if ids_to_delete:
                self.client.delete(
                    collection_name=self.history_collection_name,
                    points_selector=models.PointIdsList(points=ids_to_delete)
                )
                logger.info(f"Deleted {len(ids_to_delete)} history entries from active collection")

            # Delete from inactive history
            scroll_result_inactive = self.client.scroll(
                collection_name=self.inactive_history_collection_name,
                scroll_filter=scroll_filter,
                limit=10000
            )
            points_inactive = scroll_result_inactive[0]
            inactive_ids_to_delete = []

            for point in points_inactive:
                payload = point.payload
                related_doc_ids = payload.get('related_doc', '').split(',')
                if file_id in related_doc_ids:
                    inactive_ids_to_delete.append(point.id)

            if inactive_ids_to_delete:
                self.client.delete(
                    collection_name=self.inactive_history_collection_name,
                    points_selector=models.PointIdsList(points=inactive_ids_to_delete)
                )
                logger.info(f"Deleted {len(inactive_ids_to_delete)} history entries from inactive collection")
            return True

        except Exception as e :
            logger.error(f"Error delete history for file_id  : {file_id}")
            return False
