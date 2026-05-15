from langchain_core.documents import Document as LangChainDocument
from langchain.embeddings.base import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, SparseVector
import os
import torch
import numpy as np
import threading
from typing import List, Dict, Tuple, Optional
from django.conf import settings
import logging
import traceback
import uuid
import hashlib
import time
import markdown as py_markdown
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Require BGEM3FlagModel for hybrid embeddings
try:
    from FlagEmbedding import BGEM3FlagModel
    BGEM3_AVAILABLE = True
    logger.info("BGEM3FlagModel available - hybrid (dense + sparse) embeddings enabled")
except ImportError:
    BGEM3_AVAILABLE = False
    BGEM3FlagModel = None
    logger.error("BGEM3FlagModel not available; hybrid embeddings are required. Install FlagEmbedding and BGEM3 model files.")

qdrant_host = settings.QDRANT_HOST
qdrant_port = settings.QDRANT_PORT
EMBED_BATCH_SIZE = 64
QDRANT_UPSERT_BATCH_SIZE = 128

def detect_lang_quick(text: str) -> str:
    """Lightweight script heuristic: returns km, en, or multi."""
    if not text:
        return 'en'
    khmer_count = sum(1 for ch in text if '\u1780' <= ch <= '\u17FF')
    latin_count = sum(1 for ch in text if ('a' <= ch.lower() <= 'z') or ch.isdigit() or ch == ' ')
    total = khmer_count + latin_count + 1e-6
    khmer_ratio = khmer_count / total
    latin_ratio = latin_count / total
    if khmer_count > 0 and latin_count > 0 and khmer_ratio >= 0.15 and latin_ratio >= 0.15:
        return 'multi'
    return 'km' if khmer_count > 0 else 'en'


def build_language_filter_values(query_language: str | None) -> List[str] | None:
    if not query_language:
        return None
    if query_language == "multi":
        return None
    return [query_language, "multi"]


def _ensure_chunk_html(html_value: str, fallback_text: str) -> str:
    html = (html_value or "").strip()
    text = (fallback_text or "").strip()

    def render_markdown_html(markdown_text: str) -> str:
        return py_markdown.markdown(
            markdown_text,
            extensions=["tables", "sane_lists", "nl2br"],
            output_format="html5",
        )

    def html_has_meaningful_text(html_text: str) -> bool:
        plain = re.sub(r"<[^>]+>", " ", html_text or "")
        plain = re.sub(r"\s+", " ", plain).strip()
        return bool(plain)

    if html:
        # Legacy payloads sometimes stored only an image fragment (<img>) in html.
        # In that case, rebuild full HTML from markdown/text and append the image fragment.
        if not text:
            return html
        if html_has_meaningful_text(html):
            return html
        try:
            rebuilt = render_markdown_html(text)
            if "<img" in html.lower() and html not in rebuilt:
                rebuilt = f"{rebuilt}\n{html}".strip()
            return rebuilt
        except Exception:
            logger.warning("Failed to rebuild chunk HTML from markdown; using stored html fragment")
            return html

    if not text:
        return ""

    try:
        return render_markdown_html(text)
    except Exception:
        logger.warning("Failed to render fallback chunk HTML from markdown text")
        return f"<p>{text}</p>"


class BGEM3EmbeddingFunction(Embeddings):
    """Wrapper for BGEM3FlagModel or SentenceTransformer to work with Qdrant"""

    def __init__(self, model_path: str):
        try:
            self.use_bgem3 = BGEM3_AVAILABLE
            if not self.use_bgem3 or BGEM3FlagModel is None:
                logger.error("BGEM3FlagModel is not available; hybrid embeddings are required. Install FlagEmbedding and provide BGEM3 model files at the configured path.")
                raise ImportError("BGEM3FlagModel is required for hybrid embeddings; ensure it is installed and available.")

            logger.info(f"Initializing BGEM3FlagModel from path: {model_path}")
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                logger.info("[VectorStore] GPU detected — loading BGE-M3 with FP16 on CUDA.")
                self.model = BGEM3FlagModel(model_path, use_fp16=True)
            else:
                logger.info("[VectorStore] No GPU detected — loading BGE-M3 with FP32 on CPU.")
                self.model = BGEM3FlagModel(model_path, use_fp16=False)

            logger.info(f"BGEM3 model loaded successfully from path: {model_path} (cuda={cuda_available})")

        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents - returns dense vectors only"""
        try:
            if not texts:
                logger.warning("Empty text list provided for embedding")
                return []

            valid_texts = [text for text in texts if text and isinstance(text, str)]
            dropped = len(texts) - len(valid_texts)
            if dropped:
                logger.warning(f"Filtered out {dropped} invalid texts (None/empty/non-str)")
            if not valid_texts:
                logger.error("No valid texts to embed")
                return []

            if self.use_bgem3:
                output = self.model.encode(valid_texts, return_dense=True, return_sparse=False)
                dense_vecs = output.get("dense_vecs", [])
                result = [vec.tolist() if hasattr(vec, 'tolist') else vec for vec in dense_vecs]
            else:
                embeddings = self.model.encode(valid_texts, normalize_embeddings=True)
                result = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

            logger.debug(f"Successfully embedded {len(result)} documents with dense vectors")
            return result
        except Exception as e:
            logger.error(f"Error in embed_documents: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def embed_documents_hybrid(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict]]:
        """Embed a list of documents - returns both dense and sparse vectors (BGEM3 only)"""
        try:
            if not texts:
                return [], []

            valid_texts = [text for text in texts if text and isinstance(text, str)]
            if not valid_texts:
                return [], []

            if self.use_bgem3:
                output = self.model.encode(valid_texts, return_dense=True, return_sparse=True)
                
                dense_vecs = output.get("dense_vecs", [])
                lexical_weights = output.get("lexical_weights", [])

                dense_result = [vec.tolist() if hasattr(vec, 'tolist') else vec for vec in dense_vecs]

                sparse_result = []
                for j in range(len(valid_texts)):
                    sparse_dict = lexical_weights[j] if j < len(lexical_weights) else {}
                    sparse_indices = [int(k) for k in sparse_dict.keys()] if sparse_dict else []
                    sparse_values = list(sparse_dict.values()) if sparse_dict else []
                    sparse_result.append({"indices": sparse_indices, "values": sparse_values})
            else:
                # SentenceTransformer doesn't support sparse vectors
                embeddings = self.model.encode(valid_texts, normalize_embeddings=True)
                dense_result = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
                # Return empty sparse vectors
                sparse_result = [{"indices": [], "values": []} for _ in range(len(valid_texts))]

            logger.debug(f"Successfully embedded {len(dense_result)} documents with hybrid vectors")
            return dense_result, sparse_result
        except Exception as e:
            logger.error(f"Error in embed_documents_hybrid: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query - returns dense vectors only"""
        try:
            if not text or not isinstance(text, str):
                logger.error(f"Invalid query text: {text}")
                raise ValueError("Query text must be a non-empty string")

            if self.use_bgem3:
                output = self.model.encode([text], return_dense=True, return_sparse=False)
                dense_vecs = output.get("dense_vecs", [])
                result = dense_vecs[0].tolist() if hasattr(dense_vecs[0], 'tolist') else dense_vecs[0]
            else:
                embedding = self.model.encode(text, normalize_embeddings=True)
                result = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

            logger.debug(f"Query embedding: dim={len(result)}")
            return result
        except Exception as e:
            logger.error(f"Error in embed_query: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def embed_query_hybrid(self, text: str) -> Tuple[List[float], Dict]:
        """Embed a single query - returns both dense and sparse vectors (BGEM3 only)"""
        try:
            if not text or not isinstance(text, str):
                logger.error(f"Invalid query text: {text}")
                raise ValueError("Query text must be a non-empty string")

            if self.use_bgem3:
                output = self.model.encode([text], return_dense=True, return_sparse=True)
                
                dense_vecs = output.get("dense_vecs", [])
                lexical_weights = output.get("lexical_weights", [])

                dense_result = dense_vecs[0].tolist() if hasattr(dense_vecs[0], 'tolist') else dense_vecs[0]

                sparse_dict = lexical_weights[0] if lexical_weights else {}
                sparse_indices = [int(k) for k in sparse_dict.keys()] if sparse_dict else []
                sparse_values = list(sparse_dict.values()) if sparse_dict else []
                sparse_result = {"indices": sparse_indices, "values": sparse_values}
            else:
                # SentenceTransformer doesn't support sparse vectors
                embedding = self.model.encode(text, normalize_embeddings=True)
                dense_result = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                sparse_result = {"indices": [], "values": []}

            logger.debug(f"Query embedding hybrid: dense dim={len(dense_result)}, sparse terms={len(sparse_result.get('indices', []))}")
            return dense_result, sparse_result
        except Exception as e:
            logger.error(f"Error in embed_query_hybrid: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

class VectorStoreService:
    """Vector Store service using Qdrant"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True

            try:
                # Initialize embedding model with the env-configured BGE-M3 path.
                local_model_path = settings.BGE_M3_MODEL_PATH
                logger.info(f"Loading BGE-M3 model from: {local_model_path}")

                # Check if BGE-M3 model exists, if not raise error
                if not os.path.exists(local_model_path):
                    raise FileNotFoundError(f"BGE-M3 model not found at: {local_model_path}")

                logger.info(f"Using BGE-M3 model: {local_model_path}")

                self.embedding = BGEM3EmbeddingFunction(local_model_path)

                # Test Qdrant connection
                logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
                self.client = QdrantClient(host=qdrant_host, port=int(qdrant_port))

                # Test the connection
                try:
                    self.client.get_collections()
                    logger.info("Qdrant connection successful")
                except Exception as e:
                    logger.error(f"Qdrant connection failed: {str(e)}")
                    raise

                self.collection_names: List[str] = []

                # BM25 retriever removed - using pure semantic search with LLM query expansion
                # from .bm25_retriever import BM25RetrieverService
                # self.bm25_retriever = BM25RetrieverService()
                # logger.info("BM25 retriever initialized for hybrid search")

                self.load_collections()

            except Exception as e:
                logger.error(f"Error initializing VectorStoreService: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

    def load_collections(self):
        """Load any existing collections"""
        try:
            collections = self.client.get_collections()
            self.collection_names = [col.name for col in collections.collections]
            logger.info(f"Found {len(self.collection_names)} existing collections: {self.collection_names}")
        except Exception as e:
            logger.error(f"Error loading existing collections: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")

    def get_or_create_collection(self, collection_name: str):
        """Get existing collection or create new one with hybrid (dense + sparse) vectors"""
        try:
            # If collection exists, ensure it has vector configuration.
            # Only delete+recreate if the collection is truly misconfigured AND empty.
            if collection_name in self.collection_names:
                try:
                    col_info = self.client.get_collection(collection_name)

                    # Correct attribute path for Qdrant Python client:
                    # col_info.config.params.vectors holds the named-vector config dict.
                    # The legacy top-level 'vectors_config' attribute does NOT exist on
                    # CollectionInfo in modern client versions — reading it always returns
                    # None, which previously caused every upload to wipe the collection.
                    try:
                        vec_conf = col_info.config.params.vectors
                    except AttributeError:
                        vec_conf = getattr(col_info, 'vectors_config', None)

                    is_unconfigured = (
                        vec_conf is None
                        or (isinstance(vec_conf, dict) and len(vec_conf) == 0)
                    )

                    if is_unconfigured:
                        # Safety guard: never delete a collection that has existing data.
                        points_count = getattr(col_info, 'points_count', None)
                        if points_count is None:
                            # Older client versions nest this under result
                            points_count = getattr(
                                getattr(col_info, 'result', col_info), 'points_count', 0
                            )

                        if points_count and int(points_count) > 0:
                            logger.warning(
                                f"Collection '{collection_name}' has unexpected vec_conf but contains "
                                f"{points_count} points — skipping recreation to preserve data"
                            )
                        else:
                            logger.warning(
                                f"Collection '{collection_name}' has no vectors configured and is empty; "
                                f"deleting and recreating"
                            )
                            try:
                                self.client.delete_collection(collection_name)
                            except Exception as del_e:
                                logger.error(f"Failed to delete empty collection {collection_name}: {del_e}")
                            try:
                                self.collection_names.remove(collection_name)
                            except ValueError:
                                pass
                except Exception as inspect_e:
                    logger.warning(f"Could not inspect existing collection {collection_name}: {inspect_e}")

            if collection_name not in self.collection_names:
                use_hybrid = self.embedding.use_bgem3
                logger.info(f"Creating new collection {'with hybrid vectors' if use_hybrid else '(dense only)'}: {collection_name}")

                # Get embedding dimension
                test_dense, test_sparse = self.embedding.embed_query_hybrid("test")
                vector_size = len(test_dense)

                # Enforce hybrid-only collections (dense + sparse)
                has_sparse = True
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(size=vector_size, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(index=SparseIndexParams())
                    }
                )
                self.collection_names.append(collection_name)
                logger.info(f"Created new collection: {collection_name} (sparse: {has_sparse})")
            return collection_name
        except Exception as e:
            logger.error(f"Error creating/getting collection {collection_name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def add_chunk(self, chunks: List[LangChainDocument], file_type: str) -> bool:
        """Add document chunks to the collection with hybrid (dense + sparse) vectors"""
        try:
            if not chunks:
                logger.error("No chunks provided to add_chunk")
                return False

            logger.info(f"Adding {len(chunks)} chunks to collection {file_type} with hybrid vectors")

            add_started_at = time.perf_counter()
            collection_name = self.get_or_create_collection(str(file_type))
            logger.info(
                "Vector indexing collection ready: %s (elapsed=%.2fs)",
                collection_name,
                time.perf_counter() - add_started_at,
            )

            points = []
            for start in range(0, len(chunks), EMBED_BATCH_SIZE):
                batch = chunks[start : start + EMBED_BATCH_SIZE]
                batch_number = (start // EMBED_BATCH_SIZE) + 1
                batch_started_at = time.perf_counter()
                logger.info(
                    "Vector batch %s started: collection=%s chunk_range=%s-%s batch_size=%s",
                    batch_number,
                    collection_name,
                    start,
                    start + len(batch) - 1,
                    len(batch),
                )
                batch_points = self._build_points_for_batch(batch, start)
                points.extend(batch_points)
                logger.info(
                    "Vector batch %s points built: collection=%s points=%s elapsed=%.2fs",
                    batch_number,
                    collection_name,
                    len(batch_points),
                    time.perf_counter() - batch_started_at,
                )

            for start in range(0, len(points), QDRANT_UPSERT_BATCH_SIZE):
                batch_points = points[start : start + QDRANT_UPSERT_BATCH_SIZE]
                upsert_batch_number = (start // QDRANT_UPSERT_BATCH_SIZE) + 1
                self._upsert_points(collection_name, batch_points, upsert_batch_number)

            logger.info(
                "Successfully added %s chunks to collection %s (total_elapsed=%.2fs)",
                len(chunks),
                file_type,
                time.perf_counter() - add_started_at,
            )
            return True
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def _normalize_chunk_identity(self, chunk: LangChainDocument, ordinal: int) -> tuple[str, int]:
        chunk_id = chunk.metadata.get("chunk_id")
        if not chunk_id:
            chunk_id = str(uuid.uuid4())
            logger.warning("Chunk missing chunk_id in metadata; generated fallback UUID")

        chunk_index = chunk.metadata.get("chunk_index")
        if chunk_index is None:
            chunk_index = ordinal
            chunk.metadata["chunk_index"] = chunk_index

        return str(chunk_id), int(chunk_index)

    def _build_point_id(self, chunk: LangChainDocument, chunk_id: str, chunk_index: int) -> str:
        source = chunk.metadata.get("source", "")
        page_number = chunk.metadata.get("page_number", chunk.metadata.get("page", 1))
        content_hash = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()
        stable_name = f"{source}:{page_number}:{chunk_index}:{chunk_id}:{content_hash}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, stable_name))

    def _build_payload(self, chunk: LangChainDocument, chunk_id: str, chunk_index: int) -> Dict:
        content_field = chunk.metadata.get("content") or chunk.page_content
        html_field = _ensure_chunk_html(chunk.metadata.get("html", ""), content_field)

        return {
            "page_content": chunk.page_content,
            "content": content_field,
            "html": html_field,
            "language": chunk.metadata.get("language", "unknown"),
            "chunk_id": chunk_id,
            "source": chunk.metadata.get("source", ""),
            "file_name": chunk.metadata.get("file_name", "PDF Document"),
            "page_number": chunk.metadata.get("page_number", chunk.metadata.get("page", 1)),
            "page_start": chunk.metadata.get("page_start", chunk.metadata.get("page_number", 1)),
            "page_end": chunk.metadata.get("page_end", chunk.metadata.get("page_number", 1)),
            "page_confidence": chunk.metadata.get("page_confidence", "medium"),
            "page_method": chunk.metadata.get("page_method", ""),
            "is_active": chunk.metadata.get("is_active", True),
            "header": chunk.metadata.get("header", ""),
            "header_level": chunk.metadata.get("header_level", 1),
            "chunk_index": chunk_index,
            "has_table": chunk.metadata.get("has_table", False),
            "extraction_method": chunk.metadata.get("extraction_method", ""),
        }

    def _build_points_for_batch(self, batch: List[LangChainDocument], batch_offset: int) -> List[models.PointStruct]:
        prepare_started_at = time.perf_counter()
        texts = []
        prepared = []

        for index, chunk in enumerate(batch):
            if not isinstance(chunk, LangChainDocument):
                raise TypeError(f"Chunk is not a LangChainDocument: {type(chunk)}")
            if not chunk.page_content or not isinstance(chunk.page_content, str):
                raise ValueError(f"Chunk has invalid page_content: {chunk.page_content}")

            chunk_id, chunk_index = self._normalize_chunk_identity(chunk, batch_offset + index)

            lang = chunk.metadata.get("language")
            if not lang:
                lang = detect_lang_quick(chunk.page_content)
                chunk.metadata["language"] = lang

            texts.append(chunk.page_content)
            prepared.append((chunk, chunk_id, chunk_index))

        logger.info(
            "Embedding batch start: chunk_offset=%s batch_size=%s prep_elapsed=%.2fs",
            batch_offset,
            len(prepared),
            time.perf_counter() - prepare_started_at,
        )
        embedding_started_at = time.perf_counter()
        dense_vecs, sparse_vecs = self.embedding.embed_documents_hybrid(texts)
        logger.info(
            "Embedding batch finished: chunk_offset=%s vectors=%s elapsed=%.2fs",
            batch_offset,
            len(dense_vecs),
            time.perf_counter() - embedding_started_at,
        )
        if len(dense_vecs) != len(prepared):
            raise ValueError(
                f"Embedding batch size mismatch: {len(dense_vecs)} vectors for {len(prepared)} chunks"
            )

        points_started_at = time.perf_counter()
        points = []
        for idx, (chunk, chunk_id, chunk_index) in enumerate(prepared):
            sparse_vec = sparse_vecs[idx] if idx < len(sparse_vecs) else {"indices": [], "values": []}
            vector_payload = {"dense": dense_vecs[idx]}
            if sparse_vec.get("indices"):
                vector_payload["sparse"] = SparseVector(
                    indices=sparse_vec["indices"],
                    values=sparse_vec["values"],
                )

            point = models.PointStruct(
                id=self._build_point_id(chunk, chunk_id, chunk_index),
                vector=vector_payload,
                payload=self._build_payload(chunk, chunk_id, chunk_index),
            )
            points.append(point)

        logger.info(
            "Point assembly finished: chunk_offset=%s points=%s elapsed=%.2fs",
            batch_offset,
            len(points),
            time.perf_counter() - points_started_at,
        )
        return points

    def _upsert_points(
        self,
        collection_name: str,
        points: List[models.PointStruct],
        batch_number: int | None = None,
    ) -> None:
        upsert_started_at = time.perf_counter()
        logger.info(
            "Qdrant upsert start: collection=%s batch=%s points=%s",
            collection_name,
            batch_number if batch_number is not None else "n/a",
            len(points),
        )
        try:
            self.client.upsert(collection_name=collection_name, points=points)
            logger.info(
                "Qdrant upsert finished: collection=%s batch=%s elapsed=%.2fs",
                collection_name,
                batch_number if batch_number is not None else "n/a",
                time.perf_counter() - upsert_started_at,
            )
        except UnexpectedResponse as e:
            err_str = str(e)
            logger.warning(f"Upsert failed, attempting retry: {err_str}")
            try:
                col_info = self.client.get_collection(collection_name)
                vec_config = getattr(col_info, 'vectors_config', None)
                existing_vector_name = None
                if isinstance(vec_config, dict) and len(vec_config) > 0:
                    existing_vector_name = list(vec_config.keys())[0]
                retry_points = []
                for point in points:
                    dense = point.vector.get('dense') if isinstance(point.vector, dict) else point.vector
                    new_vector_payload = {existing_vector_name: dense} if existing_vector_name else dense
                    retry_points.append(
                        models.PointStruct(id=point.id, vector=new_vector_payload, payload=point.payload)
                    )
                self.client.upsert(collection_name=collection_name, points=retry_points)
                logger.info(
                    "Upsert retry with existing vector name succeeded: collection=%s batch=%s elapsed=%.2fs",
                    collection_name,
                    batch_number if batch_number is not None else "n/a",
                    time.perf_counter() - upsert_started_at,
                )
            except Exception as retry_error:
                logger.error(f"Retry upsert failed: {retry_error}")
                raise

    def search_similar(self, query: str, k: int = 5, file_type: str = None, query_language: str = None):
        """Search for similar documents using hybrid vectors stored in Qdrant"""
        try:
            if not query or not isinstance(query, str):
                logger.error(f"Invalid query: {query}")
                return [], []

            logger.info(f"Searching for similar documents (hybrid): query='{query[:5]}', k={k}, file_type={file_type}, query_language={query_language}")

            if file_type:
                results = self.search_single_collection(str(file_type), query, k, query_language)
            else:
                results = self.search_all_collections(query, k, query_language)

            logger.info(f"Found {len(results)} results")

            return results, []

        except Exception as e:
            logger.error(f"Error in search_similar: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [], []

    def search_all_collections(self, query: str, k: int, query_language: str = None):
        """Search across all collections - uses dense query but collection has hybrid vectors"""
        if not self.collection_names:
            logger.warning("No collections available for search")
            return []

        all_results = []
        query_dense, _ = self.embedding.embed_query_hybrid(query)

        for collection_name in self.collection_names:
            try:
                # Build filter for language if specified
                query_filter = None
                filter_values = build_language_filter_values(query_language)
                if filter_values:
                    query_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="language",
                                match=models.MatchAny(any=filter_values)
                            )
                        ]
                    )

                # Use dense search (Qdrant uses both dense+sparse index automatically)
                search_result = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_dense,
                    limit=k,
                    query_filter=query_filter
                )

                for hit in search_result:
                    payload = hit.payload
                    doc = LangChainDocument(
                        page_content=payload.get("page_content", ""),
                        metadata={
                            "chunk_id": payload.get("chunk_id", hit.id),
                            "score": hit.score,
                            "language": payload.get("language"),
                            "source": payload.get("source"),
                            "file_name": payload.get("file_name"),
                            "page_number": payload.get("page_number"),
                            "file_type": collection_name,
                            "header": payload.get("header", ""),
                            "header_level": payload.get("header_level", 1),
                            "chunk_index": payload.get("chunk_index", 0),
                            "html": payload.get("html", ""),
                        }
                    )
                    all_results.append((doc, hit.score))

            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {str(e)}")
                continue

        # Sort by score and take top k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in all_results[:k]]

    def search_single_collection(self, collection_name: str, query: str, k: int, query_language: str = None):
        """Search a single collection - prefer hybrid prefetch+fusion queries. Returns list of LangChainDocument objects."""
        try:
            points = self.query_collection_hybrid(collection_name, query, k, query_language)
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

                doc = LangChainDocument(
                    page_content=payload.get("page_content", ""),
                    metadata={
                        "chunk_id": payload.get("chunk_id", chunk_id),
                        "score": score,
                        "language": payload.get("language"),
                        "source": payload.get("source"),
                        "file_name": payload.get("file_name"),
                        "page_number": payload.get("page_number"),
                        "file_type": collection_name,
                        "chunk_index": payload.get("chunk_index", 0),
                        "header": payload.get("header", ""),
                        "html": payload.get("html", ""),
                    }
                )
                results.append(doc)

            logger.info(f"Found {len(results)} results in collection {collection_name} (hybrid)")
            return results

        except Exception as e:
            logger.error(f"Error in search_single_collection: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def query_collection_hybrid(self, collection_name: str, query: str, k: int, query_language: str = None, source_filter: str = None):
        """Run a hybrid prefetch+fusion query against a single collection using BGEM3.

        Filters are pushed down to the Qdrant query level (not post-processed in Python)
        so the returned `k` results are already filtered — you always get `k` relevant hits.

        Args:
            source_filter: When set, restricts results to chunks whose ``source``
                           payload field equals this value (the document UUID /
                           file_id).  This is the correct way to honour document
                           mentions so that all ``k`` slots come from that document.

        Returns a list of hit objects (client-specific shapes) which include .payload and .score.
        """
        try:
            from qdrant_client.models import Prefetch, FusionQuery, Fusion

            # Build Qdrant-level filter: always require is_active=True + optional language
            must_conditions = [
                models.FieldCondition(
                    key="is_active",
                    match=models.MatchValue(value=True),
                )
            ]
            filter_values = build_language_filter_values(query_language)
            if filter_values:
                must_conditions.append(
                    models.FieldCondition(
                        key="language",
                        match=models.MatchAny(any=filter_values),
                    )
                )
            # Restrict to a single document when a mention is active
            if source_filter:
                must_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=str(source_filter)),
                    )
                )
            qdrant_filter = models.Filter(must=must_conditions)

            # Get dense + sparse query vectors
            query_dense, query_sparse = self.embedding.embed_query_hybrid(query)

            prefetchs = []
            # Dense prefetch with filter applied at candidate level
            prefetchs.append(Prefetch(
                query=query_dense,
                using="dense",
                limit=max(10, k * 2),
                filter=qdrant_filter,
            ))
            # Sparse prefetch only if available
            if isinstance(query_sparse, dict) and query_sparse.get("indices"):
                sparse_vec = SparseVector(
                    indices=query_sparse.get("indices", []),
                    values=query_sparse.get("values", []),
                )
                prefetchs.append(Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=max(10, k * 2),
                    filter=qdrant_filter,
                ))

            fusion_query = FusionQuery(fusion=Fusion.RRF)

            resp = self.client.query_points(
                collection_name=collection_name,
                prefetch=prefetchs,
                query=fusion_query,
                limit=k,
                with_payload=True,
                query_filter=qdrant_filter,
            )

            return getattr(resp, 'points', resp)

        except Exception as e:
            logger.error(f"Error in query_collection_hybrid: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def delete_chunk(self, chunk_ids: List[str], file_type: str) -> bool:
        """Delete specific chunks by their IDs"""
        try:
            collection_name = str(file_type)

            # Find point IDs that have the chunk_ids in payload
            point_ids_to_delete = []
            for chunk_id in chunk_ids:
                scroll_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="chunk_id",
                                match=models.MatchValue(value=chunk_id)
                            )
                        ]
                    ),
                    limit=1
                )
                for point in scroll_result[0]:
                    point_ids_to_delete.append(point.id)

            if point_ids_to_delete:
                # Delete points by IDs
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=point_ids_to_delete)
                )
                logger.info(f"Deleted {len(point_ids_to_delete)} points for chunks: {chunk_ids}")
            else:
                logger.warning(f"No points found for chunks: {chunk_ids}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks: {str(e)}")
            return False

    def get_chunk_ids_by_file_id(self, file_id: str, file_type: str) -> List[str]:
        """Retrieve all chunk IDs associated with a specific file ID from a collection."""
        try:
            collection_name = str(file_type)

            # Query for points with matching source
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=file_id)
                        )
                    ]
                ),
                limit=10000  # Adjust as needed
            )

            chunk_ids = [point.payload.get('chunk_id', point.id) for point in scroll_result[0]]
            logger.info(f"Found {len(chunk_ids)} chunk IDs for file_id: {file_id}")
            return chunk_ids

        except Exception as e:
            logger.error(f"Error retrieving chunk IDs: {str(e)}")
            return []

    def find_file_ids_by_filename(self, file_name: str, collection_name: str) -> List[str]:
        """Find all distinct source UUIDs (file_ids) for active chunks with a given file_name.

        Used to detect duplicate uploads: if the same file_name already has active chunks
        in the collection, those old file_ids are returned so they can be deactivated
        before the new version is indexed.
        """
        try:
            if collection_name not in self.collection_names:
                return []
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_name",
                            match=models.MatchValue(value=file_name),
                        ),
                        models.FieldCondition(
                            key="is_active",
                            match=models.MatchValue(value=True),
                        ),
                    ]
                ),
                limit=10000,
                with_payload=True,
            )
            file_ids: set = set()
            for point in scroll_result[0]:
                payload = getattr(point, 'payload', {}) or {}
                src = payload.get('source')
                if src:
                    file_ids.add(str(src))
            logger.info(f"find_file_ids_by_filename '{file_name}' → {len(file_ids)} existing file_id(s)")
            return list(file_ids)
        except Exception as e:
            logger.error(f"find_file_ids_by_filename error for '{file_name}': {e}")
            return []

    def deactivate_document(self, file_id: str, file_type: str):
        """Mark all chunks for file_id as is_active=False in-place.

        Previously this method moved chunks to a separate 'inactive' collection.
        That approach caused data loss because the inactive collection had a different
        vector configuration (Default 1024 Cosine) from active collections (named
        dense+sparse vectors) — Qdrant would silently reject the upsert while the
        subsequent delete still ran, wiping the source chunks permanently.

        The new approach keeps chunks in their original collection and simply
        sets is_active=False via set_payload.  The is_active=True filter in every
        query already ensures inactive chunks are never retrieved.
        """
        try:
            if not file_id or not file_type:
                logger.error(f"Invalid parameters: file_id={file_id}, file_type={file_type}")
                return False

            source_collection = str(file_type)

            # Locate the collection that holds this document
            if source_collection not in self.collection_names:
                logger.warning(
                    f"Collection '{source_collection}' not found. "
                    f"Searching all collections for file_id={file_id}"
                )
                source_collection = None
                for col in self.collection_names:
                    if col in ("inactive", "history", "history_inactive"):
                        continue
                    try:
                        probe = self.client.scroll(
                            collection_name=col,
                            scroll_filter=models.Filter(
                                must=[models.FieldCondition(
                                    key="source",
                                    match=models.MatchValue(value=file_id)
                                )]
                            ),
                            limit=1,
                        )
                        if probe[0]:
                            source_collection = col
                            break
                    except Exception:
                        continue

                if not source_collection:
                    logger.warning(f"Document {file_id} not found in any collection — skipping deactivation")
                    return True  # Not an error

            # Scroll to collect point IDs (no vectors needed — no copy is made)
            scroll_result = self.client.scroll(
                collection_name=source_collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=file_id)
                    )]
                ),
                limit=10000,
                with_payload=False,
                with_vectors=False,
            )

            if not scroll_result[0]:
                logger.warning(f"No chunks found for file_id={file_id}")
                return True

            point_ids = [p.id for p in scroll_result[0]]
            logger.info(f"Deactivating {len(point_ids)} chunks for file_id={file_id} in '{source_collection}'")

            # Set is_active=False in-place — safe, no vector copy, no data loss
            self.client.set_payload(
                collection_name=source_collection,
                payload={"is_active": False},
                points=point_ids,
            )

            logger.info(f"Deactivated {len(point_ids)} chunks for file {file_id}")
            return True

        except Exception as e:
            logger.error(f"Error deactivating document: {str(e)}")
            return False

    def activate_document(self, file_id: str):
        """Set is_active=True for all chunks belonging to file_id.

        Primary path: chunks live in their original collection with is_active=False
        (the new in-place deactivation approach) → set_payload is_active=True.

        Fallback path: chunks were previously moved to the 'inactive' collection
        (old approach) → copy back to original collection and delete from inactive.
        """
        try:
            if not file_id:
                logger.error(f"Invalid file_id: {file_id}")
                return False

            activated = 0

            # ── Primary path: set_payload in each active collection ──────────
            for col in self.collection_names:
                if col in ("inactive", "history", "history_inactive"):
                    continue
                try:
                    scroll_result = self.client.scroll(
                        collection_name=col,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="source",
                                    match=models.MatchValue(value=file_id)
                                ),
                                models.FieldCondition(
                                    key="is_active",
                                    match=models.MatchValue(value=False)
                                ),
                            ]
                        ),
                        limit=10000,
                        with_payload=False,
                        with_vectors=False,
                    )
                    if scroll_result[0]:
                        point_ids = [p.id for p in scroll_result[0]]
                        self.client.set_payload(
                            collection_name=col,
                            payload={"is_active": True},
                            points=point_ids,
                        )
                        activated += len(point_ids)
                        logger.info(f"Re-activated {len(point_ids)} chunks in '{col}' for file_id={file_id}")
                except Exception as col_err:
                    logger.warning(f"activate_document: error scanning '{col}': {col_err}")

            if activated > 0:
                logger.info(f"Activated {activated} chunks for file {file_id}")
                return True

            # ── Fallback path: chunks in the legacy 'inactive' collection ────
            inactive_collection = "inactive"
            if inactive_collection not in self.collection_names:
                logger.warning(f"No chunks found for file_id={file_id} in any collection")
                return False

            scroll_result = self.client.scroll(
                collection_name=inactive_collection,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=file_id)
                    )]
                ),
                limit=10000,
                with_vectors=True,
            )

            if not scroll_result[0]:
                logger.warning(f"No chunks found in inactive collection for file_id={file_id}")
                return False

            points_by_type: dict = {}
            ids_to_delete = []

            for point in scroll_result[0]:
                if point.vector is None:
                    continue
                payload = point.payload or {}
                file_type = payload.get("file_type", "unknown")
                payload["is_active"] = True

                # Normalize vector
                vec = point.vector
                if isinstance(vec, np.ndarray):
                    vec = vec.tolist()
                elif isinstance(vec, dict):
                    vec = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vec.items()}

                new_point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=payload,
                )
                points_by_type.setdefault(file_type, []).append(new_point)
                ids_to_delete.append(point.id)

            total_moved = 0
            for file_type, points in points_by_type.items():
                target = self.get_or_create_collection(file_type)
                self.client.upsert(collection_name=target, points=points)
                total_moved += len(points)

            if ids_to_delete:
                self.client.delete(
                    collection_name=inactive_collection,
                    points_selector=models.PointIdsList(points=ids_to_delete),
                )

            logger.info(f"Activated {total_moved} chunks (from legacy inactive) for file {file_id}")
            return True

        except Exception as e:
            logger.error(f"Error activating document: {str(e)}")
            return False

    def hybrid_search(self, query: str, k: int = 10, file_type: str = None,
                      query_language: str = None, alpha: float = 0.5, expand_context: bool = False) -> List[Tuple[LangChainDocument, float]]:
        """
        Enhanced hybrid search with query expansion, multi-stage retrieval, and re-ranking.
        """
        try:
            logger.info(f"Enhanced hybrid search: query='{query[:50]}...', k={k}, alpha={alpha}")

            # Stage 1: Query Expansion and Multi-Query Generation
            expanded_queries = self._expand_query(query, query_language)
            logger.info(f"Query expansion generated {len(expanded_queries)} variants")

            # Dynamic Alpha Adjustment based on query characteristics
            alpha = self._calculate_optimal_alpha(query, expanded_queries)

            candidate_k = min(k * 3, 30)  # Increased for better coverage

            # Stage 2: Multi-Stage Retrieval
            all_dense_results = []
            all_sparse_results = []

            # Search with original and expanded queries
            for query_variant in [query] + expanded_queries[:2]:  # Limit to 3 queries max
                # Dense retrieval for each query variant
                dense_results, _ = self.search_similar(query_variant, candidate_k, file_type, query_language)
                all_dense_results.extend(dense_results)

                # BM25 retrieval for each query variant
                collection_name = str(file_type) if file_type else "all"
                if collection_name not in self.bm25_retriever.indices:
                    if file_type:
                        docs = self._get_all_docs_from_collection(file_type)
                        self.bm25_retriever.build_index(collection_name, docs)
                        logger.info(f"Built BM25 index for collection '{collection_name}'")

                bm25_results = self.bm25_retriever.search(query_variant, candidate_k, collection_name)
                all_sparse_results.extend(bm25_results)

            # Remove duplicates while preserving scores
            dense_results = self._deduplicate_results(all_dense_results)
            sparse_results = self._deduplicate_sparse_results(all_sparse_results)

            logger.info(f"Dense retrieval returned {len(dense_results)} unique results")
            logger.info(f"BM25 retrieval returned {len(sparse_results)} unique results")

            # Stage 3: Advanced RRF Merging with Re-ranking
            merged_results = self._advanced_rrf_merge(
                dense_results=dense_results,
                sparse_results=sparse_results,
                k=k,
                alpha=alpha,
                original_query=query
            )

            # Stage 4: Optional Context Expansion
            if expand_context and len(merged_results) < k:
                expanded_results = self._expand_context(merged_results, query, k, file_type)
                merged_results.extend(expanded_results)
                merged_results = merged_results[:k]  # Trim to requested k

            logger.info(f"Enhanced hybrid search completed with {len(merged_results)} results")
            return merged_results

        except Exception as e:
            logger.error(f"Error in enhanced hybrid_search: {str(e)}")
            # Fallback to basic dense search
            dense_results, _ = self.search_similar(query, k, file_type, query_language)
            return [(doc, doc.metadata.get('score', 0.0)) for doc in dense_results]

    def _expand_query(self, query: str, language: str = None) -> List[str]:
        """
        Generate query expansions for better retrieval coverage.
        """
        try:
            expansions = []

            # Basic keyword expansion
            words = query.split()
            if len(words) > 2:
                # Add bigrams
                for i in range(len(words) - 1):
                    expansions.append(f"{words[i]} {words[i+1]}")

            # Language-specific expansions
            if language == 'km':
                # Khmer-specific expansions could be added here
                pass
            elif language == 'en':
                # English synonyms and related terms
                synonym_map = {
                    'policy': ['guideline', 'procedure', 'rule', 'regulation'],
                    'benefit': ['advantage', 'perk', 'compensation'],
                    'salary': ['pay', 'wage', 'compensation'],
                    'leave': ['vacation', 'holiday', 'time off'],
                }

                for word in words:
                    word_lower = word.lower()
                    if word_lower in synonym_map:
                        for synonym in synonym_map[word_lower][:2]:  # Limit synonyms
                            expanded = query.replace(word, synonym)
                            if expanded != query:
                                expansions.append(expanded)

            # Remove duplicates and limit
            expansions = list(set(expansions))[:3]  # Max 3 expansions

            return expansions

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return []

    def _calculate_optimal_alpha(self, original_query: str, expanded_queries: List[str]) -> float:
        """
        Dynamically calculate optimal alpha based on query characteristics.
        """
        try:
            words = original_query.split()
            has_digits = any(char.isdigit() for char in original_query)
            has_expansions = len(expanded_queries) > 0

            # Base alpha calculation
            if len(words) < 3 or has_digits:
                alpha = 0.2  # Favor BM25 for short/numeric queries
            elif len(words) > 8:
                alpha = 0.8  # Favor dense for long conceptual queries
            else:
                alpha = 0.5  # Balanced for medium queries

            # Adjust based on expansions
            if has_expansions:
                alpha = min(alpha + 0.1, 0.8)  # Slightly favor dense when expansions available

            logger.info(f"Calculated optimal alpha: {alpha} (words: {len(words)}, digits: {has_digits}, expansions: {len(expanded_queries)})")
            return alpha

        except Exception as e:
            logger.warning(f"Alpha calculation failed: {e}")
            return 0.5

    def _deduplicate_results(self, results: List[LangChainDocument]) -> List[LangChainDocument]:
        """Remove duplicate documents based on content similarity."""
        try:
            seen_chunks = set()
            deduplicated = []

            for doc in results:
                # Use chunk_id or content hash for deduplication
                chunk_id = doc.metadata.get('chunk_id', hash(doc.page_content) % 10000)
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    deduplicated.append(doc)

            return deduplicated

        except Exception as e:
            logger.warning(f"Deduplication failed: {e}")
            return results

    def _deduplicate_sparse_results(self, results: List[Tuple[LangChainDocument, float]]) -> List[Tuple[LangChainDocument, float]]:
        """Remove duplicate sparse results while preserving scores."""
        try:
            seen_chunks = set()
            deduplicated = []

            for doc, score in results:
                chunk_id = doc.metadata.get('chunk_id', hash(doc.page_content) % 10000)
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    deduplicated.append((doc, score))

            return deduplicated

        except Exception as e:
            logger.warning(f"Sparse deduplication failed: {e}")
            return results

    def _advanced_rrf_merge(self, dense_results: List[LangChainDocument],
                           sparse_results: List[Tuple[LangChainDocument, float]],
                           k: int, alpha: float, original_query: str) -> List[Tuple[LangChainDocument, float]]:
        """
        Advanced RRF merging with re-ranking based on relevance to original query.
        """
        try:
            K = 60
            doc_scores: Dict[str, Tuple[LangChainDocument, float, Dict]] = {}

            # Process dense results with enhanced scoring
            for rank, doc in enumerate(dense_results, start=1):
                chunk_id = doc.metadata.get('chunk_id', str(rank))
                base_rrf_score = alpha / (K + rank)

                # Calculate query relevance bonus
                relevance_bonus = self._calculate_query_relevance(doc.page_content, original_query)

                total_score = base_rrf_score * (1 + relevance_bonus)
                doc_scores[chunk_id] = (doc, total_score, {'source': 'dense', 'rank': rank, 'relevance_bonus': relevance_bonus})

            # Process sparse results
            for rank, (doc, bm25_score) in enumerate(sparse_results, start=1):
                chunk_id = doc.metadata.get('chunk_id', f"bm25_{rank}")
                sparse_rrf_score = (1 - alpha) / (K + rank) * bm25_score

                # Calculate query relevance bonus
                relevance_bonus = self._calculate_query_relevance(doc.page_content, original_query)

                total_score = sparse_rrf_score * (1 + relevance_bonus)

                if chunk_id in doc_scores:
                    # Merge scores if document appears in both
                    existing_doc, existing_score, existing_meta = doc_scores[chunk_id]
                    combined_score = existing_score + total_score
                    combined_meta = {
                        'source': 'hybrid',
                        'dense_rank': existing_meta.get('rank'),
                        'sparse_rank': rank,
                        'relevance_bonus': max(existing_meta.get('relevance_bonus', 0), relevance_bonus)
                    }
                    doc_scores[chunk_id] = (existing_doc, combined_score, combined_meta)
                else:
                    doc_scores[chunk_id] = (doc, total_score, {'source': 'sparse', 'rank': rank, 'relevance_bonus': relevance_bonus})

            # Sort by combined score and return top k
            sorted_results = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)

            final_results = []
            for doc, score, meta in sorted_results[:k]:
                # Add metadata about ranking source
                doc.metadata['ranking_info'] = meta
                doc.metadata['final_score'] = score
                final_results.append((doc, score))

            logger.info(f"Advanced RRF merged {len(sorted_results)} candidates to {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Advanced RRF merge failed: {e}")
            # Fallback to basic RRF
            return self.merge_with_rrf(dense_results, sparse_results, k, alpha)

    def _calculate_query_relevance(self, content: str, query: str) -> float:
        """
        Calculate relevance bonus based on query term matches in content.
        """
        try:
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())

            # Calculate word overlap
            overlap = len(query_words.intersection(content_words))
            overlap_ratio = overlap / len(query_words) if query_words else 0

            # Bonus for exact phrase matches
            phrase_bonus = 1.0 if query.lower() in content.lower() else 0.0

            # Calculate final relevance bonus (max 0.5)
            relevance_bonus = min((overlap_ratio * 0.3) + (phrase_bonus * 0.2), 0.5)

            return relevance_bonus

        except Exception as e:
            logger.warning(f"Query relevance calculation failed: {e}")
            return 0.0

    def _expand_context(self, existing_results: List[Tuple[LangChainDocument, float]],
                       query: str, target_k: int, file_type: str = None) -> List[Tuple[LangChainDocument, float]]:
        """
        Expand context by finding related documents when we have fewer than requested results.
        """
        try:
            if len(existing_results) >= target_k:
                return []

            # Extract keywords from existing results to find related content
            keywords = self._extract_keywords_from_results(existing_results)

            expanded_results = []
            for keyword in keywords[:3]:  # Limit to 3 keywords
                # Search for related content using keywords
                related_results, _ = self.search_similar(keyword, target_k - len(existing_results), file_type)

                # Filter out already retrieved documents
                existing_ids = {doc.metadata.get('chunk_id') for doc, _ in existing_results}
                new_results = [(doc, score * 0.7) for doc, score in related_results
                             if doc.metadata.get('chunk_id') not in existing_ids]

                expanded_results.extend(new_results[:2])  # Max 2 per keyword

            logger.info(f"Context expansion added {len(expanded_results)} related documents")
            return expanded_results

        except Exception as e:
            logger.warning(f"Context expansion failed: {e}")
            return []

    def _extract_keywords_from_results(self, results: List[Tuple[LangChainDocument, float]]) -> List[str]:
        """Extract important keywords from existing results for context expansion."""
        try:
            keywords = set()

            for doc, _ in results[:3]:  # Analyze top 3 results
                content = doc.page_content.lower()
                words = content.split()

                # Extract noun-like words (simple heuristic)
                for word in words:
                    word = word.strip('.,!?()[]{}')
                    if len(word) > 4 and not word.isdigit():  # Longer meaningful words
                        keywords.add(word)

            return list(keywords)[:5]  # Return top 5 keywords

        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []

    def _get_all_docs_from_collection(self, collection_name: str) -> List[LangChainDocument]:
        """Get all documents from a collection for BM25 indexing"""
        try:
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                limit=10000
            )

            docs = []
            for point in scroll_result[0]:
                payload = point.payload
                doc = LangChainDocument(
                    page_content=payload.get("page_content", ""),
                    metadata={
                        "chunk_id": payload.get("chunk_id", point.id),
                        "source": payload.get("source"),
                        "file_name": payload.get("file_name"),
                        "html": payload.get("html", ""),
                    }
                )
                docs.append(doc)
            return docs
        except Exception as e:
            logger.error(f"Error getting docs from collection: {str(e)}")
            return []

    def merge_with_rrf(self, dense_results: List[LangChainDocument],
                       sparse_results: List[Tuple[LangChainDocument, float]],
                       k: int, alpha: float = 0.5) -> List[Tuple[LangChainDocument, float]]:
        """
        Merge dense and sparse results using Reciprocal Rank Fusion (RRF).
        """
        try:
            K = 60
            doc_scores: Dict[str, Tuple[LangChainDocument, float]] = {}

            # Process dense results
            for rank, doc in enumerate(dense_results, start=1):
                chunk_id = doc.metadata.get('chunk_id', str(rank))
                rrf_score = alpha / (K + rank)
                doc_scores[chunk_id] = (doc, rrf_score)

            # Process sparse results
            for rank, (doc, _) in enumerate(sparse_results, start=1):
                chunk_id = doc.metadata.get('chunk_id', f"bm25_{rank}")
                rrf_score = (1 - alpha) / (K + rank)

                if chunk_id in doc_scores:
                    existing_doc, existing_score = doc_scores[chunk_id]
                    doc_scores[chunk_id] = (existing_doc, existing_score + rrf_score)
                else:
                    doc_scores[chunk_id] = (doc, rrf_score)

            # Sort and return top k
            merged = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
            results = []
            for doc, score in merged[:k]:
                doc.metadata['rrf_score'] = score
                results.append((doc, score))

            return results

        except Exception as e:
            logger.error(f"Error in merge_with_rrf: {str(e)}")
            return [(doc, doc.metadata.get('score', 0.0)) for doc in dense_results[:k]]

    def filter_by_score(self, results: List[Tuple[LangChainDocument, float]],
                        threshold: float = 0.5) -> List[LangChainDocument]:
        """
        Filter results by minimum score threshold.
        """
        filtered = [doc for doc, score in results if score >= threshold]
        logger.info(f"Filtered {len(results)} results to {len(filtered)} (threshold={threshold})")
        return filtered
