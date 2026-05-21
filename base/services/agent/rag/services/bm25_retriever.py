"""
BM25 Retriever Service for Sparse Keyword-Based Retrieval

This service provides BM25 (Best Matching 25) sparse retrieval to complement
the dense semantic retrieval. BM25 is particularly effective for:
- Exact keyword matching
- Proper noun searches
- Technical term queries
- Short queries with specific keywords

The service supports both English and Khmer languages with appropriate tokenization.
"""

from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict, Optional
import logging
import re
from langchain_core.documents import Document as LangChainDocument

logger = logging.getLogger(__name__)


def tokenize_text(text: str, language: str = 'en') -> List[str]:
    """
    Tokenize text based on language.
    
    Args:
        text: Input text to tokenize
        language: 'en' for English, 'km' for Khmer
        
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    text = text.lower().strip()
    
    if language == 'km':
        # Khmer tokenization: split by spaces and common Khmer punctuation
        # Khmer is written with spaces between phrases/words
        tokens = re.findall(r'[\u1780-\u17FF]+|[a-zA-Z0-9]+', text)
    else:
        # English tokenization: split by whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
    
    return [t for t in tokens if len(t) > 1]  # Filter out single characters


def detect_language(text: str) -> str:
    """Quick language detection based on character set."""
    if not text:
        return 'en'
    khmer_count = sum(1 for ch in text if '\u1780' <= ch <= '\u17FF')
    total_chars = len(text.replace(' ', ''))
    return 'km' if khmer_count / (total_chars + 1e-6) > 0.2 else 'en'


class BM25RetrieverService:
    """
    BM25 sparse retriever for keyword-based document search.
    
    Uses BM25Okapi algorithm with language-aware tokenization.
    Builds and caches indices per collection for fast retrieval.
    """
    
    def __init__(self):
        """Initialize BM25 retriever with empty index cache."""
        self.indices: Dict[str, BM25Okapi] = {}  # collection_name -> BM25 index
        self.documents: Dict[str, List[LangChainDocument]] = {}  # collection_name -> docs
        self.tokenized_corpus: Dict[str, List[List[str]]] = {}  # collection_name -> tokenized docs
        logger.info("BM25RetrieverService initialized")
    
    def build_index(self, collection_name: str, documents: List[LangChainDocument]) -> bool:
        """
        Build BM25 index for a collection of documents.
        
        Args:
            collection_name: Name of the collection (e.g., file_type)
            documents: List of LangChain documents to index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning(f"No documents provided for collection: {collection_name}")
                return False
            
            logger.info(f"Building BM25 index for collection '{collection_name}' with {len(documents)} documents")
            
            # Detect language from first document
            sample_text = documents[0].page_content if documents else ""
            language = detect_language(sample_text)
            logger.info(f"Detected language for BM25 indexing: {language}")
            
            # Tokenize all documents
            tokenized_corpus = []
            for doc in documents:
                tokens = tokenize_text(doc.page_content, language)
                tokenized_corpus.append(tokens)
            
            # Build BM25 index
            bm25_index = BM25Okapi(tokenized_corpus)
            
            # Cache the index and documents
            self.indices[collection_name] = bm25_index
            self.documents[collection_name] = documents
            self.tokenized_corpus[collection_name] = tokenized_corpus
            
            logger.info(f"BM25 index built successfully for '{collection_name}': {len(documents)} docs")
            return True
            
        except Exception as e:
            logger.error(f"Error building BM25 index for '{collection_name}': {e}")
            return False
    
    def search(self, query: str, k: int = 10, collection_name: str = None) -> List[Tuple[LangChainDocument, float]]:
        """
        Search documents using BM25 scoring.
        
        Args:
            query: Search query
            k: Number of top results to return
            collection_name: Specific collection to search (None = search all)
            
        Returns:
            List of (document, bm25_score) tuples, sorted by score descending
        """
        try:
            if not query:
                logger.warning("Empty query provided to BM25 search")
                return []
            
            # Detect query language and tokenize
            query_lang = detect_language(query)
            tokenized_query = tokenize_text(query, query_lang)
            
            if not tokenized_query:
                logger.warning(f"Query tokenization resulted in empty tokens: '{query}'")
                return []
            
            logger.info(f"BM25 search query (lang={query_lang}): '{query}' -> tokens: {tokenized_query}")
            
            all_results = []
            
            # Determine which collections to search
            collections_to_search = (
                [collection_name] if collection_name and collection_name in self.indices
                else list(self.indices.keys())
            )
            
            if not collections_to_search:
                logger.warning(f"No BM25 indices available for search (requested: {collection_name})")
                return []
            
            # Search each collection
            for coll_name in collections_to_search:
                bm25_index = self.indices[coll_name]
                documents = self.documents[coll_name]
                
                # Get BM25 scores for all documents
                scores = bm25_index.get_scores(tokenized_query)
                
                # Pair documents with scores
                doc_score_pairs = list(zip(documents, scores))
                
                # Filter out zero scores (no keyword match)
                doc_score_pairs = [(doc, score) for doc, score in doc_score_pairs if score > 0]
                
                all_results.extend(doc_score_pairs)
            
            # Sort by score descending and take top k
            all_results.sort(key=lambda x: x[1], reverse=True)
            top_results = all_results[:k]
            
            logger.info(f"BM25 search returned {len(top_results)} results (from {len(all_results)} total matches)")
            if top_results:
                logger.debug(f"Top BM25 score: {top_results[0][1]:.4f}, Bottom: {top_results[-1][1]:.4f}")
            
            return top_results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def update_index(self, collection_name: str, new_documents: List[LangChainDocument]) -> bool:
        """
        Update existing index with new documents.
        
        Args:
            collection_name: Collection to update
            new_documents: New documents to add
            
        Returns:
            True if successful
        """
        try:
            if collection_name not in self.documents:
                logger.info(f"Collection '{collection_name}' not found, creating new index")
                return self.build_index(collection_name, new_documents)
            
            # Append new documents and rebuild index
            existing_docs = self.documents[collection_name]
            all_docs = existing_docs + new_documents
            
            return self.build_index(collection_name, all_docs)
            
        except Exception as e:
            logger.error(f"Error updating BM25 index for '{collection_name}': {e}")
            return False
    
    def clear_index(self, collection_name: str = None) -> bool:
        """
        Clear BM25 index for a collection or all collections.
        
        Args:
            collection_name: Specific collection to clear (None = clear all)
            
        Returns:
            True if successful
        """
        try:
            if collection_name:
                if collection_name in self.indices:
                    del self.indices[collection_name]
                    del self.documents[collection_name]
                    del self.tokenized_corpus[collection_name]
                    logger.info(f"Cleared BM25 index for collection '{collection_name}'")
            else:
                self.indices.clear()
                self.documents.clear()
                self.tokenized_corpus.clear()
                logger.info("Cleared all BM25 indices")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing BM25 index: {e}")
            return False







