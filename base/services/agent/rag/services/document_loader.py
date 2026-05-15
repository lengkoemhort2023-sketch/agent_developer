import logging
import torch
import threading
from .token_logger import TokenLogger
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from django.conf import settings
logger = logging.getLogger(__name__)
from langchain_core.documents import Document as LangChainDocument
from PyPDF2 import PdfReader, PdfWriter
import concurrent.futures
from typing import List, Optional, Dict, Any
import base64
import os
import io
import json
import uuid
import time
from langdetect import detect, DetectorFactory
from typing import Optional
import re
import docx
from PIL import Image
# Additional imports for new file types
import mammoth  # For DOC files
from pptx import Presentation  # For PowerPoint files
import openpyxl  # For XLSX files
import xlrd  # For XLS files
import pandas as pd  # For CSV/TSV files

# Semantic chunking imports
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings

# Import BGE-M3 for semantic chunking (same model used for vector embeddings)
try:
    from FlagEmbedding import BGEM3FlagModel
    BGEM3_AVAILABLE = True
except ImportError:
    BGEM3_AVAILABLE = False
    BGEM3FlagModel = None
    logger.warning("FlagEmbedding not available for semantic chunking")


# Import Surya OCR dependencies for fallback
try:
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    SURYA_MODELS_AVAILABLE = True
except ImportError:
    SURYA_MODELS_AVAILABLE = False


class BGEM3SemanticEmbeddings(Embeddings):
    """
    Minimal LangChain Embeddings wrapper around BGEM3FlagModel for use with
    SemanticChunker. Only dense vectors are used (sparse not needed for chunking).
    """

    def __init__(self, model_path: str):
        cuda_available = torch.cuda.is_available()
        use_fp16 = cuda_available
        device_label = "CUDA (FP16)" if cuda_available else "CPU (FP32)"
        logger.info(f"[DocumentLoader] Loading BGE-M3 for semantic chunking on {device_label}.")
        self._model = BGEM3FlagModel(model_path, use_fp16=use_fp16)
        logger.info(f"[DocumentLoader] BGE-M3 semantic chunking model ready (cuda={cuda_available}).")

    def embed_documents(self, texts: list) -> list:
        output = self._model.encode(texts, return_dense=True, return_sparse=False)
        return output["dense_vecs"].tolist()

    def embed_query(self, text: str) -> list:
        output = self._model.encode([text], return_dense=True, return_sparse=False)
        return output["dense_vecs"][0].tolist()


class DocumentLoaderService:
    """LangChain document loading """

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
            self.token_logger = TokenLogger("logdata/token_usage.csv")
            # self.configure_langsmith()

            # Text Splitters - using langchain_text_splitters for better table and structure handling
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,  # updated chunk size
                chunk_overlap=204,  # updated overlap
            )

            # Markdown-aware splitter for structured content
            self.markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                    ("####", "Header 4"),
                ]
            )

            # Semantic chunker will be initialized lazily to avoid CUDA issues in forked processes
            self._semantic_chunker_initialized = False
            self.semantic_chunker = None
            self.semantic_embeddings = None

    def _initialize_semantic_chunker(self):
        """Lazy initialization of semantic chunker to avoid CUDA issues in forked processes."""
        if self._semantic_chunker_initialized:
            return

        try:
            local_model_path = settings.BGE_M3_MODEL_PATH
            logger.info(f"Loading BGE-M3 model from: {local_model_path}")

            if not os.path.exists(local_model_path):
                raise FileNotFoundError(f"BGE-M3 model not found at: {local_model_path}")

            if not BGEM3_AVAILABLE:
                raise ImportError("FlagEmbedding (BGEM3FlagModel) is required for semantic chunking.")

            self.semantic_embeddings = BGEM3SemanticEmbeddings(local_model_path)

            self.semantic_chunker = SemanticChunker(
                embeddings=self.semantic_embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=75,  # More aggressive chunking
                buffer_size=1,
                min_chunk_size=100,
            )
            logger.info("Semantic chunker initialized successfully")
            self._semantic_chunker_initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize semantic chunker: {e}. Falling back to traditional chunking.")
            self.semantic_chunker = None
            self._semantic_chunker_initialized = True  # Don't try again
    def _deduplicate_pdf_content(self, langchain_docs: List[LangChainDocument]) -> List[LangChainDocument]:
        """
        Remove duplicate content from PDF pages.
        This handles cases where the same content appears on multiple pages.
        """
        if not langchain_docs:
            return langchain_docs

        seen_content = set()
        unique_docs = []
        
        for doc in langchain_docs:
            # Create a hash of the content to detect duplicates
            content_hash = hash(doc.page_content.strip())
            
            # Also check for similar content (allowing for minor variations)
            content_normalized = re.sub(r'\s+', ' ', doc.page_content.strip().lower())
            
            if content_hash not in seen_content and len(content_normalized) > 100:  # Skip very short content
                seen_content.add(content_hash)
                unique_docs.append(doc)
            else:
                logger.debug(f"Skipping duplicate content on page {doc.metadata.get('page', 'unknown')}")

        logger.info(f"Deduplication: {len(langchain_docs)} -> {len(unique_docs)} documents")
        return unique_docs

    def load_pdf(self, file_byte: bytes, file_id: str, file_name: str):
        """Loading PDF using direct extraction first, with Surya as fallback"""
        logger.info(f"Starting PDF loading for {file_name} (ID: {file_id})")

        # Only use Surya OCR for PDF extraction, skip direct extraction and conversion
        try:
            logger.info(f"Extracting PDF using Surya OCR only for {file_name}")
            from .extraction.surya_docling import extract_pdf_to_langchain_docs

            langchain_docs = extract_pdf_to_langchain_docs(
                file_bytes=file_byte,
                file_id=file_id,
                file_name=file_name,
                max_pages=None,  # Process all pages
                use_layout=True
            )

            if langchain_docs:
                print(f"Successfully processed PDF {file_name} using Surya OCR: {len(langchain_docs)} chunks")

                # Deduplicate content to handle repetitive pages
                langchain_docs = self._deduplicate_pdf_content(langchain_docs)

                # Log the processing
                total_chars = sum(len(doc.page_content) for doc in langchain_docs)
                total_words = sum(len(doc.page_content.split()) for doc in langchain_docs)

                self.token_logger.log_aggregated_tokens(
                    model="surya_ocr",
                    operation="pdf_processing_complete",
                    total_input_tokens=0,
                    total_output_tokens=0,
                    meta_json={
                        "file_name": file_name,
                        "file_id": file_id,
                        "total_chunks": len(langchain_docs),
                        "extraction_method": "surya_ocr_only",
                        "total_characters": total_chars,
                        "total_words": total_words
                    }
                )

                return langchain_docs
            else:
                print(f"No content extracted from PDF {file_name} using Surya OCR")
                return []

        except Exception as e:
            logger.error(f"Surya OCR failed for PDF {file_name}: {str(e)}")
            return []

    def load_text(self, file_bytes: bytes, file_id: str, file_name: str):
        """Load and chunk text files (txt, md)"""
        try:
            # Decode bytes to text
            text = file_bytes.decode('utf-8')

            # Split text into chunks
            chunks = self.text_splitter.split_text(text)

            langchain_docs = []
            for idx, chunk in enumerate(chunks):
                detected_lang = self.detect_languages(chunk)
                doc = LangChainDocument(
                    page_content=chunk,
                    metadata={
                        'source': file_id,
                        'chunk_id': str(uuid.uuid4()),
                        'page': 1,  # Text files are single page
                        'file_name': file_name,
                        'language': detected_lang,
                    }
                )
                langchain_docs.append(doc)

            return langchain_docs

        except Exception as e:
            print(f"Error loading text file: {str(e)}")
            return []

    def load_word(self, file_bytes: bytes, file_id: str, file_name: str):
        """Load and chunk Word documents using the single DOCX parser path (no fallback)."""
        try:
            from .extraction.docx_extraction import extract_word_to_langchain_docs

            langchain_docs = extract_word_to_langchain_docs(
                file_bytes=file_bytes,
                file_id=file_id,
                file_name=file_name,
            )

            if not langchain_docs:
                logger.warning(f"No content extracted from Word document {file_name}")
                return []

            print(
                f"Successfully processed Word document {file_name} "
                f"using deterministic DOCX parser: {len(langchain_docs)} chunks"
            )
            return langchain_docs
        except Exception as e:
            logger.error(f"Word extraction failed for {file_name}: {e}")
            return []

    def extract_pdf_pages_base64(self, pdf_content, page_numbers):
        """
        Extract specific pages from a PDF
        Args:
            pdf_content: Either a file path (str) or base64-encoded PDF content (str)
            page_numbers: List of page numbers to extract
        """
        try:
            # Check if pdf_content is base64 or file path
            if os.path.isfile(pdf_content):
                # It's a file path
                reader = PdfReader(pdf_content)
            else:
                # It's base64 content, decode it first
                pdf_bytes = base64.b64decode(pdf_content)
                reader = PdfReader(io.BytesIO(pdf_bytes))

            total_pages = len(reader.pages)
            if not total_pages:
                raise ValueError("Source PDF has no pages")

            valid_pages = [p for p in sorted(set(page_numbers)) if 1 <= p <= total_pages]
            if not valid_pages:
                raise ValueError(f"No valid pages found in {page_numbers}")

            writer = PdfWriter()
            for p in valid_pages:
                writer.add_page(reader.pages[p - 1])

            buffer = io.BytesIO()
            writer.write(buffer)
            file_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return file_base64

        except Exception as e:
            print(f"Error in extract_pdf_pages_base64: {str(e)}")
            return None

    def clean_text(self, text, keep_languages="khmer+english"):
        """Filters recognized text to keep only Khmer + English"""
        if keep_languages == "khmer+english":
            pattern = re.compile(r'[\u1780-\u17FFa-zA-Z0-9\s.!?;:()\'%-,""""]+')
            matches = pattern.findall(text)
            return "".join(matches).strip()
        return text.strip()

    def chunk_text(self, text: str, chunk_size: int = 1024, overlap: int = 204):
        """Smart text chunking that preserves sentence boundaries"""
        # Split by sentences first (handles both English and Khmer)
        sentences = re.split(r'(?<=[។ .!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def semantic_chunk_text(self, text: str, file_id: str, file_name: str, page_num: int = 1, use_structure: bool = True):
        """
        Advanced semantic chunking that preserves document structure and semantic meaning.

        Features:
        - Semantic similarity-based chunking
        - Document structure preservation (headings, sections)
        - Table relationship preservation
        - Multi-modal content handling
        - Language-aware processing (Khmer + English)

        Args:
            text: Raw text content to chunk
            file_id: Document identifier
            file_name: Original file name
            page_num: Page number for citation
            use_structure: Whether to use structural analysis

        Returns:
            List of LangChain documents with semantic chunks
        """
        try:
            if not text or not text.strip():
                logger.warning(f"Empty text provided for semantic chunking: {file_name}")
                return []

            # Preprocessing: Clean and normalize text
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                logger.warning(f"Text cleaning resulted in empty content: {file_name}")
                return []

            langchain_docs = []

            # Initialize semantic chunker if needed
            if use_structure:
                self._initialize_semantic_chunker()

            if use_structure and self.semantic_chunker:
                try:
                    # Attempt semantic chunking with structure preservation
                    logger.info(f"Applying semantic chunking to {file_name} (page {page_num})")

                    # First, try to identify document structure
                    structured_chunks = self._analyze_document_structure(cleaned_text, file_id, file_name, page_num)

                    if structured_chunks:
                        # Use structure-aware chunking
                        logger.info(f"Using structure-aware chunking: {len(structured_chunks)} structured sections")
                        langchain_docs.extend(structured_chunks)
                    else:
                        # Fall back to pure semantic chunking
                        logger.info("Structure analysis failed, using pure semantic chunking")
                        semantic_docs = self._apply_semantic_chunking(cleaned_text, file_id, file_name, page_num)
                        langchain_docs.extend(semantic_docs)

                except Exception as e:
                    logger.warning(f"Semantic chunking failed: {e}, falling back to traditional chunking")
                    # Fallback to traditional recursive chunking
                    traditional_docs = self._apply_traditional_chunking(cleaned_text, file_id, file_name, page_num)
                    langchain_docs.extend(traditional_docs)
            else:
                # Use traditional chunking if semantic chunker not available
                logger.info(f"Using traditional chunking for {file_name} (semantic chunker unavailable)")
                traditional_docs = self._apply_traditional_chunking(cleaned_text, file_id, file_name, page_num)
                langchain_docs.extend(traditional_docs)

            logger.info(f"Semantic chunking completed: {len(langchain_docs)} chunks from {len(cleaned_text)} characters")
            return langchain_docs

        except Exception as e:
            logger.error(f"Critical error in semantic chunking: {e}")
            # Ultimate fallback
            return self._apply_traditional_chunking(text, file_id, file_name, page_num)

    def _analyze_document_structure(self, text: str, file_id: str, file_name: str, page_num: int):
        """
        Analyze document structure to identify headings, sections, tables, and other structural elements.

        Returns structured chunks that preserve document hierarchy and relationships.
        """
        try:
            chunks = []

            # Split text into paragraphs first
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

            if not paragraphs:
                return []

            # Identify structural elements
            structured_sections = self._identify_structural_elements(paragraphs)

            # Group related content
            current_section = {
                'title': '',
                'content': [],
                'type': 'body',
                'level': 0
            }

            for item in structured_sections:
                if item['type'] in ['heading', 'title']:
                    # Save previous section if it has content
                    if current_section['content']:
                        chunk_doc = self._create_structured_chunk(
                            current_section, file_id, file_name, page_num, len(chunks)
                        )
                        if chunk_doc:
                            chunks.append(chunk_doc)

                    # Start new section
                    current_section = {
                        'title': item['text'],
                        'content': [],
                        'type': item['type'],
                        'level': item.get('level', 1)
                    }
                else:
                    # Add content to current section
                    current_section['content'].append(item['text'])

            # Don't forget the last section
            if current_section['content']:
                chunk_doc = self._create_structured_chunk(
                    current_section, file_id, file_name, page_num, len(chunks)
                )
                if chunk_doc:
                    chunks.append(chunk_doc)

            return chunks if chunks else None

        except Exception as e:
            logger.warning(f"Document structure analysis failed: {e}")
            return None

    def _identify_structural_elements(self, paragraphs):
        """
        Identify headings, titles, lists, tables, and other structural elements in text.
        """
        elements = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Detect headings (common patterns)
            if self._is_heading(para):
                elements.append({
                    'type': 'heading',
                    'text': para,
                    'level': self._get_heading_level(para)
                })
            # Detect lists
            elif self._is_list_item(para):
                elements.append({
                    'type': 'list_item',
                    'text': para
                })
            # Detect tables (simple pattern matching)
            elif self._is_table_content(para):
                elements.append({
                    'type': 'table',
                    'text': para
                })
            else:
                elements.append({
                    'type': 'paragraph',
                    'text': para
                })

        return elements

    def _is_heading(self, text: str) -> bool:
        """Detect if text is likely a heading."""
        # Common heading patterns
        if re.match(r'^#{1,6}\s+', text):  # Markdown headings
            return True
        if len(text) < 100 and not text.endswith('.'):  # Short, no period
            return True
        if text.isupper():  # All caps
            return True
        if re.match(r'^\d+\.?\s+', text):  # Numbered sections
            return True
        if any(text.startswith(prefix) for prefix in ['Chapter', 'Section', 'Article', 'Part']):
            return True
        return False

    def _get_heading_level(self, text: str) -> int:
        """Determine heading level (1-6)."""
        if re.match(r'^#{1,6}\s+', text):
            return len(re.match(r'^(#+)', text).group(1))
        return 1

    def _is_list_item(self, text: str) -> bool:
        """Detect if text is a list item."""
        # Common list patterns
        if re.match(r'^[-\*\+]\s+', text):  # Bullet points
            return True
        if re.match(r'^\d+\.\s+', text):  # Numbered lists
            return True
        if re.match(r'^[a-zA-Z]\.\s+', text):  # Lettered lists
            return True
        return False

    def _is_table_content(self, text: str) -> bool:
        """Detect if text contains table-like content."""
        # Look for pipe separators (markdown tables) or multiple separators
        if '|' in text and len(text.split('|')) > 2:
            return True
        # Look for multiple tabs or consistent spacing
        if '\t' in text and len(text.split('\t')) > 2:
            return True
        return False

    def _create_structured_chunk(self, section: dict, file_id: str, file_name: str, page_num: int, chunk_idx: int):
        """Create a LangChain document from a structured section."""
        try:
            # Combine title and content
            if section['title']:
                chunk_content = f"{section['title']}\n\n{chr(10).join(section['content'])}"
            else:
                chunk_content = chr(10).join(section['content'])

            # Detect language
            detected_lang = self.detect_languages(chunk_content)

            # Create metadata with structural information
            metadata = {
                'source': file_id,
                'chunk_id': str(uuid.uuid4()),
                'page': page_num,
                'file_name': file_name,
                'language': detected_lang,
                'chunk_type': 'semantic_structured',
                'section_title': section.get('title', ''),
                'section_type': section.get('type', 'body'),
                'section_level': section.get('level', 0),
                'chunk_index': chunk_idx,
                'extraction_method': 'semantic_chunking'
            }

            return LangChainDocument(
                page_content=chunk_content,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error creating structured chunk: {e}")
            return None

    def _apply_semantic_chunking(self, text: str, file_id: str, file_name: str, page_num: int):
        """Apply pure semantic chunking without structural analysis."""
        try:
            # Initialize if not already done
            if not self._semantic_chunker_initialized:
                self._initialize_semantic_chunker()

            if not self.semantic_chunker:
                raise Exception("Semantic chunker not available")

            # Create a single document for semantic chunking
            doc = LangChainDocument(page_content=text)

            # Apply semantic chunking
            semantic_chunks = self.semantic_chunker.split_documents([doc])

            langchain_docs = []
            for idx, chunk in enumerate(semantic_chunks):
                detected_lang = self.detect_languages(chunk.page_content)

                # Create new document with enhanced metadata
                metadata = {
                    'source': file_id,
                    'chunk_id': str(uuid.uuid4()),
                    'page': page_num,
                    'file_name': file_name,
                    'language': detected_lang,
                    'chunk_type': 'semantic_similarity',
                    'chunk_index': idx,
                    'extraction_method': 'semantic_chunking'
                }

                langchain_doc = LangChainDocument(
                    page_content=chunk.page_content,
                    metadata=metadata
                )
                langchain_docs.append(langchain_doc)

            logger.info(f"Semantic chunking produced {len(langchain_docs)} chunks")
            return langchain_docs

        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}")
            raise

    def _apply_traditional_chunking(self, text: str, file_id: str, file_name: str, page_num: int):
        """Fallback to traditional recursive chunking."""
        try:
            # Use the existing text splitter
            chunks = self.text_splitter.split_text(text)

            langchain_docs = []
            for idx, chunk_text in enumerate(chunks):
                detected_lang = self.detect_languages(chunk_text)

                metadata = {
                    'source': file_id,
                    'chunk_id': str(uuid.uuid4()),
                    'page': page_num,
                    'file_name': file_name,
                    'language': detected_lang,
                    'chunk_type': 'traditional_recursive',
                    'chunk_index': idx,
                    'extraction_method': 'traditional_chunking'
                }

                doc = LangChainDocument(
                    page_content=chunk_text,
                    metadata=metadata
                )
                langchain_docs.append(doc)

            return langchain_docs

        except Exception as e:
            logger.error(f"Traditional chunking failed: {e}")
            # Create single chunk as last resort
            detected_lang = self.detect_languages(text)
            return [LangChainDocument(
                page_content=text,
                metadata={
                    'source': file_id,
                    'chunk_id': str(uuid.uuid4()),
                    'page': page_num,
                    'file_name': file_name,
                    'language': detected_lang,
                    'chunk_type': 'fallback_single',
                    'extraction_method': 'fallback_chunking'
                }
            )]

    def detect_languages(self, text:str):
        """Detect primary script for input text using a single deterministic path."""
        cleaned_text = re.sub(r"[^\w\s]", " ", text).strip()
        if not cleaned_text:
            return "en"

        khmer_count = sum(1 for char in cleaned_text if "\u1780" <= char <= "\u17FF")
        latin_count = sum(
            1 for char in cleaned_text if ("a" <= char.lower() <= "z") or char.isdigit()
        )

        if khmer_count and latin_count:
            total = khmer_count + latin_count
            khmer_ratio = khmer_count / total
            latin_ratio = latin_count / total
            if khmer_ratio >= 0.15 and latin_ratio >= 0.15:
                return "multi"

        if khmer_count > 0:
            return "km"

        return "en"

    def get_pdf_page_count(self,pdf_path):
        reader = PdfReader(pdf_path)
        return len(reader.pages)

    def get_pdf_page_count_from_base64(self, pdf_base64: str):
        import base64
        from io import BytesIO
        if pdf_base64.startswith("data:"):
            pdf_base64 = pdf_base64.split(",", 1)[1]

        # Decode
        try:
            pdf_bytes = base64.b64decode(pdf_base64)
        except Exception as e:
            raise ValueError(f"Base64 decode failed: {e}")

        # Verify PDF signature
        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError("Decoded file is not a valid PDF")

        # Try to read
        pdf_file = BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        return len(reader.pages)



    def load_presentation(self, file_bytes: bytes, file_id: str, file_name: str):
        """Load and chunk PowerPoint presentation files (PPTX, PPT)"""
        try:
            # Create a temporary file to work with PPTX
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pptx') as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            try:
                # Load the presentation
                presentation = Presentation(temp_file_path)

                # Extract text from all slides
                text_content = []
                slide_num = 1
                for slide in presentation.slides:
                    slide_text = []
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            slide_text.append(shape.text)
                    if slide_text:
                        text_content.append(f"Slide {slide_num}: {' '.join(slide_text)}")
                        slide_num += 1

                # Join all text
                full_text = '\n\n'.join(text_content)

                if not full_text.strip():
                    print(f"No content found in presentation file: {file_name}")
                    return []

                # Split text into chunks
                chunks = self.text_splitter.split_text(full_text)

                langchain_docs = []
                for idx, chunk in enumerate(chunks):
                    detected_lang = self.detect_languages(chunk)
                    doc = LangChainDocument(
                        page_content=chunk,
                        metadata={
                            'source': file_id,
                            'chunk_id': str(uuid.uuid4()),
                            'page': 1,  # Presentations don't have traditional pages
                            'file_name': file_name,
                            'language': detected_lang,
                        }
                    )
                    langchain_docs.append(doc)

                return langchain_docs

            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

        except Exception as e:
            print(f"Error loading presentation file: {str(e)}")
            return []

    def load_spreadsheet(self, file_bytes: bytes, file_id: str, file_name: str):
        """Load and chunk Excel spreadsheet files (XLSX, XLS)"""
        try:
            # Create a temporary file to work with Excel
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            try:
                # Try to load as XLSX first, then XLS
                try:
                    workbook = openpyxl.load_workbook(temp_file_path, data_only=True)
                    is_xlsx = True
                except:
                    # Fallback to XLS format
                    workbook = xlrd.open_workbook(temp_file_path)
                    is_xlsx = False

                # Extract text from all sheets
                text_content = []

                if is_xlsx:
                    for sheet_name in workbook.sheetnames:
                        sheet = workbook[sheet_name]
                        sheet_text = [f"Sheet: {sheet_name}"]
                        for row in sheet.iter_rows(values_only=True):
                            row_text = [str(cell) for cell in row if cell is not None]
                            if row_text:
                                sheet_text.append(' | '.join(row_text))
                        text_content.append('\n'.join(sheet_text))
                else:
                    for sheet_idx in range(workbook.nsheets):
                        sheet = workbook.sheet_by_index(sheet_idx)
                        sheet_text = [f"Sheet: {sheet.name}"]
                        for row_idx in range(sheet.nrows):
                            row = sheet.row(row_idx)
                            row_text = [str(cell.value) for cell in row if cell.value is not None]
                            if row_text:
                                sheet_text.append(' | '.join(row_text))
                        text_content.append('\n'.join(sheet_text))

                # Join all text
                full_text = '\n\n'.join(text_content)

                if not full_text.strip():
                    print(f"No content found in spreadsheet file: {file_name}")
                    return []

                # Split text into chunks
                chunks = self.text_splitter.split_text(full_text)

                langchain_docs = []
                for idx, chunk in enumerate(chunks):
                    detected_lang = self.detect_languages(chunk)
                    doc = LangChainDocument(
                        page_content=chunk,
                        metadata={
                            'source': file_id,
                            'chunk_id': str(uuid.uuid4()),
                            'page': 1,
                            'file_name': file_name,
                            'language': detected_lang,
                        }
                    )
                    langchain_docs.append(doc)

                return langchain_docs

            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

        except Exception as e:
            print(f"Error loading spreadsheet file: {str(e)}")
            return []

    def load_csv_tsv(self, file_bytes: bytes, file_id: str, file_name: str, delimiter: str = ','):
        """Load and chunk CSV/TSV files"""
        try:
            # Decode bytes to text
            text = file_bytes.decode('utf-8', errors='ignore')

            # Parse CSV/TSV using pandas
            df = pd.read_csv(io.StringIO(text), delimiter=delimiter, engine='python')

            # Convert DataFrame to text representation
            text_content = [f"CSV File: {file_name}"]
            text_content.append("Headers: " + ', '.join(df.columns.tolist()))

            # Add each row as text
            for idx, row in df.iterrows():
                row_text = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
                if row_text:
                    text_content.append(f"Row {idx + 1}: {' | '.join(row_text)}")

            full_text = '\n'.join(text_content)

            if not full_text.strip():
                print(f"No content found in CSV/TSV file: {file_name}")
                return []

            # Split text into chunks
            chunks = self.text_splitter.split_text(full_text)

            langchain_docs = []
            for idx, chunk in enumerate(chunks):
                detected_lang = self.detect_languages(chunk)
                doc = LangChainDocument(
                    page_content=chunk,
                    metadata={
                        'source': file_id,
                        'chunk_id': str(uuid.uuid4()),
                        'page': 1,
                        'file_name': file_name,
                        'language': detected_lang,
                    }
                )
                langchain_docs.append(doc)

            return langchain_docs

        except Exception as e:
            print(f"Error loading CSV/TSV file: {str(e)}")
            return []



    def load_image(self, file_bytes: bytes, file_id: str, file_name: str):
        """Image processing not available - Gemini removed"""
        print(f"Image processing not supported: {file_name} (Gemini removed)")
        return []
