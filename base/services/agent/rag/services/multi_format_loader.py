"""
Multi-Format Document Loader Service for RAG System
Supports: PDF, DOC/DOCX, PPT/PPTX, XLS/XLSX, TXT, MD, CSV, TSV, Images
Note: Only formats with kept dependencies are supported
"""

import os
import io
import logging
import csv
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiFormatDocumentLoader:
    """Universal document loader supporting multiple file formats"""

    # Supported file extensions by category (only formats with kept dependencies)
    SUPPORTED_FORMATS = {
        'pdf': ['.pdf'],
        'word': ['.docx'],  # Only .docx (python-docx), .doc removed (mammoth dependency)
        'powerpoint': ['.ppt', '.pptx'],
        'excel': ['.xls', '.xlsx'],
        'text': ['.txt', '.md'],  # Removed .rtf (striprtf dependency)
        'csv': ['.csv', '.tsv'],
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.tif', '.svg', '.heic', '.heif']
        # Removed: 'odt', 'epub' (dependencies removed)
    }
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=256,
            )
            
            # logger.info("✓ MultiFormatDocumentLoader initialized")
    
    def get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename"""
        return Path(filename).suffix.lower()
    
    def is_supported(self, filename: str) -> bool:
        """Check if file format is supported"""
        ext = self.get_file_extension(filename)
        return any(ext in formats for formats in self.SUPPORTED_FORMATS.values())
    
    def detect_format_category(self, filename: str) -> str:
        """Detect which category the file belongs to"""
        ext = self.get_file_extension(filename)
        for category, extensions in self.SUPPORTED_FORMATS.items():
            if ext in extensions:
                return category
        return 'unknown'
    
    def load_document(self, file_bytes: bytes, file_id: str, file_name: str) -> List[LangChainDocument]:
        """
        Load document from any supported format and return LangChain Document chunks
        
        Args:
            file_bytes: File content as bytes
            file_id: Unique identifier for the document
            file_name: Original filename with extension
            
        Returns:
            List of LangChain Document chunks
        """
        try:
            category = self.detect_format_category(file_name)
            logger.info(f"Loading {file_name} (category: {category}, {len(file_bytes)} bytes)")
            
            if category == 'pdf':
                return self._load_pdf(file_bytes, file_id, file_name)
            elif category == 'word':
                return self._load_word(file_bytes, file_id, file_name)
            elif category == 'powerpoint':
                return self._load_powerpoint(file_bytes, file_id, file_name)
            elif category == 'excel':
                return self._load_excel(file_bytes, file_id, file_name)
            elif category == 'text':
                return self._load_text(file_bytes, file_id, file_name)
            elif category == 'csv':
                return self._load_csv(file_bytes, file_id, file_name)
            elif category == 'image':
                return self._load_image(file_bytes, file_id, file_name)
            else:
                logger.error(f"Unsupported file format: {file_name}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading document {file_name}: {e}")
            return []
    
    def _load_pdf(self, file_bytes: bytes, file_id: str, file_name: str) -> List[LangChainDocument]:
        """Load PDF - currently disabled in this loader"""
        try:
            logger.warning(f"PDF processing is disabled in MultiFormatDocumentLoader. Use DocumentLoaderService instead for PDF: {file_name}")
            return []
        except Exception as e:
            logger.error(f"PDF loading error: {e}")
            return []
    
    def _load_word(self, file_bytes: bytes, file_id: str, file_name: str) -> List[LangChainDocument]:
        """Load DOCX files (only .docx supported, .doc requires mammoth which was removed)"""
        try:
            from docx import Document as DocxDocument

            ext = self.get_file_extension(file_name)

            if ext == '.docx':
                # Use python-docx for DOCX
                doc = DocxDocument(io.BytesIO(file_bytes))
                text = '\n\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            else:
                logger.error(f"Unsupported Word format: {ext}. Only .docx is supported (mammoth for .doc was removed)")
                return []

            return self._create_chunks(text, file_id, file_name)

        except ImportError:
            logger.error("python-docx not installed. Run: pip install python-docx")
            return []
        except Exception as e:
            logger.error(f"Word document loading error: {e}")
            return []
    
    def _load_text(self, file_bytes: bytes, file_id: str, file_name: str) -> List[LangChainDocument]:
        """Load TXT and MD files (RTF support removed - striprtf dependency was removed)"""
        try:
            ext = self.get_file_extension(file_name)

            if ext == '.rtf':
                logger.error(f"RTF files are not supported. striprtf dependency was removed: {file_name}")
                return []
            else:
                # TXT and MD files
                text = file_bytes.decode('utf-8', errors='ignore')

            return self._create_chunks(text, file_id, file_name)

        except Exception as e:
            logger.error(f"Text file loading error: {e}")
            return []
    
    def _load_powerpoint(self, file_bytes: bytes, file_id: str, file_name: str) -> List[LangChainDocument]:
        """Load PPT/PPTX files"""
        try:
            from pptx import Presentation
            
            prs = Presentation(io.BytesIO(file_bytes))
            text_content = []
            
            for slide_num, slide in enumerate(prs.slides, start=1):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text.append(shape.text)
                
                if slide_text:
                    text_content.append(f"[Slide {slide_num}]\n" + '\n'.join(slide_text))
            
            full_text = '\n\n'.join(text_content)
            return self._create_chunks(full_text, file_id, file_name)
            
        except ImportError:
            logger.error("python-pptx not installed. Run: pip install python-pptx")
            return []
        except Exception as e:
            logger.error(f"PowerPoint loading error: {e}")
            return []
    
    def _load_excel(self, file_bytes: bytes, file_id: str, file_name: str) -> List[LangChainDocument]:
        """Load XLS/XLSX files"""
        try:
            import pandas as pd
            
            # Read all sheets
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
            text_content = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                # Convert dataframe to markdown-like format
                sheet_text = f"[Sheet: {sheet_name}]\n"
                sheet_text += df.to_string(index=False, na_rep='')
                text_content.append(sheet_text)
            
            full_text = '\n\n'.join(text_content)
            return self._create_chunks(full_text, file_id, file_name)
            
        except ImportError:
            logger.error("pandas and openpyxl not installed. Run: pip install pandas openpyxl xlrd")
            return []
        except Exception as e:
            logger.error(f"Excel loading error: {e}")
            return []
    
    def _load_csv(self, file_bytes: bytes, file_id: str, file_name: str) -> List[LangChainDocument]:
        """Load CSV/TSV files"""
        try:
            import pandas as pd
            
            ext = self.get_file_extension(file_name)
            separator = '\t' if ext == '.tsv' else ','
            
            df = pd.read_csv(io.BytesIO(file_bytes), sep=separator, encoding='utf-8', on_bad_lines='skip')
            text_content = df.to_string(index=False, na_rep='')
            
            return self._create_chunks(text_content, file_id, file_name)
            
        except ImportError:
            logger.error("pandas not installed. Run: pip install pandas")
            return []
        except Exception as e:
            logger.error(f"CSV loading error: {e}")
            return []

    def _load_image(self, file_bytes: bytes, file_id: str, file_name: str) -> List[LangChainDocument]:
        """Load images - currently disabled in this loader"""
        try:
            logger.warning(f"Image processing is disabled in MultiFormatDocumentLoader. Use DocumentLoaderService instead for image: {file_name}")
            return []

        except Exception as e:
            logger.error(f"Image loading error: {e}")
            return []

    def _create_chunks(self, text: str, file_id: str, file_name: str) -> List[LangChainDocument]:
        """Create LangChain document chunks from text"""
        try:
            if not text or not text.strip():
                logger.warning(f"No text content found in {file_name}")
                return []
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            documents = []
            
            for idx, chunk_text in enumerate(chunks):
                doc = LangChainDocument(
                    page_content=chunk_text,
                    metadata={
                        'file_id': file_id,
                        'file_name': file_name,
                        'page_number': 1,  # Non-PDF documents treated as single page
                        'chunk_id': str(uuid.uuid4()),
                        'source': file_name,
                        'extraction_method': 'text_extraction'
                    }
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} chunks from {file_name}")
            return documents
            
        except Exception as e:
            logger.error(f"Chunk creation error: {e}")
            return []







