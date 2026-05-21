import os
import logging
import requests
from pathlib import Path
from typing import Optional

from django.conf import settings

logger = logging.getLogger(__name__)


class DocumentConverter:
    """Convert DOCX documents to PDF using conversion API"""

    # Formats that can be converted to PDF
    CONVERTIBLE_FORMATS = {
        'docx',  # Word DOCX
        'doc',   # Word DOC
        'rtf',   # Rich Text Format
        'odt',   # OpenDocument Text
    }

    def __init__(self):
        pass

    def is_convertible(self, file_format: str) -> bool:
        """Check if file format can be converted to PDF"""
        return file_format.lower() in self.CONVERTIBLE_FORMATS

    def convert_to_pdf(self, input_path: str, output_dir: str = None) -> Optional[str]:
        """
        Convert document to PDF using conversion API

        Args:
            input_path: Path to the input document
            output_dir: Directory for output PDF (defaults to same as input)

        Returns:
            Path to generated PDF file, or None if conversion failed
        """
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return None

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(input_path)

        os.makedirs(output_dir, exist_ok=True)

        try:
            logger.info(f"Converting {input_path} to PDF using conversion API...")

            # Use the conversion API
            with open(input_path, "rb") as f:
                response = requests.post(
                    settings.DOC_CONVERTER_URL,
                    files={"file": f}
                )

            if response.status_code == 200:
                # Determine output PDF path
                input_filename = os.path.basename(input_path)
                pdf_filename = os.path.splitext(input_filename)[0] + '.pdf'
                pdf_path = os.path.join(output_dir, pdf_filename)

                # Save the PDF content to the output path
                with open(pdf_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"✓ PDF created: {pdf_path}")
                return pdf_path
            else:
                logger.error(f"API conversion failed with status {response.status_code}: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during PDF conversion: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error converting {input_path} to PDF: {e}")
            return None
    
    def get_pdf_path_for_document(self, original_path: str) -> str:
        """
        Get the expected PDF path for a document
        
        Args:
            original_path: Path to original document
            
        Returns:
            Expected path for converted PDF
        """
        # Replace extension with .pdf
        base_path = os.path.splitext(original_path)[0]
        return f"{base_path}_preview.pdf"


# Singleton instance
_converter_instance = None

def get_converter() -> DocumentConverter:
    """Get global DocumentConverter instance"""
    global _converter_instance
    if _converter_instance is None:
        _converter_instance = DocumentConverter()
    return _converter_instance







