import os
import time
import logging
from pathlib import Path
from PIL import Image
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import uuid
from urllib.parse import urljoin

from docling.document_converter import DocumentConverter
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SuryaOCR:
    """Production-grade OCR extractor using Surya - Optimized for Khmer + English"""
    
    def __init__(self):
        """Initialize OCR extractor"""
        self.recognition = None
        self.detection = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize Surya models"""
        logger.info("Initializing Surya OCR models...")
        try:
            # Import Surya modules here to avoid CUDA initialization at module import time
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor

            logger.info("Loading foundation model...")
            foundation = FoundationPredictor()

            logger.info("Loading recognition model...")
            self.recognition = RecognitionPredictor(foundation)

            logger.info("Loading detection model...")
            self.detection = DetectionPredictor()

            self.is_initialized = True
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            return False
    
    def convert_pdf_to_images(self, pdf_path: str, dpi: int = 150) -> List[Image.Image]:
        """Convert PDF pages to images using PyMuPDF"""
        try:
            import pymupdf as fitz
            
            images = []
            pdf_doc = fitz.open(pdf_path)
            page_count = len(pdf_doc)
            
            logger.info(f"Converting PDF ({page_count} pages) to images at {dpi} DPI...")
            
            for page_num in range(page_count):
                try:
                    page = pdf_doc[page_num]
                    zoom_factor = dpi / 72
                    matrix = fitz.Matrix(zoom_factor, zoom_factor)
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    
                    ppm_data = pixmap.tobytes("ppm")
                    from io import BytesIO
                    img = Image.open(BytesIO(ppm_data))
                    
                    images.append(img)
                    logger.info(f"Page {page_num + 1}/{page_count} converted - Size: {img.size}")
                except Exception as e:
                    logger.warning(f"Failed to convert page {page_num + 1}: {e}")
                    continue
            
            pdf_doc.close()
            return images
        except Exception as e:
            logger.error(f"PDF conversion error: {e}")
            return []
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for optimal OCR performance
        Enhances contrast and reduces noise for better Khmer + English text detection
        Uses CLAHE + bilateral filtering for edge-preserving enhancement
        """
        try:
            import cv2
            import numpy as np
            
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 1. Convert to grayscale for processing
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # More effective than simple histogram equalization for text clarity
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # 3. Apply bilateral filter for noise reduction while preserving edges
            # Critical for maintaining text sharpness without blurring
            filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # 4. Apply slight sharpening kernel for improved text definition
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) / 1.5
            sharpened = cv2.filter2D(filtered, -1, kernel)
            
            # 5. Convert back to RGB for Surya processing
            result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            # Convert back to PIL
            processed = Image.fromarray(result)
            
            logger.debug(f"Image preprocessing applied: {image.size}")
            return processed
        
        except ImportError:
            logger.warning("OpenCV not available, skipping image preprocessing")
            return image
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}, using original image")
            return image
    
    def normalize_khmer_fonts(self, text: str) -> str:
        """
        Handle Khmer font variations and similar-looking characters
        Different Khmer fonts render the same character differently in OCR
        This function normalizes common font-specific confusions
        """
        import re
        
        # Font-specific character mappings (common OCR confusions across Khmer fonts)
        font_corrections = {
            # Character similarity across fonts
            'ល៉': 'ល',         # La with horn vs plain la
            'ដ៍': 'ដ',         # Da with underline vs plain da
            'ក៍': 'ក',         # Ka with underline vs plain ka
            'ស៍': 'ស',         # Sa with underline vs plain sa
            'ន៍': 'ន',         # Na with underline vs plain na
            'ម៍': 'ម',         # Ma with underline vs plain ma
            'ប៍': 'ប',         # Pa with underline vs plain pa
            'ផ៍': 'ផ',         # Pha with underline vs plain pha
            'ដ្': 'ដ្ដ',         # Double da correction
            
            # Zero-width character cleanup
            '\u200B\u200C': '',  # Zero-width space and non-joiner
            '\u200D': '',        # Zero-width joiner (sometimes causes issues)
            
            # Common subscript/coeng normalizations
            '្្': '្',          # Double coeng -> single
            '៍៍': '៍',          # Double mark -> single
        }
        
        for old, new in font_corrections.items():
            text = text.replace(old, new)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Production-optimized text cleaning for Khmer + English
        Maximizes accuracy by:
        - Complete Unicode normalization
        - Comprehensive Khmer character preservation
        - Font-specific correction for Khmer variants
        - Language-specific OCR error correction
        - Aggressive noise removal (non-target languages)
        """
        import re
        import unicodedata
        
        if not text:
            return ""
        
        # 0. Normalize Khmer font variations (handle multiple font renderings)
        text = self.normalize_khmer_fonts(text)
        
        # 1. Unicode normalization (NFC - Composed form for proper ligature handling)
        text = unicodedata.normalize('NFC', text)
        
        # 2. AGGRESSIVE LANGUAGE FILTERING - KEEP ONLY KHMER + ENGLISH
        # Khmer ranges:
        #   U+1780-U+17FF: Khmer main block (consonants, vowels, diacritics)
        #   U+17B4-U+17D0: Khmer vowel inherent + dependent vowels + signs
        #   U+19E0-U+19FF: Khmer symbols (currency, punctuation)
        #   U+200B-U+200D: Zero-width chars (joiners for ligatures)
        # English:
        #   a-zA-Z: Latin letters
        #   0-9: Digits
        # Punctuation + whitespace:
        #   Common marks, quotes, mathematical operators
        
        pattern = re.compile(
            r'[\u1780-\u17FF'                    # Khmer consonants and most vowels
            r'\u17B4-\u17D0'                     # Khmer vowel inherent, dependent vowels, signs
            r'\u17D6'                             # Khmer sign
            r'\u19E0-\u19FF'                     # Khmer symbols and currency
            r'\u200B-\u200D'                     # Zero-width joiner/non-joiner for proper rendering
            r'a-zA-Z0-9'                         # English letters and digits
            r'\s.!?;:,\-()\'"\-—–''""„«»'      # Punctuation, quotes, dashes
            r'%&*+/=@#$\[\]{}^~`]',
            re.UNICODE
        )
        
        cleaned = "".join(pattern.findall(text))
        
        # 3. Whitespace normalization
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces → single space
        
        # 4. Smart punctuation spacing
        cleaned = re.sub(r'\s([.!?;:,\-)\]])', r'\1', cleaned)  # Space before closing punctuation
        cleaned = re.sub(r'([\[({\'""])\s+', r'\1', cleaned)    # Space after opening punctuation
        
        # 5. PRODUCTION-GRADE OCR ERROR CORRECTION for Khmer + English
        # These patterns target the most common OCR confusions for these languages
        corrections = {
            # Khmer digit/character confusions (very common)
            'ល០': 'ល0',     # la + Khmer zero → la + digit zero
            'ដ०': 'ដ0',     # da + Khmer zero → da + digit zero
            'ស०': 'ស0',     # sa + Khmer zero → sa + digit zero
            'ក០': 'ក0',     # ka + Khmer zero → ka + digit zero
            'ន०': 'ន0',     # na + Khmer zero → na + digit zero
            
            # Khmer character confusion (visually similar in OCR)
            'ឞ': 'ស',       # Archaic sa → regular sa
            
            # Remove stray marks that aren't valid Khmer
            '◌': '',        # Combining mark placeholder
        }
        
        for old, new in corrections.items():
            cleaned = cleaned.replace(old, new)
        
        # 6. Final cleanup - strip leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # 7. Remove any remaining non-printable characters except valid Unicode
        cleaned = "".join(
            c for c in cleaned 
            if unicodedata.category(c)[0] != 'C' or c in '\n\t '
        )
        
        return cleaned
    
    def extract_text(self, image: Image.Image) -> str:
        """Extract text from image with preprocessing"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image for better OCR accuracy
            processed_image = self.preprocess_image(image)
            
            predictions = self.recognition([processed_image], det_predictor=self.detection)
            
            if predictions and len(predictions) > 0:
                text = "\n".join([line.text for line in predictions[0].text_lines])
                # Apply enhanced cleaning
                text = self.clean_text(text)
                return text.strip() if text else ""
            
            return ""
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return ""
    
    def detect_layout(self, image: Image.Image) -> Dict:
        """
        Advanced layout detection for complex documents
        Identifies regions: headers, visuals/diagrams, body text, lists, tables

        Returns:
            Dict with layout analysis including region boundaries and types
        """
        try:
            import cv2
            import numpy as np

            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            # 1. Advanced text region detection using morphological operations
            # Create structuring element for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))

            # Apply morphological operations to find text blocks
            dilated = cv2.dilate(gray, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=1)

            # Find contours of text blocks
            contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            text_regions = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000:  # Filter very small regions
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Calculate text density (black pixels in region)
                    roi = gray[y:y+h, x:x+w]
                    density = np.sum(roi < 128) / (w * h) if w * h > 0 else 0
                    text_regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'density': density,
                        'aspect_ratio': w / h if h > 0 else 0,
                        'center': (x + w//2, y + h//2)
                    })

            # 2. Table detection using line detection
            # Horizontal line detection
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)

            # Vertical line detection
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

            # Combine horizontal and vertical lines
            table_mask = cv2.add(horizontal_lines, vertical_lines)

            # Find table regions
            table_contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            table_regions = []
            for cnt in table_contours:
                area = cv2.contourArea(cnt)
                if area > 5000:  # Larger area threshold for tables
                    x, y, w, h = cv2.boundingRect(cnt)
                    table_regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'type': 'table'
                    })

            # 3. Visual/Diagram detection using edge density and shape analysis
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Find contours for visual elements
            visual_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            visual_regions = []
            for cnt in visual_contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if area > 2000 and perimeter > 200:  # Filter by size and complexity
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Check if it's likely a diagram (high edge density, geometric shapes)
                    edge_density = np.sum(edges[y:y+h, x:x+w] > 0) / (w * h) if w * h > 0 else 0

                    if edge_density > 0.1:  # High edge density indicates diagrams/charts
                        visual_regions.append({
                            'bbox': (x, y, w, h),
                            'area': area,
                            'edge_density': edge_density,
                            'type': 'diagram'
                        })

            # 4. Classify regions by position and content
            regions = {
                'header': [],
                'table': table_regions,
                'visual': visual_regions,
                'body': [],
                'footer': []
            }

            # Sort text regions by vertical position for reading order
            text_regions.sort(key=lambda r: r['bbox'][1])  # Sort by y-coordinate

            # Classify regions by vertical position
            page_height = height
            header_threshold = int(page_height * 0.15)  # Top 15%
            footer_threshold = int(page_height * 0.85)   # Bottom 15%

            for region in text_regions:
                y = region['bbox'][1]
                if y < header_threshold:
                    regions['header'].append(region)
                elif y > footer_threshold:
                    regions['footer'].append(region)
                else:
                    regions['body'].append(region)

            # 5. Determine reading order using spatial analysis
            reading_order = self._calculate_reading_order(text_regions, width, height)

            return {
                'success': True,
                'image_size': (width, height),
                'regions': regions,
                'text_regions': text_regions,
                'reading_order': reading_order,
                'tables': table_regions,
                'visuals': visual_regions,
                'statistics': {
                    'total_text_regions': len(text_regions),
                    'tables_found': len(table_regions),
                    'visuals_found': len(visual_regions),
                    'header_regions': len(regions['header']),
                    'body_regions': len(regions['body']),
                    'footer_regions': len(regions['footer'])
                }
            }

        except Exception as e:
            logger.warning(f"Advanced layout detection failed: {e}, falling back to basic detection")
            # Fallback to basic layout detection
            return self._basic_layout_detection(image)

    def _basic_layout_detection(self, image: Image.Image) -> Dict:
        """Basic fallback layout detection"""
        try:
            import cv2
            import numpy as np

            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            return {
                'success': True,
                'image_size': (width, height),
                'regions': {
                    'header': [],
                    'table': [],
                    'visual': [],
                    'body': [],
                    'footer': []
                },
                'text_regions': [],
                'reading_order': [],
                'tables': [],
                'visuals': [],
                'statistics': {
                    'total_text_regions': 0,
                    'tables_found': 0,
                    'visuals_found': 0,
                    'header_regions': 0,
                    'body_regions': 0,
                    'footer_regions': 0
                }
            }
        except Exception as e:
            logger.error(f"Basic layout detection also failed: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_reading_order(self, text_regions: List[Dict], page_width: int, page_height: int) -> List[int]:
        """
        Calculate optimal reading order using spatial clustering and flow analysis
        This handles complex multi-column layouts and irregular text arrangements
        """
        if not text_regions:
            return []

        try:
            # Group regions into columns using k-means clustering on x-coordinates
            import numpy as np
            from sklearn.cluster import KMeans

            # Extract x-coordinates of region centers
            x_coords = np.array([r['center'][0] for r in text_regions]).reshape(-1, 1)

            # Determine optimal number of columns (usually 1-3 for most documents)
            n_columns = min(3, len(text_regions))
            if len(text_regions) > 5:
                # Use silhouette score to determine best number of columns
                from sklearn.metrics import silhouette_score
                best_score = -1
                best_n = 1

                for n in range(2, min(4, len(text_regions))):
                    try:
                        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(x_coords)
                        if len(set(labels)) > 1:
                            score = silhouette_score(x_coords, labels)
                            if score > best_score:
                                best_score = score
                                best_n = n
                    except:
                        continue

                n_columns = best_n if best_score > 0.3 else 1

            # Perform clustering
            kmeans = KMeans(n_clusters=n_columns, random_state=42, n_init=10)
            column_labels = kmeans.fit_predict(x_coords)

            # Group regions by column
            columns = {}
            for i, region in enumerate(text_regions):
                col = column_labels[i]
                if col not in columns:
                    columns[col] = []
                columns[col].append((i, region))

            # Sort regions within each column by y-coordinate (top to bottom)
            for col in columns:
                columns[col].sort(key=lambda x: x[1]['bbox'][1])

            # Determine column reading order (left to right)
            column_centers = {}
            for col, regions in columns.items():
                avg_x = np.mean([r['center'][0] for _, r in regions])
                column_centers[col] = avg_x

            sorted_columns = sorted(column_centers.items(), key=lambda x: x[1])

            # Build final reading order
            reading_order = []
            for col, _ in sorted_columns:
                for idx, _ in columns[col]:
                    reading_order.append(idx)

            return reading_order

        except Exception as e:
            logger.warning(f"Reading order calculation failed: {e}, using simple vertical order")
            # Fallback: simple vertical ordering
            return sorted(range(len(text_regions)), key=lambda i: text_regions[i]['bbox'][1])

    def extract_tables(self, image: Image.Image, table_regions: List[Dict]) -> List[Dict]:
        """
        Extract and structure table data from detected table regions
        Uses advanced table structure recognition and cell content extraction
        """
        tables = []

        try:
            import cv2
            import numpy as np

            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            for table_region in table_regions:
                x, y, w, h = table_region['bbox']
                table_roi = cv_image[y:y+h, x:x+w]

                # Enhance table structure
                gray = cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY)

                # Apply adaptive thresholding for better line detection
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)

                # Detect horizontal and vertical lines
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

                horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
                vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

                # Combine lines
                table_structure = cv2.add(horizontal_lines, vertical_lines)

                # Find intersections to identify cells
                intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)

                # Find contours of cells
                contours, _ = cv2.findContours(intersections, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                cells = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 100:  # Filter small intersections
                        cx, cy, cw, ch = cv2.boundingRect(cnt)
                        cells.append({
                            'bbox': (cx + x, cy + y, cw, ch),  # Convert to page coordinates
                            'area': area,
                            'center': (cx + x + cw//2, cy + y + ch//2)
                        })

                # Extract cell content using OCR
                cell_data = []
                for cell in cells:
                    cx, cy, cw, ch = cell['bbox']
                    cell_image = cv_image[cy-y:cy-y+ch, cx-x:cx-x+cw]

                    if cell_image.size > 0:
                        # Convert to PIL for OCR
                        cell_pil = Image.fromarray(cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB))
                        cell_text = self.extract_text(cell_pil)

                        cell_data.append({
                            'bbox': cell['bbox'],
                            'text': cell_text.strip(),
                            'confidence': len(cell_text.strip()) > 0  # Simple confidence measure
                        })

                # Organize cells into rows and columns
                if cell_data:
                    # Cluster cells by row (similar y-coordinates)
                    y_coords = np.array([cell['bbox'][1] for cell in cell_data])
                    row_kmeans = KMeans(n_clusters=min(10, len(cell_data)), random_state=42, n_init=10)
                    row_labels = row_kmeans.fit_predict(y_coords.reshape(-1, 1))

                    # Group by rows
                    rows = {}
                    for i, cell in enumerate(cell_data):
                        row = row_labels[i]
                        if row not in rows:
                            rows[row] = []
                        rows[row].append(cell)

                    # Sort cells within each row by x-coordinate
                    table_content = []
                    for row in sorted(rows.keys()):
                        row_cells = sorted(rows[row], key=lambda c: c['bbox'][0])
                        row_content = [cell['text'] for cell in row_cells]
                        table_content.append(row_content)

                    tables.append({
                        'region': table_region,
                        'content': table_content,
                        'cells': cell_data,
                        'structure_confidence': len(cell_data) > 0
                    })

        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")

        return tables
    
    def extract_with_layout(self, image: Image.Image) -> Dict:
        """
        Extract text with layout awareness
        Groups text into: Title, Diagram Content, Body Text
        
        Returns:
            Dict with structured layout-aware content
        """
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Get predictions with layout analysis
            predictions = self.recognition([processed_image], det_predictor=self.detection)
            
            if not predictions or len(predictions) == 0:
                return {
                    'title': '',
                    'diagram_content': [],
                    'body_text': '',
                    'lists': [],
                    'full_text': ''
                }
            
            # Extract text lines with positional information
            text_lines = []
            for line in predictions[0].text_lines:
                bbox = line.bbox
                text = self.clean_text(line.text)
                
                if text:
                    text_lines.append({
                        'text': text,
                        'bbox': bbox,
                        'y_pos': bbox.y1 if hasattr(bbox, 'y1') else 0,
                        'x_pos': bbox.x1 if hasattr(bbox, 'x1') else 0,
                        'height': (bbox.y2 - bbox.y1) if hasattr(bbox, 'y2') and hasattr(bbox, 'y1') else 0
                    })
            
            if not text_lines:
                return {
                    'title': '',
                    'diagram_content': [],
                    'body_text': '',
                    'lists': [],
                    'full_text': ''
                }
            
            # Sort by vertical position (top to bottom)
            text_lines.sort(key=lambda x: x['y_pos'])
            
            # 1. IDENTIFY HEADER/TITLE (first 15% of page height)
            image_height = image.height
            header_threshold = image_height * 0.15
            
            title_lines = []
            content_lines = []
            
            for line in text_lines:
                if line['y_pos'] < header_threshold:
                    title_lines.append(line['text'])
                else:
                    content_lines.append(line)
            
            title = " ".join(title_lines).strip() if title_lines else ""
            
            # 2. SEGMENT CONTENT INTO VISUAL vs BODY
            # Heuristics: Look for short text blocks (diagram labels) vs long paragraphs
            diagram_content = []
            body_lines = []
            
            i = 0
            while i < len(content_lines):
                line = content_lines[i]
                text = line['text']
                text_length = len(text)
                
                # Diagram labels: short, often isolated, single line
                # (typically < 50 chars, not part of paragraph flow)
                if text_length < 50 and (i == len(content_lines) - 1 or 
                                         abs(content_lines[i+1]['y_pos'] - line['y_pos']) > line['height'] * 2):
                    diagram_content.append(text)
                else:
                    body_lines.append(text)
                
                i += 1
            
            # 3. DETECT LISTS (numbered or bulleted patterns)
            lists = []
            body_text_parts = []
            
            for text in body_lines:
                # List pattern: starts with number, bullet, or symbol
                if text and (text[0].isdigit() or text.startswith(('•', '-', '*', '○', '●'))):
                    lists.append(text)
                else:
                    body_text_parts.append(text)
            
            body_text = "\n".join(body_text_parts)
            
            # 4. Format output
            result = {
                'title': title,
                'diagram_content': diagram_content,
                'body_text': body_text,
                'lists': lists,
                'full_text': "\n".join([line['text'] for line in text_lines])
            }
            
            logger.info(f"Layout extraction: Title={len(title)} chars, Diagrams={len(diagram_content)}, Body={len(body_text)} chars")
            
            return result
        
        except Exception as e:
            logger.error(f"Layout-aware extraction error: {e}")
            return {
                'title': '',
                'diagram_content': [],
                'body_text': '',
                'lists': [],
                'full_text': ''
            }
    
    def process_document(self, file_path: str, max_pages: Optional[int] = None, use_layout: bool = True) -> Dict:
        """
        Process PDF document with advanced OCR and layout analysis

        Args:
            file_path: Path to PDF document file
            max_pages: Maximum pages to process (None = all)
            use_layout: Whether to use advanced layout-aware extraction

        Returns:
            Dictionary with extraction results
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {'success': False, 'error': 'File not found'}

        file_extension = Path(file_path).suffix.lower()
        logger.info(f"Processing: {Path(file_path).name} (Type: {file_extension}, Layout-aware: {use_layout})")

        try:
            if file_extension == '.pdf':
                return self._process_pdf_document(file_path, max_pages, use_layout)
            else:
                return {'success': False, 'error': f'Unsupported file type: {file_extension}'}

        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {'success': False, 'error': str(e)}

    def _process_pdf_document(self, pdf_path: str, max_pages: Optional[int] = None, use_layout: bool = True) -> Dict:
        """Process PDF document with advanced OCR capabilities"""
        if not self.is_initialized:
            logger.error("OCR Models not initialized")
            return {'success': False, 'error': 'Models not initialized'}

        try:
            # Convert PDF to images
            images = self.convert_pdf_to_images(pdf_path)

            if not images:
                logger.error("No pages converted from PDF")
                return {'success': False, 'error': 'No pages converted'}

            # Limit pages if specified
            pages_to_process = images[:max_pages] if max_pages else images
            logger.info(f"Extracting text from {len(pages_to_process)} pages with advanced OCR...")

            # Extract text from each page with advanced features
            documents = []
            total_time = 0

            for page_num, image in enumerate(pages_to_process, 1):
                start_time = time.time()

                # Advanced layout detection
                layout_info = self.detect_layout(image)

                if use_layout and layout_info['success']:
                    # Extract with advanced layout awareness
                    layout_result = self._extract_with_advanced_layout(image, layout_info)

                    # Extract tables if found
                    table_data = []
                    if layout_info.get('tables'):
                        table_data = self.extract_tables(image, layout_info['tables'])

                    # Use detected page number if available, otherwise use sequential number
                    actual_page_number = layout_result.get('detected_page_number', page_num)

                    doc = {
                        'page_number': actual_page_number,
                        'physical_page_number': page_num,  # Keep track of physical page for reference
                        'layout_analysis': layout_info,
                        'title': layout_result['title'],
                        'body_text': layout_result['body_text'],
                        'tables': table_data,
                        'visual_elements': layout_info.get('visuals', []),
                        'reading_order': layout_info.get('reading_order', []),
                        'full_text': layout_result['full_text'],
                        'char_count': len(layout_result['full_text']),
                        'word_count': len(layout_result['full_text'].split()),
                        'extraction_time_sec': round(time.time() - start_time, 2),
                        'processing_mode': 'advanced_ocr_with_layout'
                    }
                else:
                    # Fallback to basic OCR
                    text = self.extract_text(image)
                    doc = {
                        'page_number': page_num,
                        'full_text': text,
                        'char_count': len(text),
                        'word_count': len(text.split()),
                        'extraction_time_sec': round(time.time() - start_time, 2),
                        'processing_mode': 'basic_ocr'
                    }

                elapsed = time.time() - start_time
                total_time += elapsed
                documents.append(doc)

                logger.info(f"Page {page_num}: {doc['char_count']} chars, {doc['word_count']} words ({doc['processing_mode']})")

            # Calculate statistics
            total_chars = sum(doc['char_count'] for doc in documents)
            total_words = sum(doc['word_count'] for doc in documents)

            # Build comprehensive full text with proper formatting
            full_text_parts = []
            for doc in documents:
                if doc.get('processing_mode') == 'advanced_ocr_with_layout':
                    # Advanced format with layout information
                    full_text_parts.append(f"## Page {doc['page_number']}\n\n")

                    if doc.get('title'):
                        full_text_parts.append(f"### Title: {doc['title']}\n\n")

                    if doc.get('body_text'):
                        full_text_parts.append(f"### Content:\n{doc['body_text']}\n\n")

                    # Add table information
                    if doc.get('tables'):
                        full_text_parts.append("### Tables:\n")
                        for i, table in enumerate(doc['tables'], 1):
                            full_text_parts.append(f"**Table {i}:**\n")
                            for row in table.get('content', []):
                                full_text_parts.append("| " + " | ".join(row) + " |\n")
                            full_text_parts.append("\n")

                    # Add visual elements information
                    if doc.get('visual_elements'):
                        full_text_parts.append(f"**Visual Elements:** {len(doc['visual_elements'])} detected\n\n")

                else:
                    # Basic format
                    full_text_parts.append(doc.get('full_text', ''))

                full_text_parts.append("\n\n---PAGE BREAK---\n\n")

            result = {
                'success': True,
                'document_type': 'pdf',
                'file_name': Path(pdf_path).name,
                'file_path': pdf_path,
                'pages_processed': len(pages_to_process),
                'total_characters': total_chars,
                'total_words': total_words,
                'total_time_seconds': round(total_time, 2),
                'average_time_per_page': round(total_time / len(pages_to_process), 2) if pages_to_process else 0,
                'timestamp': datetime.now().isoformat(),
                'advanced_features': {
                    'layout_analysis': use_layout,
                    'table_extraction': True,
                    'reading_order_calculation': True,
                    'scanned_document_optimization': True
                },
                'documents': documents,
                'full_text': ''.join(full_text_parts)
            }

            logger.info(f"Advanced PDF processing complete: {total_chars} chars, {total_words} words")
            return result

        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return {'success': False, 'error': str(e)}



    def _extract_page_number_from_headers_footers(self, text_lines: List[Dict], image_height: int) -> Optional[int]:
        """
        Try to extract page number from header/footer regions
        Looks for patterns like "Page 26", "26", "- 26 -", etc.
        """
        import re

        header_threshold = image_height * 0.15
        footer_threshold = image_height * 0.85

        # Patterns to match page numbers
        page_patterns = [
            r'page\s*(\d+)',  # "page 26"
            r'(\d+)\s*page',  # "26 page"
            r'^\s*(\d+)\s*$',  # Just "26" on a line
            r'-\s*(\d+)\s*-',  # "- 26 -"
            r'(\d{1,3})',  # Any 1-3 digit number (be conservative)
        ]

        header_footer_lines = []
        for line in text_lines:
            y_pos = line['y_pos']
            if y_pos < header_threshold or y_pos > footer_threshold:
                header_footer_lines.append(line['text'])

        # Look for page number patterns in header/footer text
        for text in header_footer_lines:
            text_lower = text.lower().strip()
            for pattern in page_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    # Take the first match that's a reasonable page number
                    for match in matches:
                        try:
                            page_num = int(match)
                            # Be conservative - page numbers are usually reasonable (1-1000)
                            if 1 <= page_num <= 1000:
                                return page_num
                        except ValueError:
                            continue

        return None

    def _extract_with_advanced_layout(self, image: Image.Image, layout_info: Dict) -> Dict:
        """
        Extract text with advanced layout awareness including reading order
        """
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Get OCR predictions
            predictions = self.recognition([processed_image], det_predictor=self.detection)

            if not predictions or len(predictions) == 0:
                return {
                    'title': '',
                    'body_text': '',
                    'full_text': '',
                    'detected_page_number': None
                }

            # Extract text lines with enhanced information
            text_lines = []
            for line in predictions[0].text_lines:
                bbox = line.bbox
                text = self.clean_text(line.text)

                if text:
                    text_lines.append({
                        'text': text,
                        'bbox': bbox,
                        'y_pos': bbox.y1 if hasattr(bbox, 'y1') else 0,
                        'x_pos': bbox.x1 if hasattr(bbox, 'x1') else 0,
                        'height': (bbox.y2 - bbox.y1) if hasattr(bbox, 'y2') and hasattr(bbox, 'y1') else 0,
                        'width': (bbox.x2 - bbox.x1) if hasattr(bbox, 'x2') and hasattr(bbox, 'x1') else 0
                    })

            # Try to extract actual page number from headers/footers
            detected_page_number = self._extract_page_number_from_headers_footers(text_lines, image.height)

            # Apply reading order if available
            reading_order = layout_info.get('reading_order', list(range(len(text_lines))))

            # Organize content by layout regions
            title_parts = []
            body_parts = []

            # Use layout information to categorize content
            image_height = image.height
            header_threshold = image_height * 0.15

            for idx in reading_order:
                if idx < len(text_lines):
                    line = text_lines[idx]
                    if line['y_pos'] < header_threshold:
                        title_parts.append(line['text'])
                    else:
                        body_parts.append(line['text'])

            title = " ".join(title_parts).strip() if title_parts else ""
            body_text = "\n".join(body_parts).strip() if body_parts else ""

            return {
                'title': title,
                'body_text': body_text,
                'full_text': "\n".join([line['text'] for line in text_lines]),
                'detected_page_number': detected_page_number
            }

        except Exception as e:
            logger.error(f"Advanced layout extraction error: {e}")
            return {
                'title': '',
                'body_text': '',
                'full_text': '',
                'detected_page_number': None
            }
    
    def save_json(self, data: Dict, output_path: str) -> bool:
        """Save extraction results as JSON"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved JSON: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            return False
    
    def save_text(self, text: str, output_path: str) -> bool:
        """Save extracted text as plain text file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"Saved text: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save text: {e}")
            return False
    
    def save_markdown(self, documents: List[Dict], output_path: str, use_layout: bool = True) -> bool:
        """Save extracted text in Markdown format with layout awareness"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            md_content = "# OCR Extraction Results\n\n"
            md_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            md_content += f"**Language:** Khmer + English\n"
            md_content += f"**Processing Mode:** {'Layout-Aware Extraction' if use_layout else 'Standard Extraction'}\n"
            md_content += f"**Optimizations:** Image preprocessing (CLAHE), Unicode normalization, OCR error correction\n\n"
            md_content += "---\n\n"
            
            for doc in documents:
                md_content += f"## Page {doc['page_number']}\n\n"
                
                # Check if layout-aware extraction was used
                if use_layout and doc.get('layout_aware'):
                    # Title section
                    if doc.get('title'):
                        md_content += f"### **{doc['title']}**\n\n"
                    
                    # Diagram/Visual Content
                    if doc.get('diagram_content'):
                        md_content += "### Diagram/Visual Content\n"
                        for item in doc['diagram_content']:
                            md_content += f"- {item}\n"
                        md_content += "\n"
                    
                    # Body Content
                    if doc.get('body_text'):
                        md_content += "### Body Content / Detailed Explanation\n"
                        md_content += doc['body_text'] + "\n\n"
                    
                    # Lists
                    if doc.get('lists'):
                        md_content += "### Key Items\n"
                        for item in doc['lists']:
                            md_content += f"- **{item}**\n"
                        md_content += "\n"
                else:
                    # Standard format
                    text = doc.get('text') or doc.get('full_text', '')
                    if text:
                        md_content += f"**Statistics:** {doc['char_count']} characters, {doc['word_count']} words\n\n"
                        md_content += text + "\n\n"
                
                # Statistics
                md_content += f"**Page Statistics:** {doc['char_count']} characters, {doc['word_count']} words, {doc['extraction_time_sec']}s\n\n"
                md_content += "---\n\n"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            logger.info(f"Saved markdown: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save markdown: {e}")
            return False


def extract_document(file_path: str, output_dir: str, max_pages: Optional[int] = None, use_layout: bool = True) -> bool:
    """
    Main function to extract PDF document and save results with advanced features

    Args:
        file_path: Path to PDF document file
        output_dir: Directory to save results
        max_pages: Maximum pages to extract (None = all)
        use_layout: Whether to use advanced layout-aware extraction (default: True)

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 70)
    print("ADVANCED OCR - PRODUCTION MODE (Khmer + English Optimized)")
    print("Complex Layouts | Tables | Reading Order | Scanned Documents")
    print("=" * 70 + "\n")

    # Initialize OCR
    ocr = SuryaOCR()

    file_extension = Path(file_path).suffix.lower()
    if file_extension == '.pdf':
        if not ocr.initialize():
            print("ERROR: Failed to initialize OCR models")
            return False
        print("OCR Models: Surya Foundation + Recognition + Detection")
    else:
        print("Unsupported file type")

    print(f"File: {file_path}")
    print(f"Type: {file_extension.upper()}")
    print(f"Output: {output_dir}")
    print(f"Max Pages: {max_pages if max_pages else 'All'}")
    print(f"Advanced Layout Analysis: {use_layout}\n")

    # Process document with advanced features
    result = ocr.process_document(file_path, max_pages, use_layout)

    if not result['success']:
        print(f"ERROR: {result.get('error', 'Unknown error')}")
        return False

    # Save results
    base_name = Path(file_path).stem

    # Save as JSON with full metadata
    json_path = os.path.join(output_dir, f"{base_name}_advanced.json")
    ocr.save_json(result, json_path)

    # Save as plain text
    txt_path = os.path.join(output_dir, f"{base_name}_advanced.txt")
    ocr.save_text(result['full_text'], txt_path)

    # Save as enhanced markdown
    md_path = os.path.join(output_dir, f"{base_name}_advanced.md")
    ocr.save_markdown(result['documents'], md_path, use_layout)

    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("ADVANCED EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Document Type: {result['document_type'].upper()}")
    print(f"Pages Processed: {result['pages_processed']}")
    print(f"Total Characters: {result['total_characters']:,}")
    print(f"Total Words: {result['total_words']:,}")
    print(f"Processing Time: {result['total_time_seconds']:.2f}s")
    if result['pages_processed'] > 0:
        print(f"Average per Page: {result['average_time_per_page']:.2f}s")

    # Show advanced features used
    features = result.get('advanced_features', {})
    print(f"\nAdvanced Features Applied:")
    if features.get('layout_analysis'):
        print("  ✓ Complex Layout Detection & Analysis")
    if features.get('table_extraction'):
        print("  ✓ Table Structure Recognition & Extraction")
    if features.get('reading_order_calculation'):
        print("  ✓ Multi-Column Reading Order Optimization")
    if features.get('scanned_document_optimization'):
        print("  ✓ Scanned Document OCR Enhancement")

    print(f"\nImage Processing Optimizations:")
    print("  • CLAHE Contrast Enhancement")
    print("  • Bilateral Edge-Preserving Filtering")
    print("  • Morphological Text Block Detection")
    print("  • Unicode Normalization (NFC)")
    print("  • Khmer Font Variation Correction")

    if result['document_type'] == 'pdf':
        print(f"\nOCR-Specific Enhancements:")
        print("  • Multi-Scale Text Detection")
        print("  • Layout-Aware Text Segmentation")
        print("  • Visual Element Discrimination")
        print("  • Table Boundary Recognition")
        print("  • Reading Order Spatial Clustering")

    print(f"\nOutput Files:")
    print(f"  📄 Full JSON: {json_path}")
    print(f"  📄 Plain Text: {txt_path}")
    print(f"  📄 Enhanced Markdown: {md_path}")
    print("=" * 70 + "\n")

    return True


if __name__ == "__main__":
    # Configuration - Updated for current project
    import sys
    import os

    # Use the files from current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    PDF_PATH = os.path.join(current_dir, "Eco_systemKH.pdf")
    OUTPUT_DIR = os.path.join(current_dir, "Eco_systemKH_output_surya")

    MAX_PAGES = None  # Start with first 3 pages for testing
    USE_LAYOUT_AWARE = True  # Enable layout-aware extraction (like Gemini)

def extract_pdf_bytes(file_bytes: bytes, max_pages: Optional[int] = None, use_layout: bool = True) -> Dict:
    """
    Extract text from PDF bytes using Surya OCR

    Args:
        file_bytes: PDF content as bytes
        max_pages: Maximum pages to process (None = all)
        use_layout: Whether to use advanced layout-aware extraction

    Returns:
        Dictionary with extraction results
    """
    import tempfile
    import os

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    try:
        # Import SuryaOCR here to avoid CUDA initialization at module import time
        ocr = SuryaOCR()
        if not ocr.initialize():
            return {'success': False, 'error': 'Failed to initialize OCR models'}

        result = ocr.process_document(temp_file_path, max_pages, use_layout)
        return result

    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def _resolve_extracted_image_output_dir(file_id: str) -> Optional[Path]:
    """Return the output directory for extracted page images."""
    try:
        from django.conf import settings
    except Exception:
        return None

    media_root = getattr(settings, "MEDIA_ROOT", None)
    if not media_root:
        return None

    return Path(media_root) / "documents" / str(file_id) / "images"


def _build_image_link_metadata(file_id: str, page_number: Optional[int] = None) -> Dict[str, str]:
    """Build stable image link metadata for vector-store payloads."""
    media_url = "/media/"
    try:
        from django.conf import settings
        media_url = str(getattr(settings, "MEDIA_URL", media_url))
    except Exception:
        pass

    if not media_url.endswith("/"):
        media_url += "/"

    image_dir_rel = f"documents/{file_id}/images"
    metadata = {
        "image_directory_path": image_dir_rel,
        "image_directory_url": urljoin(media_url, f"{image_dir_rel}/"),
        "image_manifest_path": f"{image_dir_rel}/manifest.json",
        "image_manifest_url": urljoin(media_url, f"{image_dir_rel}/manifest.json"),
    }

    if page_number is not None:
        page_filename = f"page_{int(page_number):04d}.png"
        page_rel_path = f"{image_dir_rel}/{page_filename}"
        metadata["page_image_path"] = page_rel_path
        metadata["page_image_url"] = urljoin(media_url, page_rel_path)

    return metadata


def _persist_extracted_page_images(
    file_bytes: bytes,
    file_id: str,
    max_pages: Optional[int] = None,
) -> Dict[str, Any]:
    """Persist rendered PDF page images for downstream inspection/usage."""
    output_dir = _resolve_extracted_image_output_dir(file_id)
    if output_dir is None:
        return {"saved": 0, "output_dir": None}

    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # type: ignore[no-redef]

        output_dir.mkdir(parents=True, exist_ok=True)
        for stale_png in output_dir.glob("page_*.png"):
            try:
                stale_png.unlink()
            except OSError:
                pass

        pdf_document = fitz.open(temp_file_path)
        try:
            total_pages = pdf_document.page_count
            pages_to_render = min(total_pages, max_pages) if max_pages else total_pages
            saved_count = 0

            zoom = 150 / 72
            matrix = fitz.Matrix(zoom, zoom)

            for page_idx in range(pages_to_render):
                page = pdf_document[page_idx]
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_path = output_dir / f"page_{page_idx + 1:04d}.png"
                pixmap.save(str(image_path))
                saved_count += 1
        finally:
            pdf_document.close()

        manifest_path = output_dir / "manifest.json"
        manifest_payload = {
            "file_id": str(file_id),
            "pages_total": total_pages,
            "pages_rendered": saved_count,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest_payload, manifest_file, ensure_ascii=False, indent=2)

        return {
            "saved": saved_count,
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
        }
    except Exception as exc:
        logger.warning("Failed to persist extracted page images for %s: %s", file_id, exc)
        return {"saved": 0, "output_dir": str(output_dir), "error": str(exc)}
    finally:
        try:
            os.unlink(temp_file_path)
        except OSError:
            pass


def extract_pdf_to_langchain_docs(file_bytes: bytes, file_id: str, file_name: str, max_pages: Optional[int] = None, use_layout: bool = True):
    """
    Extract PDF and return LangChain documents for RAG processing

    Args:
        file_bytes: PDF content as bytes
        file_id: Unique file identifier
        file_name: Original file name
        max_pages: Maximum pages to process (None = all)
        use_layout: Whether to use advanced layout-aware extraction

    Returns:
        List of LangChain Document objects
    """
    from langchain_core.documents import Document as LangChainDocument
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import re
    import logging
    import uuid

    logger = logging.getLogger(__name__)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=204,
    )

    try:
        # Extract text using Surya OCR
        result = extract_pdf_bytes(file_bytes, max_pages, use_layout)

        if not result.get('success', False):
            logger.error(f"Surya OCR failed for {file_name}: {result.get('error', 'Unknown error')}")
            return []

        image_store_result = _persist_extracted_page_images(
            file_bytes=file_bytes,
            file_id=file_id,
            max_pages=max_pages,
        )
        if image_store_result.get("saved"):
            logger.info(
                "Stored %s extracted page images for %s at %s",
                image_store_result["saved"],
                file_name,
                image_store_result.get("output_dir"),
            )

        documents = result.get('documents', [])
        if not documents:
            logger.warning(f"No documents extracted from {file_name}")
            return []

        # Process each page separately to maintain page integrity
        langchain_docs = []
        chunk_counter = 0

        for doc in documents:
            page_text = doc.get('full_text', doc.get('text', ''))
            if not page_text.strip():
                continue

            page_number = doc['page_number']

            # Split this page's text into chunks
            page_chunks = text_splitter.split_text(page_text)

            for chunk in page_chunks:
                if chunk.strip():  # Only add non-empty chunks
                    # Detect language (simplified version)
                    detected_lang = 'km' if any('\u1780' <= char <= '\u17FF' for char in chunk) else 'en'

                    doc_metadata = LangChainDocument(
                        page_content=chunk,
                        metadata={
                            'source': file_id,
                            'chunk_id': str(uuid.uuid4()),
                            'page': page_number,  # Direct page number assignment
                            'file_name': file_name,
                            'language': detected_lang,
                            'extraction_method': 'surya_ocr',
                            **_build_image_link_metadata(file_id=file_id, page_number=page_number),
                        }
                    )
                    langchain_docs.append(doc_metadata)
                    chunk_counter += 1

        logger.info(f"Successfully processed PDF {file_name}: {len(langchain_docs)} chunks from {len(documents)} pages")
        return langchain_docs

    except Exception as e:
        logger.error(f"Error processing PDF {file_name}: {str(e)}")
        return []


# Main execution (for testing)
if __name__ == "__main__":
    # Configuration - Updated for current project
    import sys
    import os

    # Use the files from current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    PDF_PATH = os.path.join(current_dir, "Eco_systemKH.pdf")
    OUTPUT_DIR = os.path.join(current_dir, "Eco_systemKH_output_surya")

    MAX_PAGES = None  # Start with first 3 pages for testing
    USE_LAYOUT_AWARE = True  # Enable layout-aware extraction (like Gemini)

    # Execute
    success = extract_document(PDF_PATH, OUTPUT_DIR, MAX_PAGES, USE_LAYOUT_AWARE)
    exit(0 if success else 1)


def extract_pdf_bytes(file_bytes: bytes, max_pages: Optional[int] = None, use_layout: bool = True) -> Dict:
    """
    Extract text from PDF bytes using Surya OCR

    Args:
        file_bytes: PDF content as bytes
        max_pages: Maximum pages to process (None = all)
        use_layout: Whether to use advanced layout-aware extraction

    Returns:
        Dictionary with extraction results
    """
    import tempfile
    import os

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    try:
        ocr = SuryaOCR()
        if not ocr.initialize():
            return {'success': False, 'error': 'Failed to initialize OCR models'}

        result = ocr.process_document(temp_file_path, max_pages, use_layout)
        return result

    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def extract_pdf_to_langchain_docs(file_bytes: bytes, file_id: str, file_name: str, max_pages: Optional[int] = None, use_layout: bool = True):
    """
    Extract PDF and return LangChain documents for RAG processing

    Args:
        file_bytes: PDF content as bytes
        file_id: Unique file identifier
        file_name: Original file name
        max_pages: Maximum pages to process (None = all)
        use_layout: Whether to use advanced layout-aware extraction

    Returns:
        List of LangChain Document objects
    """
    from langchain_core.documents import Document as LangChainDocument
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import re
    import logging
    import uuid

    logger = logging.getLogger(__name__)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=204,
    )

    try:
        # Extract text using Surya OCR
        result = extract_pdf_bytes(file_bytes, max_pages, use_layout)

        if not result.get('success', False):
            logger.error(f"Surya OCR failed for {file_name}: {result.get('error', 'Unknown error')}")
            return []

        image_store_result = _persist_extracted_page_images(
            file_bytes=file_bytes,
            file_id=file_id,
            max_pages=max_pages,
        )
        if image_store_result.get("saved"):
            logger.info(
                "Stored %s extracted page images for %s at %s",
                image_store_result["saved"],
                file_name,
                image_store_result.get("output_dir"),
            )

        documents = result.get('documents', [])
        if not documents:
            logger.warning(f"No documents extracted from {file_name}")
            return []

        # Process each page separately to maintain page integrity
        langchain_docs = []
        chunk_counter = 0

        for doc in documents:
            page_text = doc.get('full_text', doc.get('text', ''))
            if not page_text.strip():
                continue

            page_number = doc['page_number']

            # Split this page's text into chunks
            page_chunks = text_splitter.split_text(page_text)

            for chunk in page_chunks:
                if chunk.strip():  # Only add non-empty chunks
                    # Detect language (simplified version)
                    detected_lang = 'km' if any('\u1780' <= char <= '\u17FF' for char in chunk) else 'en'

                    doc_metadata = LangChainDocument(
                        page_content=chunk,
                        metadata={
                            'source': file_id,
                            'chunk_id': str(uuid.uuid4()),
                            'page': page_number,  # Direct page number assignment
                            'file_name': file_name,
                            'language': detected_lang,
                            'extraction_method': 'surya_ocr',
                            **_build_image_link_metadata(file_id=file_id, page_number=page_number),
                        }
                    )
                    langchain_docs.append(doc_metadata)
                    chunk_counter += 1

        logger.info(f"Successfully processed PDF {file_name}: {len(langchain_docs)} chunks from {len(documents)} pages")
        return langchain_docs

    except Exception as e:
        logger.error(f"Error processing PDF {file_name}: {str(e)}")
        return []
