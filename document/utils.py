import base64
import logging
import os
import time
from datetime import date, datetime

from django.conf import settings

from base.services.agent.rag.config import SUPPORTED_EXTENSIONS
from base.services.agent.rag.providers import RagProviders

logger = logging.getLogger(__name__)


SUPPORTED_DATE_INPUT_FORMATS = ("%Y-%m-%d", "%d-%m-%Y")


def parse_document_date(value, *, field_name: str, allow_blank: bool = False):
    """Parse a document date from API input."""
    if value is None or isinstance(value, date):
        return value

    if isinstance(value, str):
        normalized = value.strip()
        if normalized == "":
            if allow_blank:
                return None
            raise ValueError(
                f"Invalid {field_name} format. Use YYYY-MM-DD or DD-MM-YYYY"
            )

        for fmt in SUPPORTED_DATE_INPUT_FORMATS:
            try:
                return datetime.strptime(normalized, fmt).date()
            except (ValueError, TypeError):
                continue

    raise ValueError(f"Invalid {field_name} format. Use YYYY-MM-DD or DD-MM-YYYY")


def validate_document_date_range(effective_date, expiry_date):
    """Ensure the expiry date is not earlier than the effective date."""
    if effective_date and expiry_date and expiry_date < effective_date:
        raise ValueError("Expiry date must be on or after effective date.")


def format_document_date(value):
    """Format a document date for API responses."""
    if value and isinstance(value, date):
        return value.strftime("%d-%m-%Y")
    return None


def clean_filename(filename):
    name, ext = os.path.splitext(filename)
    name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
    return f"{name}{ext}"


def generate_unique_filename(uploaded_file):
    original_name = clean_filename(uploaded_file.name)
    timestamp = int(time.time())
    name, ext = os.path.splitext(original_name)
    unique_name = f"{name}_{timestamp}{ext}"
    return unique_name


def upload_pdf_to_rag(
    file_path: str,
    file_id: str,
    file_name: str,
    file_type_id: str,
    file_type_name: str,
):
    """
    Read a file, encode it in base64, and upload it using RagProviders.

    Supports all file types defined in SUPPORTED_EXTENSIONS plus PDF.
    """
    with open(file_path, "rb") as file_handle:
        file_content = file_handle.read()
        base64_content = base64.b64encode(file_content).decode("utf-8")

    provider = RagProviders()
    return provider.upload_file(
        file_id=file_id,
        file_base64=base64_content,
        file_name=file_name,
        file_type=file_type_name,
        file_type_id=file_type_id,
    )


def delete_file_from_rag(file_id: str, file_type: str):
    """Delete all chunks for a file from the vector store."""
    provider = RagProviders()
    return provider.delete_file(file_id=file_id, file_type=file_type)


def deactivate_document_from_rag(file_id: str, file_type: str):
    """Deactivate a document in the RAG system."""
    provider = RagProviders()
    return provider.deactivate_document(file_id=file_id, file_type=file_type)


def activate_document_from_rag(file_id: str):
    """Activate a document in the RAG system."""
    provider = RagProviders()
    return provider.activate_document(file_id=file_id)


def is_supported_file_type(filename: str) -> tuple[bool, str]:
    """Check whether a filename is supported by the current ingestion stack."""
    file_ext = os.path.splitext(filename.lower())[1]

    for category, extensions in SUPPORTED_EXTENSIONS.items():
        if file_ext in extensions:
            return True, category

    if file_ext == ".pdf":
        return True, "pdf"

    return False, None


def get_supported_extensions() -> dict:
    """Return all supported extensions grouped by category."""
    supported = SUPPORTED_EXTENSIONS.copy()
    supported["pdf"] = [".pdf"]
    return supported


def get_all_supported_extensions_flat() -> list:
    """Return all supported extensions as a flat list."""
    extensions = []
    for category_extensions in SUPPORTED_EXTENSIONS.values():
        extensions.extend(category_extensions)
    extensions.append(".pdf")
    return extensions


def convert_to_pdf(input_path: str, output_path: str = None, timeout: int = None) -> str:
    """
    Convert a document to PDF using the conversion API.
    """
    import requests

    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file does not exist: {input_path}")

        if output_path is None:
            input_dir = os.path.dirname(input_path)
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(input_dir, f"{input_name}.pdf")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        logger.info("Input file size: %.1fMB", file_size_mb)

        if timeout is None:
            env_timeout_raw = str(os.environ.get("DOC_CONVERTER_TIMEOUT_SECONDS", "")).strip()
            env_timeout = None
            if env_timeout_raw:
                try:
                    env_timeout = int(float(env_timeout_raw))
                except ValueError:
                    logger.warning(
                        "Invalid DOC_CONVERTER_TIMEOUT_SECONDS=%r; ignoring environment override",
                        env_timeout_raw,
                    )

            # DOCX->PDF conversions can take significantly longer than a simple
            # file-size heuristic, especially under load. Use a safer baseline.
            auto_timeout = max(120, int(60 + (file_size_mb * 25)))
            timeout = env_timeout if env_timeout and env_timeout > 0 else auto_timeout
            logger.info(
                "Auto-calculated timeout: %s seconds (based on %.1fMB file, env_override=%s)",
                timeout,
                file_size_mb,
                env_timeout if env_timeout and env_timeout > 0 else "none",
            )
        else:
            logger.info("Using provided timeout: %s seconds", timeout)

        with open(input_path, "rb") as file_handle:
            response = requests.post(
                settings.DOC_CONVERTER_URL,
                files={"file": file_handle},
                timeout=timeout,
            )

        logger.info("API response status: %s", response.status_code)

        if response.status_code == 200:
            with open(output_path, "wb") as file_handle:
                file_handle.write(response.content)

            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(
                "Successfully converted %s to %s using API", input_path, output_path
            )
            logger.info("Output PDF size: %.1fMB", output_size_mb)
            return output_path

        error_msg = f"API conversion failed with status {response.status_code}"
        try:
            error_msg += f": {response.json()}"
        except Exception:
            error_msg += f": {response.text[:200]}"

        logger.error(error_msg)
        raise Exception(error_msg)

    except requests.exceptions.Timeout:
        logger.error(
            "Conversion API timeout (%ss) while converting %s (%.1fMB)",
            timeout,
            input_path,
            file_size_mb,
        )
        raise Exception(
            f"Conversion API timeout after {timeout}s - document may be too large or server is slow. Try again later."
        )
    except requests.exceptions.ConnectionError as exc:
        raise Exception(f"Cannot connect to conversion API server: {str(exc)}")
    except requests.exceptions.RequestException as exc:
        logger.error("Network error during PDF conversion: %s", str(exc))
        raise Exception(f"Failed to connect to conversion API: {str(exc)}")
    except Exception as exc:
        logger.error("Error converting document to PDF: %s", str(exc))
        raise


def get_pdf_version_path(document) -> str:
    """Return the path for the PDF version used by document viewing."""
    if not document.full_file_path:
        return None

    original_path = document.full_file_path
    return os.path.splitext(original_path)[0] + "_view.pdf"


def ensure_pdf_version_exists(document, *, force_refresh: bool = False) -> bool:
    """Ensure a PDF version exists for the document viewer."""
    try:
        pdf_path = get_pdf_version_path(document)

        if not pdf_path:
            logger.warning("Cannot determine PDF path for document %s", document.id)
            return False

        if not document.full_file_path or not os.path.exists(document.full_file_path):
            logger.warning("Original file not found for document %s", document.id)
            return False

        if force_refresh and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except OSError as exc:
                logger.warning(
                    "Failed to remove cached PDF for document %s before refresh: %s",
                    document.id,
                    exc,
                )

        # Uploaded document versions are immutable in practice, so reuse the cached
        # sibling PDF whenever it already exists. Explicit refresh=true still forces
        # regeneration for maintenance/debugging.
        if not force_refresh and os.path.exists(pdf_path):
            logger.info("Using cached PDF version for document %s", document.id)
            return True

        logger.info("Converting document %s to PDF for viewing", document.id)
        convert_to_pdf(document.full_file_path, pdf_path)

        return os.path.exists(pdf_path)

    except Exception as exc:
        logger.error(
            "Error ensuring PDF version exists for document %s: %s",
            document.id,
            str(exc),
        )
        return False


def get_document_preview_mode(document) -> str:
    """Return the secure preview mode for the document."""
    normalized_file_format = (getattr(document, "file_format", "") or "").strip().lower()
    if not normalized_file_format and getattr(document, "original_filename", None):
        normalized_file_format = os.path.splitext(document.original_filename)[1].lstrip(".").lower()

    if normalized_file_format == "txt":
        return "text"

    return "pages"


def get_preview_source_pdf_path(document) -> str | None:
    """Return the PDF source path used for secure preview generation."""
    if not document.full_file_path or not os.path.exists(document.full_file_path):
        return None

    if get_document_preview_mode(document) == "text":
        return None

    normalized_file_format = (getattr(document, "file_format", "") or "").strip().lower()
    if not normalized_file_format and getattr(document, "original_filename", None):
        normalized_file_format = os.path.splitext(document.original_filename)[1].lstrip(".").lower()

    if normalized_file_format == "pdf":
        return document.full_file_path

    pdf_path = get_pdf_version_path(document)
    if ensure_pdf_version_exists(document):
        return pdf_path

    return None


def get_preview_image_directory(document) -> str | None:
    """Return the directory containing rendered preview page images."""
    source_pdf_path = get_preview_source_pdf_path(document)
    if not source_pdf_path:
        return None
    return os.path.splitext(source_pdf_path)[0] + "_preview_pages"


def get_preview_image_path(document, page_number: int) -> str | None:
    """Return the rendered image path for a preview page."""
    preview_dir = get_preview_image_directory(document)
    if not preview_dir:
        return None
    return os.path.join(preview_dir, f"page_{page_number}.png")


def get_preview_page_count(document) -> int:
    """Return the number of preview pages available for a document."""
    source_pdf_path = get_preview_source_pdf_path(document)
    if not source_pdf_path:
        raise FileNotFoundError("Preview source PDF could not be prepared")

    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # type: ignore[no-redef]

    pdf_document = fitz.open(source_pdf_path)
    try:
        return pdf_document.page_count
    finally:
        pdf_document.close()


def ensure_preview_page_exists(document, page_number: int, *, zoom_factor: float = 2.0) -> str:
    """Render a preview page image if needed and return its path."""
    if page_number < 1:
        raise ValueError("Preview page numbers start at 1")

    source_pdf_path = get_preview_source_pdf_path(document)
    if not source_pdf_path:
        raise FileNotFoundError("Preview source PDF could not be prepared")

    preview_image_path = get_preview_image_path(document, page_number)
    if not preview_image_path:
        raise FileNotFoundError("Preview image path could not be prepared")

    source_mtime = os.path.getmtime(source_pdf_path)
    if os.path.exists(preview_image_path) and os.path.getmtime(preview_image_path) >= source_mtime:
        return preview_image_path

    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # type: ignore[no-redef]

    pdf_document = fitz.open(source_pdf_path)
    try:
        if page_number > pdf_document.page_count:
            raise IndexError("Preview page does not exist")

        page = pdf_document.load_page(page_number - 1)
        matrix = fitz.Matrix(zoom_factor, zoom_factor)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        os.makedirs(os.path.dirname(preview_image_path), exist_ok=True)
        pixmap.save(preview_image_path)
    finally:
        pdf_document.close()

    return preview_image_path


def load_text_preview(document) -> str:
    """Return a text-only preview for plain text documents."""
    if not document.full_file_path or not os.path.exists(document.full_file_path):
        raise FileNotFoundError("Preview source file does not exist")

    with open(document.full_file_path, "r", encoding="utf-8", errors="replace") as file_handle:
        return file_handle.read()


def get_view_only_group_names() -> tuple[str, ...]:
    """Return the group names that represent the restricted User role."""
    configured_user_group = getattr(settings, "AUTH_LDAP_DJANGO_USER_GROUP", "User")
    names: list[str] = []
    for name in ("User", configured_user_group):
        if name and name not in names:
            names.append(name)
    return tuple(names)


def is_view_only_role_user(user) -> bool:
    """Return whether the authenticated user belongs to the restricted User role."""
    if not getattr(user, "is_authenticated", False):
        return False
    return user.groups.filter(name__in=get_view_only_group_names()).exists()


def get_pdf_download_group_names() -> tuple[str, ...]:
    """Return the group names that should download office documents as PDFs."""
    return ("UserDownload",)


def is_pdf_download_role_user(user) -> bool:
    """Return whether the authenticated user belongs to the PDF-download role."""
    if not getattr(user, "is_authenticated", False):
        return False
    return user.groups.filter(name__in=get_pdf_download_group_names()).exists()
