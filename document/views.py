# ------------------------------------------------------------------------------------
# Document Views
# ------------------------------------------------------------------------------------
# Functions in this file:

# - upload_document: Uploads a new document.
# - list_documents: Lists all active documents with pagination/filtering.
# - download_document: Downloads a document file.
# - upload_new_version: Uploads a new version of an existing document.
# - update_document: Updates document metadata.
# - get_document_versions: Lists all versions of a document.
# - delete_document: Deletes a document (soft/hard, single/all versions).
# - DocumentDetailView: Retrieves document details (DRF generic view).
# - restore_document: Restores a soft-deleted document.
# - update_document_version: Manually updates document version and fields.
# - list_inactive_documents: Lists all inactive (soft-deleted) documents.
# ------------------------------------------------------------------------------------

from django.shortcuts import get_object_or_404
from django.http import FileResponse, Http404, JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.template.response import TemplateResponse
from django.urls import reverse
from urllib.parse import quote
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework import status
from rest_framework.response import Response
from rest_framework.exceptions import NotAuthenticated
from django.middleware.csrf import get_token
from django.db import models 
from django.db.models import Prefetch
from django.conf import settings
import os
import uuid
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from .models import Document, DocumentIndexingJob
from department.models import Department
from docs_type.models import DocumentType
from base.utils import success_response, error_response, handle_exception
from base.utils.validators import (
    DocumentUploadRequest,
    DocumentDeleteRequest,
)
from base.utils.exceptions import (
    DocumentError,
    DocumentNotFoundError,
    InvalidDocumentError,
    ConflictError,
    InternalServerError,
    ValidationError as APIValidationError
)
from pydantic import ValidationError
import logging
from base.utils.api_pagination import StandardResultsSetPagination
from .utils import (
    delete_file_from_rag,
    ensure_preview_page_exists,
    ensure_pdf_version_exists,
    format_document_date,
    generate_unique_filename,
    get_document_preview_mode,
    get_preview_page_count,
    is_pdf_download_role_user,
    is_view_only_role_user,
    load_text_preview,
    parse_document_date,
    validate_document_date_range,
)
from rest_framework import generics
from .serializers import (
    DocumentDetailSerializer,
    DocumentUpdateSerializer,
    DocumentListSerializer,
    DocumentSuggestionSerializer,
    get_indexing_message,
    get_indexing_status_label,
)
from .tasks import activate_document, cleanup_deleted_document_artifacts, deactivate_document
from base.application.document_indexing import enqueue_document_indexing, get_latest_indexing_job

logger = logging.getLogger(__name__)

DOCX_ONLY_UPLOAD_EXTENSIONS = {".docx"}
DOCX_ONLY_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/octet-stream",
}


def _is_docx_filename(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in DOCX_ONLY_UPLOAD_EXTENSIONS


def _validate_docx_upload(file_name: str, content_type: str | None = None) -> tuple[bool, str]:
    if not _is_docx_filename(file_name):
        return False, "Only DOCX files are allowed right now."

    if content_type and content_type not in DOCX_ONLY_MIME_TYPES:
        return False, "Invalid DOCX file type."

    return True, ""


INDEXING_JOBS_PREFETCH = Prefetch(
    "indexing_jobs",
    queryset=DocumentIndexingJob.objects.order_by("-created_at"),
    to_attr="prefetched_indexing_jobs",
)


def _get_root_document(document: Document) -> Document:
    """Return the top-most ancestor in the document version chain."""
    root = document
    while root.parent_document_id:
        root = root.parent_document
    return root


def _get_document_family_ids(root_document_id) -> set:
    """Return all IDs in a version family (root + descendants)."""
    family_ids = {root_document_id}
    frontier = {root_document_id}

    while frontier:
        child_ids = set(
            Document.objects.filter(parent_document_id__in=frontier).values_list("id", flat=True)
        )
        new_ids = child_ids - family_ids
        if not new_ids:
            break
        family_ids.update(new_ids)
        frontier = new_ids

    return family_ids


def _get_document_family_queryset(root_document: Document, include_inactive: bool = True):
    """Return queryset over all documents in a version family."""
    family_ids = _get_document_family_ids(root_document.id)
    queryset = Document.objects.filter(id__in=family_ids)
    if not include_inactive:
        queryset = queryset.filter(is_active=True)
    return queryset


def _resolve_document_access(request, pk, *, allow_token: bool = True):
    """Resolve document access for an authenticated user or a signed token."""
    token = request.GET.get("token") if allow_token else None
    can_view = False
    can_download = False
    force_secure_preview = False
    force_pdf_download = False

    if token:
        from django.core import signing
        from django.utils import timezone

        try:
            token_data = signing.loads(token)
            if token_data["document_id"] != str(pk):
                raise Http404("Invalid token")
            if timezone.now().timestamp() > token_data["expires_at"]:
                raise Http404("Token expired")
        except (signing.BadSignature, KeyError):
            logger.error("Invalid token for document %s", pk)
            raise Http404("Invalid token")

        can_view = True
        can_download = token_data.get("can_download", False)
        force_secure_preview = token_data.get("force_secure_preview", False)
        force_pdf_download = token_data.get("force_pdf_download", False)
    else:
        if not request.user.is_authenticated:
            raise NotAuthenticated("Authentication required")

        can_view = request.user.has_perm("document.can_view_document")
        can_download = request.user.has_perm("document.can_download_document")
        force_secure_preview = is_view_only_role_user(request.user) and not can_download
        force_pdf_download = is_pdf_download_role_user(request.user) and can_download

        if not can_view and not can_download:
            logger.warning("Access denied for user %s on document %s", request.user, pk)
            raise Http404("You don't have permission to access this document")

    document = get_object_or_404(Document, id=pk, is_active=True)
    if not document.file_path:
        logger.error("File path not set for document %s", pk)
        raise Http404("File path not found")

    return {
        "document": document,
        "can_view": can_view,
        "can_download": can_download,
        "force_secure_preview": force_secure_preview,
        "force_pdf_download": force_pdf_download,
        "token": token,
    }


def _append_token_query(url: str, token: str | None) -> str:
    """Append a signed token query parameter when present."""
    if not token:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}token={quote(token, safe='')}"


def _render_secure_preview(request, document: Document, *, token: str | None = None):
    """Render a view-only preview page without exposing the original file bytes."""
    preview_mode = get_document_preview_mode(document)
    context = {
        "document": document,
        "preview_mode": preview_mode,
        "page_urls": [],
        "page_count": 0,
        "preview_text": "",
        "watermark_text": "",
    }

    viewer_name = ""
    if request.user.is_authenticated:
        viewer_name = request.user.get_full_name().strip() or request.user.username
    elif token:
        viewer_name = ""

    if viewer_name:
        context["watermark_text"] = f"VIEW ONLY | {viewer_name}"

    try:
        if preview_mode == "text":
            context["preview_text"] = load_text_preview(document)
        else:
            page_count = get_preview_page_count(document)
            context["page_count"] = page_count
            context["page_urls"] = [
                {
                    "number": page_number,
                    "url": _append_token_query(
                        reverse(
                            "document:view_document_preview_page",
                            args=[document.id, page_number],
                        ),
                        token,
                    ),
                }
                for page_number in range(1, page_count + 1)
            ]
    except (FileNotFoundError, IndexError, OSError, ValueError) as exc:
        logger.error("Failed to build secure preview for document %s: %s", document.id, exc)
        return error_response(
            "Failed to generate secure preview",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    response = TemplateResponse(request, "document_preview.html", context)
    response["Cache-Control"] = "no-store"
    response["Pragma"] = "no-cache"
    response["X-Content-Type-Options"] = "nosniff"
    return response


def _get_pdf_filename(document: Document) -> str:
    """Return the filename used when serving a PDF representation."""
    original_filename = document.original_filename or document.stored_filename or str(document.id)
    base_name = os.path.splitext(original_filename)[0] or str(document.id)
    return f"{base_name}.pdf"


def _build_pdf_file_response(request, document: Document, pdf_path: str, *, as_attachment: bool = False):
    """Build a PDF file response without closing the file before streaming starts."""
    try:
        file_obj = open(pdf_path, "rb")
    except (OSError, IOError) as exc:
        logger.error("Error opening PDF file %s: %s", pdf_path, exc)
        raise Http404("File cannot be opened")

    response = FileResponse(file_obj, content_type="application/pdf")
    pdf_filename = quote(_get_pdf_filename(document), safe="")
    disposition_type = "attachment" if as_attachment else "inline"
    response["Content-Disposition"] = (
        f'{disposition_type}; filename="{pdf_filename}"; filename*=UTF-8\'\'{pdf_filename}'
    )
    response["Access-Control-Allow-Origin"] = request.headers.get("Origin", "*")
    response["Access-Control-Allow-Credentials"] = "true"
    response["X-Content-Type-Options"] = "nosniff"
    return response

@api_view(['GET'])
@permission_classes([])  # Allow unauthenticated access
def get_csrf_token(request):
    """Endpoint to provide CSRF token to frontend for cross-origin requests."""
    token = get_token(request)
    return Response({'csrfToken': token})

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_document(request):
    """Document upload: accepts JSON with base64-encoded file data instead of multipart."""
    # Permission check
    if not request.user.has_perm('document.can_upload_document'):
        return error_response("Forbidden", status_code=403)
    
    try:
        logger.info(f"[DEBUG] upload_document: received request with content-type: {request.content_type}")
        logger.info(f"[DEBUG] upload_document: request.body length: {len(request.body) if hasattr(request, 'body') else 'N/A'}")
        # Parse JSON body (avoids multipart parsing issues)
        data = request.data
        
        # Extract file data from base64
        file_data_b64 = data.get('file_data')
        if not file_data_b64:
            return error_response("No file provided", status_code=status.HTTP_400_BAD_REQUEST)
        
        # Decode base64 to bytes
        import base64
        try:
            file_bytes = base64.b64decode(file_data_b64)
        except Exception as e:
            logger.error(f"Invalid base64 encoding: {e}")
            return error_response("Invalid file data encoding", status_code=status.HTTP_400_BAD_REQUEST)
        
        file_name = data.get('file_name', 'document')
        is_valid_file, file_error = _validate_docx_upload(file_name)
        if not is_valid_file:
            return error_response(file_error, status_code=status.HTTP_400_BAD_REQUEST)
        logger.info(f"File upload from {request.user}: {file_name} ({len(file_bytes)} bytes)")
        
        # Extract form fields
        form_data = {
            'title': data.get('title'),
            'department_id': data.get('department_id'),
            'type_id': data.get('type_id'),
            'effective_date': data.get('effective_date'),
            'expiry_date': data.get('expiry_date'),
            'memos': data.get('memos', ''),
        }
        
    except Exception as e:
        logger.error(f"Error parsing request: {type(e).__name__}: {str(e)}")
        return error_response("Error parsing request", status_code=status.HTTP_400_BAD_REQUEST)
    
    # Validate request data with Pydantic
    try:
        validated_data = DocumentUploadRequest(**form_data)
        logger.debug(f"Validation successful. type_id={validated_data.type_id}, department_id={validated_data.department_id}")
    except ValidationError as e:
        error_messages = []
        for error in e.errors():
            error_messages.append({
                "field": error.get("loc", [None])[0],
                "message": error.get("msg", "Validation error")
            })
        logger.warning(f"Validation error: {error_messages}")
        return error_response(
            "Invalid request data",
            data={"errors": error_messages},
            status_code=status.HTTP_400_BAD_REQUEST
        )
    
    # Validate file size
    max_file_size = 100 * 1024 * 1024  # 100MB
    if len(file_bytes) > max_file_size:
        return error_response(
            f"File too large. Maximum {max_file_size // (1024*1024)}MB allowed",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        )
    
    # Parse and validate dates
    try:
        effective_date = parse_document_date(
            validated_data.effective_date,
            field_name="effective_date",
        )
        expiry_date = parse_document_date(
            validated_data.expiry_date,
            field_name="expiry_date",
        )
        validate_document_date_range(effective_date, expiry_date)
    except ValueError as exc:
        return error_response(str(exc), status_code=status.HTTP_400_BAD_REQUEST)
    
    # Validate required fields
    if not validated_data.department_id or not validated_data.type_id:
        return error_response(
            "Department and document type required",
            status_code=status.HTTP_400_BAD_REQUEST
        )
    
    # Get and validate department and document type
    try:
        department = Department.objects.get(id=validated_data.department_id)
        doc_type = DocumentType.objects.get(id=validated_data.type_id)
    except (Department.DoesNotExist, DocumentType.DoesNotExist):
        return error_response(
            "Invalid department or document type",
            status_code=status.HTTP_404_NOT_FOUND
        )
    
    # Save file to disk
    file_ext = os.path.splitext(file_name)[1]
    unique_name = f"{uuid.uuid4()}_{file_name}"
    file_path = f"documents/{unique_name}"
    full_path = os.path.join(settings.MEDIA_ROOT, file_path)
    
    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            f.write(file_bytes)
        logger.info(f"File saved to {full_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        return error_response("Failed to save file", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    # Create document record
    try:
        first_name = request.user.first_name
        last_name = request.user.last_name
        document = Document.objects.create(
            title=validated_data.title,
            description=validated_data.memos or '',
            original_filename=file_name,
            stored_filename=unique_name,
            file_path=file_path,
            file_size=len(file_bytes),
            file_format=file_ext[1:] if file_ext else '',
            department=department,
            type=doc_type,
            effective_date=effective_date or datetime.now().date(),
            expiry_date=expiry_date,
            publisher_id=request.user.id,
            publisher_name=f"{first_name} {last_name}".strip() or request.user.username,
        )
        logger.info(f"Document created: {document.id}")
    except Exception as e:
        logger.error(f"Failed to create document record: {e}")
        # Clean up the file
        try:
            os.remove(full_path)
        except:
            pass
        return error_response("Failed to create document record", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    # Queue for vector DB processing
    try:
        indexing = enqueue_document_indexing(document)
        logger.info(f"Queued vector processing for {document.id} with job {indexing.job.id}")
    except Exception as e:
        logger.error(f"Failed to queue vector processing: {e}")
        # Don't fail the upload if vector processing fails
    
    return success_response(
        "Document uploaded successfully and is processing.", 
        {
            "id": str(document.id),
            "title": document.title,
            "original_filename": document.original_filename,
            "file_size": document.file_size,
            "status": "Processing",
            "indexing_job_id": str(indexing.job.id) if 'indexing' in locals() else None,
        },
        status.HTTP_201_CREATED
    )


@api_view(['GET'])
@handle_exception
def get_document_index_status(request, pk):
    if not request.user.has_perm('document.can_view_document'):
        return error_response("Forbidden", status_code=403)

    document = get_object_or_404(Document, id=pk)
    latest_job = get_latest_indexing_job(document)

    return success_response(
        "Document indexing status retrieved successfully",
        {
            "document_id": str(document.id),
            "is_vector_processed": document.is_vector_processed,
            "status": get_indexing_status_label(document, latest_job),
            "indexing_message": get_indexing_message(document, latest_job),
            "job": None if not latest_job else {
                "id": str(latest_job.id),
                "status": latest_job.status,
                "stage": latest_job.stage,
                "attempts": latest_job.attempts,
                "error_message": latest_job.error_message,
                "source_path": latest_job.source_path,
                "started_at": latest_job.started_at.isoformat() if latest_job.started_at else None,
                "finished_at": latest_job.finished_at.isoformat() if latest_job.finished_at else None,
                "created_at": latest_job.created_at.isoformat(),
                "updated_at": latest_job.updated_at.isoformat(),
            },
        },
    )


@api_view(['GET'])
@handle_exception
def list_documents(request):
    if not request.user.has_perm('document.can_view_document'):
        return error_response("Forbidden", status_code=403)
    """List all documents with basic info and pagination, with filtering and sorting."""
    department_id = request.query_params.get('department_id')
    type_id = request.query_params.get('type_id')
    title = request.query_params.get('title')
    description = request.query_params.get('description')  # For memos search
    sort = request.query_params.get('sort', '-created_at')
    is_active = request.query_params.get('is_active', 'true').lower() == 'true'
    documents = Document.objects.filter(
        is_active=is_active,
        is_latest_version=True
    ).select_related('department', 'type', 'parent_document').prefetch_related(INDEXING_JOBS_PREFETCH)
    year = request.query_params.get('year')
    if year:
        try:
            year_int = int(year)
            documents = documents.filter(effective_date__year=year_int)
        except ValueError:
            pass
    # Apply filters
    if department_id:
        documents = documents.filter(department_id=department_id)
    if type_id:
        documents = documents.filter(type_id=type_id)
    if title:
        documents = documents.filter(title__icontains=title.strip())
    if description:
        documents = documents.filter(description__icontains=description.strip())
    allowed_sorts = ['created_at', '-created_at', 'title', '-title', 'version', '-version']
    if sort in allowed_sorts:
        documents = documents.order_by(sort)
    else:
        documents = documents.order_by('-created_at')
    # Use large pagination for document management to show all documents
    from base.utils.api_pagination import LargeResultsSetPagination
    paginator = LargeResultsSetPagination()
    result_page = paginator.paginate_queryset(documents, request)
    serializer = DocumentListSerializer(result_page, many=True)
    paginated = paginator.get_paginated_response(serializer.data)
    return success_response(message="Documents retrieved successfully", data=paginated.data)


@api_view(['GET'])
@handle_exception
def list_document_suggestions(request):
    if not request.user.has_perm('document.can_view_document'):
        return error_response("Forbidden", status_code=403)

    documents = (
        Document.objects.filter(is_active=True, is_latest_version=True)
        .select_related('type')
        .only('id', 'title', 'original_filename', 'type__name')
        .order_by('title')
    )

    serializer = DocumentSuggestionSerializer(documents, many=True)
    return success_response(
        message="Document suggestions retrieved successfully",
        data=serializer.data,
    )

@api_view(['GET'])
@permission_classes([AllowAny])  # Allow token-based access without auth
def download_document(request, pk):
    """Serve document file for viewing or downloading based on user permission."""
    logger.info("Attempting to access document %s", pk)

    access = _resolve_document_access(request, pk, allow_token=True)
    document = access["document"]
    can_download = access["can_download"]
    force_secure_preview = access["force_secure_preview"]
    force_pdf_download = access["force_pdf_download"]
    token = access["token"]

    if force_secure_preview:
        logger.info("Serving secure preview for view-only access to document %s", pk)
        return _render_secure_preview(request, document, token=token)

    is_view_request = request.GET.get('view', 'false').lower() == 'true'
    force_refresh_pdf = request.GET.get('refresh', 'false').lower() == 'true'
    normalized_file_format = (document.file_format or '').strip().lower()
    if not normalized_file_format and document.original_filename:
        normalized_file_format = os.path.splitext(document.original_filename)[1].lstrip('.').lower()

    if not can_download:
        use_inline = True
        if not is_view_request:
            logger.warning("User without download permission attempted to download document %s", pk)
            raise Http404("You don't have permission to download this document")
    else:
        use_inline = is_view_request

    office_formats = ['doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'rtf', 'odt']

    if normalized_file_format in office_formats and (is_view_request or force_pdf_download):
        logger.info(
            "%s for Office document, preparing cached PDF version",
            "PDF download request" if force_pdf_download and not is_view_request else "View request",
        )
        pdf_path = document.pdf_view_path

        # Ensure PDF version exists using the current converter
        try:
            if ensure_pdf_version_exists(document, force_refresh=force_refresh_pdf):
                pdf_path = document.pdf_view_path
                logger.info(f"Serving PDF version from: {pdf_path}")
                try:
                    response = _build_pdf_file_response(
                        request,
                        document,
                        pdf_path,
                        as_attachment=force_pdf_download and not is_view_request,
                    )
                    logger.info(f"Successfully serving PDF file")
                    return response
                except Http404:
                    # Fall through to serve original file
                    pass
                except Exception as e:
                    logger.error(f"Unexpected error serving PDF file {pdf_path}: {type(e).__name__}: {e}")
                    # Fall through to serve original file
            else:
                logger.warning(f"Failed to create PDF version for viewing, serving original file")
        except Exception as e:
            logger.error(f"Error converting document to PDF: {str(e)}")
            logger.warning(f"Failed to create PDF version for viewing, serving original file")

    file_path = os.path.join(settings.MEDIA_ROOT, document.file_path)
    if not os.path.exists(file_path):
        logger.error(f"File not found at path: {file_path}")
        raise Http404("File not found")
    try:
        file_obj = open(file_path, 'rb')
    except (OSError, IOError) as e:
        logger.error(f"Error opening file {file_path}: {e}")
        raise Http404("File cannot be opened")
    
    # Map file format to content type
    content_type_map = {
        'pdf': 'application/pdf',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'txt': 'text/plain',
    }
    content_type = content_type_map.get(normalized_file_format, 'application/octet-stream')
    response = FileResponse(file_obj, content_type=content_type)

    disposition_type = 'inline' if use_inline else 'attachment'
    filename = quote(document.original_filename, safe='')
    response['Content-Disposition'] = f'{disposition_type}; filename="{filename}"; filename*=UTF-8\'\'{filename}'
    logger.info(f"Successfully serving original file: {document.original_filename} (disposition: {disposition_type})")
    response["Access-Control-Allow-Origin"] = request.headers.get('Origin', '*')
    response["Access-Control-Allow-Credentials"] = "true"
    response["X-Content-Type-Options"] = "nosniff"
    response["Content-Security-Policy"] = "default-src 'none'; script-src 'none'; object-src 'none';"
    return response


@api_view(['GET'])
@permission_classes([AllowAny])
def view_document_preview(request, pk):
    """Render the secure preview viewer for a document."""
    access = _resolve_document_access(request, pk, allow_token=True)
    if not access["force_secure_preview"]:
        raise Http404("Secure preview is only available for the User role")
    return _render_secure_preview(request, access["document"], token=access["token"])


@api_view(['GET'])
@permission_classes([AllowAny])
def view_document_preview_page(request, pk, page_number):
    """Serve a rendered preview image for a specific document page."""
    access = _resolve_document_access(request, pk, allow_token=True)
    if not access["force_secure_preview"]:
        raise Http404("Secure preview is only available for the User role")
    document = access["document"]

    if get_document_preview_mode(document) == "text":
        raise Http404("Text previews do not expose rendered pages")

    try:
        preview_image_path = ensure_preview_page_exists(document, page_number)
    except (FileNotFoundError, IndexError, OSError, ValueError) as exc:
        logger.error(
            "Failed to render preview page %s for document %s: %s",
            page_number,
            document.id,
            exc,
        )
        raise Http404("Preview page not found")

    try:
        file_obj = open(preview_image_path, "rb")
    except (OSError, IOError) as exc:
        logger.error("Error opening preview page %s for document %s: %s", page_number, document.id, exc)
        raise Http404("Preview page not found")

    response = FileResponse(file_obj, content_type="image/png")
    preview_filename = quote(
        f"{os.path.splitext(document.original_filename)[0]}_page_{page_number}.png",
        safe="",
    )
    response["Content-Disposition"] = (
        f'inline; filename="{preview_filename}"; filename*=UTF-8\'\'{preview_filename}'
    )
    response["Cache-Control"] = "private, max-age=300"
    response["Access-Control-Allow-Origin"] = request.headers.get('Origin', '*')
    response["Access-Control-Allow-Credentials"] = "true"
    response["X-Content-Type-Options"] = "nosniff"
    return response

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_download_token(request, pk):
    """Generate a one-time token for viewing or downloading documents."""
    from django.core import signing
    from django.utils import timezone
    
    logger.info(f"Generating access token for document: {pk}")
    
    try:
        document = Document.objects.get(id=pk, is_active=True)
    except Document.DoesNotExist:
        # Check if document exists but is inactive
        try:
            inactive_doc = Document.objects.get(id=pk, is_active=False)
            logger.warning(f"Document {pk} exists but is inactive/deleted.")
            return Response({
                "error": "This document has been deleted or is no longer available"
            }, status=status.HTTP_404_NOT_FOUND)
        except Document.DoesNotExist:
            logger.error(f"Document with pk {pk} not found in database.")
            return Response({
                "error": "Document not found"
            }, status=status.HTTP_404_NOT_FOUND)
    
    # Check if user has view OR download permission
    can_view = request.user.has_perm('document.can_view_document')
    can_download = request.user.has_perm('document.can_download_document')
    force_secure_preview = is_view_only_role_user(request.user) and not can_download
    force_pdf_download = is_pdf_download_role_user(request.user) and can_download
    
    if not can_view and not can_download:
        logger.warning(f"User {request.user} lacks view/download permission for document {pk}")
        return Response({"error": "You don't have permission to access this document"}, status=status.HTTP_403_FORBIDDEN)
    
    # Generate signed token valid for 5 minutes
    expires_at = int((timezone.now() + timezone.timedelta(minutes=5)).timestamp())
    token_data = {
        'document_id': str(pk),
        'user_id': str(request.user.id),  # Convert UUID to string
        'expires_at': expires_at,
        'can_download': can_download,  # Store download permission in token
        'force_secure_preview': force_secure_preview,
        'force_pdf_download': force_pdf_download,
    }
    token = signing.dumps(token_data)
    
    download_url = f"/api/documents/{pk}/download/?token={token}" if can_download else None
    view_url = (
        f"/api/documents/{pk}/download/?token={token}&view=true"
        if can_download
        else (
            f"/api/documents/{pk}/preview/?token={token}"
            if force_secure_preview
            else f"/api/documents/{pk}/download/?token={token}&view=true"
        )
    )
    
    return Response({
        "download_url": download_url,
        "view_url": view_url,
        "token": token,
        "expires_at": expires_at,
        "preview_only": force_secure_preview,
    }, status=status.HTTP_200_OK)

@api_view(['POST'])
@handle_exception
def upload_new_version(request, pk):
    if not request.user.has_perm('document.can_upload_document'):
        return error_response("Forbidden", status_code=403)
    """Upload a new version of existing document with optional metadata updates."""
    try:
        # Resolve full version family and work from the true root/latest.
        original_doc = get_object_or_404(Document, id=pk, is_active=True)
        root_document = _get_root_document(original_doc)
        latest_version = (
            _get_document_family_queryset(root_document, include_inactive=False)
            .order_by('-version', '-created_at')
            .first()
        ) or root_document
        if 'file' not in request.FILES:
            return error_response("No file provided", status_code=status.HTTP_400_BAD_REQUEST)
        uploaded_file = request.FILES['file']
        is_valid_file, file_error = _validate_docx_upload(uploaded_file.name, uploaded_file.content_type)
        if not is_valid_file:
            return error_response(file_error, status_code=status.HTTP_400_BAD_REQUEST)
        new_title = request.data.get('title', '').strip()
        description = request.data.get('description', latest_version.description)
        department_id = request.data.get('department_id')
        type_id = request.data.get('type_id')
        title = new_title if new_title else latest_version.title

        try:
            effective_date = (
                parse_document_date(
                    request.data.get('effective_date'),
                    field_name="effective_date",
                )
                if 'effective_date' in request.data
                else latest_version.effective_date
            )
            expiry_date = (
                parse_document_date(
                    request.data.get('expiry_date'),
                    field_name="expiry_date",
                    allow_blank=True,
                )
                if 'expiry_date' in request.data
                else latest_version.expiry_date
            )
            validate_document_date_range(effective_date, expiry_date)
        except ValueError as exc:
            return error_response(str(exc), status_code=status.HTTP_400_BAD_REQUEST)
        department = latest_version.department
        if department_id:
            try:
                department = Department.objects.get(id=department_id)
            except Department.DoesNotExist:
                return error_response("Invalid department", status_code=status.HTTP_400_BAD_REQUEST)
        doc_type = latest_version.type
        if type_id:
            try:
                doc_type = DocumentType.objects.get(id=type_id)
            except DocumentType.DoesNotExist:
                return error_response("Invalid document type", status_code=status.HTTP_400_BAD_REQUEST)
        file_ext = os.path.splitext(uploaded_file.name)[1]
        unique_name = f"{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
        file_path = f"documents/{unique_name}"
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
        except PermissionError as e:
            logger.error(f"Permission denied when writing file: {full_path}")
            raise DocumentError(f"Permission denied when writing file: {str(e)}")
        except (OSError, IOError) as e:
            logger.error(f"Error writing file: {full_path} - {e}")
            raise DocumentError(f"Failed to write file: {str(e)}")
        # Increment total_versions for all related documents
        root_document.total_versions += 1
        root_document.save()

        # Update total_versions for all versions in the family
        family_ids = _get_document_family_ids(root_document.id)
        Document.objects.filter(id__in=family_ids).update(total_versions=root_document.total_versions)

        # Mark all previous as not latest
        _get_document_family_queryset(root_document, include_inactive=False).update(is_latest_version=False)
        # Set version number
        highest_version = (
            _get_document_family_queryset(root_document, include_inactive=True)
            .aggregate(max_version=models.Max('version'))
            .get('max_version')
            or 0
        )

        first_name = request.user.first_name
        last_name = request.user.last_name

        new_version = Document.objects.create(
            title=title,
            department=department,
            type=doc_type,
            effective_date=effective_date,
            expiry_date=expiry_date,

            version=highest_version + 1,
            parent_document=latest_version,  # Always link to the tip of the branch
            description=description,
            original_filename=uploaded_file.name,
            stored_filename=unique_name,
            file_path=file_path,
            file_size=uploaded_file.size,
            file_format=file_ext[1:] if file_ext else '',
            publisher_id=request.user.id,  # Update with current user
            publisher_name=f"{first_name} {last_name}".strip() or request.user.username,  # Update with current user
            is_latest_version=True,
        )

        # Process the new version for vector DB
        try:
            indexing = enqueue_document_indexing(new_version)
            logger.info(f"Vector processing task queued for new version {new_version.id} with job {indexing.job.id}")
        except (OSError, IOError) as e:
            logger.error(f"Failed to queue vector processing for new version {new_version.id}: {str(e)}")
            raise DocumentError(f"Failed to process document: {str(e)}")

        data = {
            'id': str(new_version.id),
            'title': new_version.title,
            'version': new_version.version,
            'department': new_version.department.name if new_version.department else None,
            'document_type': new_version.type.name if new_version.type else None,
            'file_size': new_version.file_size_display,
            'description': new_version.description,
            'publisher_name': new_version.publisher_name,
            'is_latest_version': new_version.is_latest_version,
            'parent_document_id': str(new_version.parent_document.id) if new_version.parent_document else None,
        }
        return success_response(f"New version {new_version.version} uploaded successfully and is processing", data, status.HTTP_201_CREATED)
    except DocumentError:
        raise  # Re-raise custom exceptions
    except Document.DoesNotExist:
        raise DocumentNotFoundError(f"Document {pk} not found")
    except Department.DoesNotExist:
        raise APIValidationError("Invalid department specified")
    except DocumentType.DoesNotExist:
        raise APIValidationError("Invalid document type specified")
    except Exception as e:
        logger.exception("Failed to upload new version")
        raise InternalServerError(f"Failed to upload new version: {str(e)}")

@api_view(['PUT'])
@handle_exception
def update_document(request, pk):
    if not request.user.has_perm('document.change_document'):
        return error_response("Forbidden", status_code=403)

    """Update document metadata (title, department, type, is_active, etc.)."""
    document = get_object_or_404(Document, id=pk, is_active=True)
    
    # Store original is active status
    original_status = document.is_active

    serializer = DocumentUpdateSerializer(document, data=request.data, partial=True)
    if not serializer.is_valid():
        return error_response("Invalid data", serializer.errors, status.HTTP_400_BAD_REQUEST)
    serializer.save()

    # Check if is_active status field was changed
    new_status = document.is_active
    if 'is_active' in request.data and original_status != new_status:
        try:
            if new_status:
                # Activate document's chunks
                activate_document.delay(str(document.id))
                logger.info(f"Document activation task queued")
            else:
                # Deactivate document's chunks
                doc_type_name = document.type.name if document.type else "unknown"
                deactivate_document.delay(str(document.id), doc_type_name)
                logger.info(f"Document deactivation task queued")
        except Exception as e:
            logger.warning(f"Failed to queue document activation/deactivation task: {str(e)}")
            # Don't raise - task failure shouldn't break the update

    detail_serializer = DocumentDetailSerializer(document)
    return success_response("Document updated successfully", detail_serializer.data)

@api_view(['GET'])
@handle_exception
def get_document_versions(request, pk):
    """Get all active versions of a document, ordered with latest version first."""
    document = get_object_or_404(Document, id=pk)
    root_document = _get_root_document(document)
    versions = list(
        _get_document_family_queryset(root_document, include_inactive=False)
        .select_related('department', 'type', 'parent_document')
    )

    # Latest first, then highest version, then newest record.
    versions.sort(
        key=lambda v: (not v.is_latest_version, -float(v.version), v.created_at),
        reverse=False,
    )

    data = []
    for version in versions:
        data.append({
            'id': str(version.id),
            'title': version.title,
            'version': str(version.version),
            'file_size': version.file_size_display,
            'department': version.department.name if version.department else None,
            'document_type': version.type.name if version.type else None,
            'created_at': version.created_at.strftime('%Y-%m-%d %H:%M'),
            'effective_date': format_document_date(version.effective_date),
            'expiry_date': format_document_date(version.expiry_date),
            'is_latest': version.is_latest_version,
            'original_filename': version.original_filename,
            'description': version.description,
            'publisher_name': version.publisher_name,
            'parent_document_id': str(version.parent_document.id) if version.parent_document else None,
        })
    
    return success_response("Document versions retrieved successfully", data)



@api_view(['DELETE'])
@handle_exception
def delete_document(request, pk):
    """Delete a document (soft delete by default) with validation."""
    # Permission check
    if not request.user.has_perm('document.delete_document'):
        return error_response("Forbidden", status_code=403)
    
    # Validate query parameters
    try:
        query_params = {
            'hard': request.query_params.get('hard', 'false').lower() == 'true',
            'all_versions': request.query_params.get('all_versions', 'false').lower() == 'true'
        }
        validated_params = DocumentDeleteRequest(**query_params)
    except ValidationError as e:
        logger.warning(f"Validation error in delete_document: {e.errors()}")
        return error_response(
            "Invalid query parameters",
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=e.errors()
        )
    
    hard_delete = validated_params.hard
    delete_all_versions = validated_params.all_versions
    
    # Get document (allow both active and inactive)
    document = get_object_or_404(Document, id=pk)
    
    # AUTO-UPGRADE to hard delete if document is already archived (inactive)
    # Deleting from archive = permanent deletion
    if not document.is_active and not hard_delete:
        hard_delete = True
        logger.info(f"Auto-upgrading to hard delete for archived document ID: {document.id}")
    
    if delete_all_versions:
        # Delete all versions of this document family
        root_document = document.root_document
        family_all_qs = _get_document_family_queryset(root_document, include_inactive=True)
        
        if hard_delete:
            all_versions = family_all_qs
        else:
            # Only get active versions for soft delete
            all_versions = _get_document_family_queryset(root_document, include_inactive=False)
        
        if hard_delete:
            # Hard delete: Remove files and database records
            cleanup_targets = []
            for version in all_versions:
                doc_type_name = version.type.name if version.type else "unknown"
                cleanup_targets.append(
                    {
                        "document_id": str(version.id),
                        "file_path": version.file_path or "",
                        "doc_type_name": doc_type_name,
                    }
                )

            # Delete all versions from database
            deleted_count = all_versions.count()
            all_versions.delete()

            for target in cleanup_targets:
                try:
                    cleanup_deleted_document_artifacts.delay(
                        target["document_id"],
                        target["file_path"],
                        target["doc_type_name"],
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to queue permanent cleanup for document ID %s: %s",
                        target["document_id"],
                        str(e),
                    )
            
            return success_response(
                f"All {deleted_count} versions permanently deleted",
                {"deleted_count": deleted_count, "type": "hard_delete_all"}
            )
        else:
            # Soft delete: Mark all versions as inactive
            deleted_count = all_versions.update(is_active=False)

            for version in all_versions:
                try:
                    doc_type_name = version.type.name if version.type else "unknown"
                    deactivate_document.delay(str(version.id), doc_type_name)
                    logger.info(f"RAG deactivation task queued for document ID: {version.id}")
                except Exception as e:
                    logger.warning(f"Failed to queue RAG deactivation task for document ID: {version.id}: {str(e)}")
                    # Don't raise - task failure shouldn't break the deletion
            
            return success_response(
                f"All {deleted_count} versions marked as deleted", 
                {"deleted_count": deleted_count, "type": "soft_delete_all"}
            )
    
    else:
        # Delete only this specific version
        document_info = {
            "id": str(document.id),
            "title": document.title,
            "version": document.version,
            "was_soft_deleted": not document.is_active
        }
        was_latest = document.is_latest_version

        # Find the root document
        root_document = document
        while root_document.parent_document:
            root_document = root_document.parent_document

        # Store root_document ID for later use
        root_document_id = root_document.id
        family_ids = _get_document_family_ids(root_document_id)
        
        # Decrement total_versions for all related documents
        root_document.total_versions -= 1
        root_document.save()

        # Update total_versions for all versions in the family
        Document.objects.filter(id__in=family_ids).update(total_versions=root_document.total_versions)

        if hard_delete:
            doc_type_name = document.type.name if document.type else "unknown"
            cleanup_target = {
                "document_id": str(document.id),
                "file_path": document.file_path or "",
                "doc_type_name": doc_type_name,
            }

            Document.objects.filter(id=document.id).delete()

            try:
                cleanup_deleted_document_artifacts.delay(
                    cleanup_target["document_id"],
                    cleanup_target["file_path"],
                    cleanup_target["doc_type_name"],
                )
            except Exception as e:
                logger.warning(
                    "Failed to queue permanent cleanup for document ID %s: %s",
                    cleanup_target["document_id"],
                    str(e),
                )
            
            # If this was the latest version, update the next highest version
            if was_latest:
                remaining_versions = Document.objects.filter(
                    id__in=family_ids,
                    is_active=True
                )
                if remaining_versions.exists():
                    new_latest = remaining_versions.order_by('-version').first()
                    new_latest.is_latest_version = True
                    new_latest.save()
                    
            status_message = "permanently deleted from trash" if document_info["was_soft_deleted"] else "permanently deleted"
            
            return success_response(
                f"Document version {document_info['version']} {status_message}",
                {**document_info, "type": "hard_delete_single"}
            )
        else:
            # Soft delete: Mark as inactive
            if not document.is_active:
                return error_response("Document is already deleted", status_code=status.HTTP_400_BAD_REQUEST)
            
            document.is_active = False
            document.save()

            try:
                # Get document type name for RAG deactivation
                doc_type_name = document.type.name if document.type else "unknown"
                deactivate_document.delay(str(document.id), doc_type_name)
                logger.info(f"RAG deactivation task queued for document ID: {document.id}")
            except Exception as e:
                logger.warning(f"Failed to queue RAG deactivation task for document ID: {document.id}: {str(e)}")
                # Don't raise - task failure shouldn't break the deletion
            
            # If this was the latest version, update the next highest version
            if was_latest and root_document:
                remaining_versions = Document.objects.filter(
                    id__in=family_ids,
                    is_active=True
                ).exclude(id=document.id)
                
                if remaining_versions.exists():
                    new_latest = remaining_versions.order_by('-version').first()
                    new_latest.is_latest_version = True
                    new_latest.save()
                    document.is_latest_version = False
                    document.save()
            
            return success_response(
                f"Document version {document.version} marked as deleted", 
                {**document_info, "type": "soft_delete_single"}
            )

class DocumentDetailView(generics.RetrieveAPIView):
    queryset = Document.objects.all()
    serializer_class = DocumentDetailSerializer

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return success_response("Document retrieved successfully", serializer.data)

@api_view(['POST'])
@handle_exception
def restore_document(request, pk):
    """Restore a soft-deleted document (set is_active=True)."""
    document = get_object_or_404(Document, id=pk, is_active=False)
    document.is_active = True

    root_document = _get_root_document(document)
    active_versions = list(_get_document_family_queryset(root_document, include_inactive=False))

    # If no other active version has a higher version number, set as latest
    has_newer = any(Decimal(str(v.version)) > Decimal(str(document.version)) for v in active_versions if v.id != document.id)
    if not has_newer:
        # Set all others to not latest
        _get_document_family_queryset(root_document, include_inactive=False).filter(
            is_latest_version=True
        ).exclude(id=document.id).update(is_latest_version=False)
        document.is_latest_version = True
    else:
        document.is_latest_version = False

    document.save()
    # Activate document chunks in RAG system
    try:
        activate_document.delay(str(document.id))
        logger.info(f"Document activation task queued for document ID: {document.id}")
    except Exception as e:
        logger.warning(f"Failed to queue document activation task for document ID {document.id}: {str(e)}")
        # Don't raise - restore should succeed even if async activation is temporarily unavailable

    detail_serializer = DocumentDetailSerializer(document)
    return success_response("Document restored successfully", detail_serializer.data)

@api_view(['PATCH'])
@handle_exception
def update_document_version(request, pk):
    """
    Manually update the version number and other fields of a document, including file upload.
    Ensures no conflict with other versions under the same root document.
    Sets this document as the latest version.
    """
    document = get_object_or_404(Document, id=pk)
    data = request.data.copy()
    new_version = data.get('version')
    if not new_version:
        return error_response("A valid 'version' is required.", status_code=status.HTTP_400_BAD_REQUEST)
    try:
        new_version = Decimal(str(new_version))
    except (InvalidOperation, ValueError):
        return error_response("Version must be a number (e.g., 2, 2.1, 5).", status_code=status.HTTP_400_BAD_REQUEST)
    # Check for version conflict
    root_doc = document.root_document
    family_qs = _get_document_family_queryset(root_doc, include_inactive=True)
    if family_qs.filter(version=new_version).exclude(id=document.id).exists():
        return error_response(f"Version {new_version} already exists for this document family.", status_code=status.HTTP_400_BAD_REQUEST)
    # Set previous latest to not latest
    family_qs.filter(is_latest_version=True).exclude(id=document.id).update(is_latest_version=False)
    # Set parent to previous latest (if not already)
    previous_latest = family_qs.filter(is_latest_version=False).exclude(id=document.id).order_by('-created_at').first()
    if previous_latest:
        document.parent_document = previous_latest
    document.version = new_version
    document.is_latest_version = True

    # --- Handle file upload if present ---
    if 'file' in request.FILES:
        uploaded_file = request.FILES['file']
        is_valid_file, file_error = _validate_docx_upload(uploaded_file.name, uploaded_file.content_type)
        if not is_valid_file:
            return error_response(file_error, status_code=status.HTTP_400_BAD_REQUEST)
        from .utils import generate_unique_filename
        file_ext = os.path.splitext(uploaded_file.name)[1]
        unique_name = generate_unique_filename(uploaded_file)
        file_path = f"documents/{unique_name}"
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
        except (OSError, IOError) as e:
            logger.error(f"Error writing file: {full_path} - {e}")
            raise DocumentError(f"Failed to write file: {str(e)}")
        # If document was previously processed, deactivate old embeddings
        if document.is_vector_processed:
            try:
                deactivate_document.delay(str(document.id), document.type.name if document.type else "unknown")
                logger.info(f"Deactivated old embeddings for document {document.id}")
            except Exception as e:
                logger.error(f"Failed to deactivate old embeddings for document {document.id}: {str(e)}")
                raise DocumentError(f"Failed to process document update: {str(e)}")

        document.original_filename = uploaded_file.name
        document.stored_filename = unique_name
        document.file_path = file_path
        document.file_size = uploaded_file.size
        document.file_format = file_ext[1:] if file_ext else ''
        # Mark as not processed yet, will be updated after processing
        document.is_vector_processed = False
    # --- End file upload ---

    # Update other fields if provided
    updatable_fields = ['title', 'description', 'department_id', 'type_id', 'effective_date', 'expiry_date', 'publisher_id', 'publisher_name', 'memos']
    for field in updatable_fields:
        if field in data:
            if field == 'memos':
                document.description = data[field]
            elif field == 'effective_date':
                if data[field]:
                    try:
                        document.effective_date = parse_document_date(
                            data[field],
                            field_name="effective_date",
                        )
                    except ValueError as exc:
                        raise APIValidationError(str(exc))
            elif field == 'expiry_date':
                try:
                    document.expiry_date = parse_document_date(
                        data[field],
                        field_name="expiry_date",
                        allow_blank=True,
                    )
                except ValueError as exc:
                    raise APIValidationError(str(exc))
            else:
                setattr(document, field, data[field])

    try:
        validate_document_date_range(document.effective_date, document.expiry_date)
    except ValueError as exc:
        raise APIValidationError(str(exc))

    document.save()

    # If a file was uploaded, process it for vector DB
    if 'file' in request.FILES:
        try:
            indexing = enqueue_document_indexing(document)
            logger.info(f"Vector processing task queued for updated document {document.id} with job {indexing.job.id}")
        except (OSError, IOError) as e:
            logger.error(f"Failed to queue vector processing for document {document.id}: {str(e)}")
            raise DocumentError(f"Failed to process document: {str(e)}")

    detail_serializer = DocumentDetailSerializer(document)
    return success_response("Document version updated successfully", detail_serializer.data)


@api_view(['GET'])
@handle_exception
def get_unique_years(request):
    """Returns a list of unique years from the effective_date of all documents (active and inactive)."""
    years = Document.objects.filter(
        effective_date__isnull=False
    ).annotate(
        year=models.functions.ExtractYear('effective_date')
    ).values_list('year', flat=True).distinct().order_by('-year')

    # Convert years to string for consistency with frontend Option type
    year_data = [{"id": str(y), "year": y} for y in years]
    return success_response(message="Unique years retrieved successfully", data=year_data)

@api_view(['GET'])
@handle_exception
def view_document_pdf(request, pk):
    """Serve PDF version of document directly for viewing."""
    document = get_object_or_404(Document, id=pk, is_active=True)

    can_view = request.user.has_perm('document.can_view_document')
    can_download = request.user.has_perm('document.can_download_document')
    download_requested = request.GET.get("download", "false").lower() == "true" and can_download
    if not can_view and not can_download:
        return error_response("Forbidden", status_code=403)

    if is_view_only_role_user(request.user) and not can_download:
        return _render_secure_preview(request, document)

    try:
        normalized_file_format = (document.file_format or "").strip().lower()
        if not normalized_file_format and document.original_filename:
            normalized_file_format = os.path.splitext(document.original_filename)[1].lstrip(".").lower()

        if normalized_file_format == "pdf":
            pdf_path = document.full_file_path
        else:
            if not ensure_pdf_version_exists(document):
                return error_response("Failed to generate PDF for viewing", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
            pdf_path = document.pdf_view_path

        try:
            return _build_pdf_file_response(
                request,
                document,
                pdf_path,
                as_attachment=download_requested,
            )
        except Http404:
            return error_response("File cannot be opened", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


    except (OSError, IOError) as e:
        logger.error(f"Error serving PDF for document {pk}: {str(e)}")
        raise DocumentError(f"Error serving PDF: {str(e)}")

@api_view(['GET'])
@handle_exception
def get_document_pdf_url(request, pk):
    """Get or create PDF version for document viewing and return the URL."""
    document = get_object_or_404(Document, id=pk, is_active=True)

    can_view = request.user.has_perm('document.can_view_document')
    can_download = request.user.has_perm('document.can_download_document')
    if not can_view and not can_download:
        return error_response("Forbidden", status_code=403)

    try:
        return success_response(
            "Document viewing URL generated successfully",
            {
                "pdf_view_url": reverse("document:view_document_pdf", args=[document.id]),
                "preview_only": is_view_only_role_user(request.user) and not can_download,
            },
        )
    except Exception as e:
        logger.error(f"Error generating document view URL for document {pk}: {str(e)}")
        raise DocumentError(f"Error generating document view URL: {str(e)}")

@api_view(['GET'])
@handle_exception
def list_inactive_documents(request):
    """List all inactive (soft-deleted) documents with pagination (take/skip) and filtering."""
    department_id = request.query_params.get('department_id')
    type_id = request.query_params.get('type_id')
    title = request.query_params.get('title')
    description = request.query_params.get('description')  # For memos search
    sort = request.query_params.get('sort', '-created_at')
    take = int(request.query_params.get('take', 10))
    skip = int(request.query_params.get('skip', 0))

    documents = Document.objects.filter(
        is_active=False
    ).select_related('department', 'type', 'parent_document').prefetch_related(INDEXING_JOBS_PREFETCH)

    year = request.query_params.get('year')
    if year:
        try:
            year_int = int(year)
            documents = documents.filter(effective_date__year=year_int)
        except ValueError:
            pass

    if department_id:
        documents = documents.filter(department_id=department_id)
    if type_id:
        documents = documents.filter(type_id=type_id)
    if title:
        documents = documents.filter(title__icontains=title.strip())

    allowed_sorts = ['created_at', '-created_at', 'title', '-title', 'version', '-version']
    if sort in allowed_sorts:
        documents = documents.order_by(sort)
    else:
        documents = documents.order_by('-created_at')

    total_count = documents.count()
    documents = documents[skip:skip+take]

    serializer = DocumentListSerializer(documents, many=True)
    
    return success_response(
        message="Inactive documents retrieved successfully",
        data={
            "count": total_count,
            "results": serializer.data
        }
    )
