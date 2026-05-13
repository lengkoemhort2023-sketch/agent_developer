from django.core import signing
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

import mimetypes
import os
import logging

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from django.http import FileResponse, Http404, HttpResponseForbidden
from django.conf import settings
from django.contrib.auth.decorators import login_required
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from base.utils import success_response, error_response
import ldap
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


def _restore_missing_docx_inline_image(normalized_path: str) -> bool:
    path_parts = [segment for segment in normalized_path.split("/") if segment]
    if len(path_parts) < 3 or path_parts[0] != "docx_extracted_images":
        return False

    file_id = path_parts[1]
    target_file_path = os.path.join(settings.MEDIA_ROOT, normalized_path)
    if os.path.exists(target_file_path):
        return True

    try:
        from document.models import Document
        from base.services.agent.rag.services.extraction.docx_extraction import _persist_docx_inline_images
    except ImportError as exc:
        logger.warning("Unable to import DOCX extraction dependencies: %s", exc)
        return False

    document = Document.objects.filter(id=file_id).first()
    if not document:
        return False

    full_file_path = getattr(document, "full_file_path", None)
    if not full_file_path or not os.path.exists(full_file_path):
        return False

    file_format = str(getattr(document, "file_format", "") or "").strip().lower()
    if file_format not in {"docx", "doc"}:
        return False

    try:
        with open(full_file_path, "rb") as doc_file:
            file_bytes = doc_file.read()
        restore_result = _persist_docx_inline_images(file_bytes, file_id)
    except OSError as exc:
        logger.warning("Failed reading source DOCX for image restore (%s): %s", file_id, exc)
        return False
    except Exception as exc:
        logger.warning("Failed restoring inline DOCX images for %s: %s", file_id, exc)
        return False

    if not restore_result.get("saved"):
        return False

    return os.path.exists(target_file_path)


def protected_media(request, path, document_root=None):
    from document.models import Document
    from document.utils import get_pdf_version_path, get_preview_image_directory, is_view_only_role_user

    filename = path.lstrip("/")
    if filename.startswith("protected/"):
        filename = filename[len("protected/"):]

    document = None
    if filename.startswith('documents/'):
        if not request.user.is_authenticated:
            raise Http404()

        can_view = request.user.has_perm("document.can_view_document")
        can_download = request.user.has_perm("document.can_download_document")
        if not can_view and not can_download:
            raise Http404()

        name_part = filename[10:]

        document = Document.objects.filter(file_path=filename, is_active=True).first()
        if not document:
            document = Document.objects.filter(stored_filename=name_part, is_active=True).first()
        if not document:
            try:
                document = Document.objects.get(original_filename=name_part, is_active=True)
            except Document.DoesNotExist:
                pass
        if not document:
            document = Document.objects.filter(title=name_part, is_active=True).order_by('-created_at').first()

        if not document:
            raise Http404()

        pdf_preview_path = get_pdf_version_path(document)
        preview_directory = get_preview_image_directory(document)
        pdf_preview_relative = (
            os.path.relpath(pdf_preview_path, settings.MEDIA_ROOT)
            if pdf_preview_path and os.path.exists(pdf_preview_path)
            else None
        )
        preview_directory_relative = (
            os.path.relpath(preview_directory, settings.MEDIA_ROOT)
            if preview_directory and os.path.exists(preview_directory)
            else None
        )

        is_original_document = filename == document.file_path
        is_pdf_preview = pdf_preview_relative == filename
        is_rendered_preview = (
            preview_directory_relative is not None
            and filename.startswith(preview_directory_relative.rstrip("/") + "/")
        )

        if is_original_document and is_view_only_role_user(request.user) and not can_download:
            raise Http404()

        if (
            not is_original_document
            and not (is_pdf_preview or is_rendered_preview)
            and is_view_only_role_user(request.user)
            and not can_download
        ):
            raise Http404()

        file_path = os.path.join(settings.MEDIA_ROOT, filename)
    else:
        file_path = os.path.join(settings.MEDIA_ROOT, filename)

    if not os.path.exists(file_path):
        raise Http404()

    try:
        file_obj = open(file_path, 'rb')
    except (OSError, IOError):
        raise Http404()

    content_type_map = {
        'pdf': 'application/pdf',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'txt': 'text/plain',
    }
    if filename.startswith('documents/') and document:
        content_type = content_type_map.get(document.file_format, 'application/octet-stream')
    else:
        guessed_content_type, _ = mimetypes.guess_type(filename)
        if filename.startswith('voice_inputs/'):
            if guessed_content_type in {None, 'video/webm'}:
                guessed_content_type = 'audio/webm'
        content_type = guessed_content_type or 'application/octet-stream'

    response = FileResponse(file_obj, content_type=content_type)
    response['X-Accel-Redirect'] = f"/media/protected/{filename}"
    # Add CORS headers for media files
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


def docx_extracted_media(request, path, document_root=None):
    normalized_path = f"docx_extracted_images/{path.lstrip('/')}"
    target_path = os.path.join(settings.MEDIA_ROOT, normalized_path)

    if not os.path.exists(target_path):
        _restore_missing_docx_inline_image(normalized_path)

    return protected_media(request, normalized_path, document_root=document_root)

@api_view(["GET"])
@permission_classes([AllowAny])
def ldap_health_check(request):
    """Endpoint to check LDAP server health.

    Attempts to resolve configured LDAP URI(s) and bind to each resolved IP so
    we can detect DNS/multi-A-record issues and provide clearer diagnostics.
    """
    LDAP_SERVER = getattr(settings, "AUTH_LDAP_SERVER_URI", None)
    LDAP_USER = getattr(settings, "AUTH_LDAP_BIND_DN", None)
    LDAP_PASSWORD = getattr(settings, "AUTH_LDAP_BIND_PASSWORD", None)

    import socket
    import urllib.parse

    if not LDAP_SERVER:
        return error_response(
            message="LDAP connection failed",
            data={"error": "No LDAP server configured"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # Support space-separated multiple URIs or a single URI
    parts = LDAP_SERVER.split()
    targets = []  # list of (ip, port, url)
    for part in parts:
        parsed = urllib.parse.urlparse(part)
        host = parsed.hostname or part
        port = parsed.port or 389
        try:
            addrs = socket.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_STREAM)
            ips = list({ai[4][0] for ai in addrs})
        except Exception:
            ips = [host]
        for ip in ips:
            targets.append((ip, port, f"ldap://{ip}:{port}"))

    errors = []
    for ip, port, url in targets:
        try:
            conn = ldap.initialize(url)
            conn.set_option(ldap.OPT_NETWORK_TIMEOUT, 2)
            conn.set_option(ldap.OPT_PROTOCOL_VERSION, ldap.VERSION3)
            conn.simple_bind_s(LDAP_USER, LDAP_PASSWORD)
            conn.unbind_s()
            return success_response(message="LDAP connection OK", data={"status": "up", "connected": url})
        except ldap.LDAPError as e:
            errors.append({"server": url, "error": str(e)})
        except Exception as e:
            errors.append({"server": url, "error": str(e)})

    return error_response(
        message="LDAP connection failed",
        data={"errors": errors},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    )

@api_view(["GET"])
def qdrant_health_check(request):
    """Endpoint to check Qdrant server health with timeout."""
    try:
        host = os.environ.get("QDRANT_HOST", "")
        port = int(os.environ.get("QDRANT_PORT", "0"))
        timeout_seconds = int(os.environ.get("QDRANT_TIMEOUT", "3"))

        client = QdrantClient(host=host, port=port)

        # Run health check with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(client.get_collections)
            collections = future.result(timeout=timeout_seconds)

        return success_response(
            message="Qdrant connection OK",
            data={"status": "up", "collections": len(collections.collections)}
        )

    except TimeoutError:
        return error_response(
            message="Qdrant health check timed out",
            data={"status": "timeout"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception as e:
        return error_response(
            message="Qdrant connection failed",
            data={"error": str(e)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
# For temporary view file
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def get_signed_download_url(request):
    file_path = request.data.get("file_path")
    expires_in = int(request.data.get("expires_in", 300))
    if not file_path:
        return Response({"error": "file_path required"}, status=status.HTTP_400_BAD_REQUEST)
    url = generate_signed_url(file_path, expires_in)
    return Response({"download_url": url}, status=status.HTTP_200_OK)

def generate_signed_url(file_path, expires_in=300):
    expires_at = int((timezone.now() + timezone.timedelta(seconds=expires_in)).timestamp())
    value = signing.dumps({'file_path': file_path, 'expires_at': expires_at})
    return f"/download/?token={value}"

def download_file(request):
    token = request.GET.get('token')
    if not token:
        raise Http404()
    try:
        data = signing.loads(token)
        file_path = data['file_path']
        expires_at = data['expires_at']
        if timezone.now().timestamp() > expires_at:
            raise Http404("Link expired")
        abs_path = os.path.join(settings.MEDIA_ROOT, file_path)
        if not os.path.exists(abs_path):
            raise Http404()
        response = FileResponse(open(abs_path, 'rb'))
        # Add CORS headers for download files
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    except Exception:
        raise Http404()
