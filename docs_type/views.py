from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view
from rest_framework import status
from .models import DocumentType
from base.utils import success_response, error_response, handle_exception
import logging

logger = logging.getLogger(__name__)

@api_view(['GET'])
@handle_exception
def list_document_types(request):
    """List all document types for dropdowns."""
    document_types = DocumentType.objects.all().order_by('name')
    data = [{'id': str(dt.id), 'name': dt.name, 'presentation': dt.presentation} for dt in document_types]
    return success_response("Document types retrieved successfully", data)

@api_view(['POST'])
@handle_exception
def create_document_type(request):
    """Create new document type (memo, policy, contract, etc.)."""
    name = request.data.get('name', '').strip()
    
    if not name:
        return error_response("Name is required", status_code=status.HTTP_400_BAD_REQUEST)
    
    if DocumentType.objects.filter(name__iexact=name).exists():
        return error_response("Document type already exists", status_code=status.HTTP_400_BAD_REQUEST)
    
    doc_type = DocumentType.objects.create(name=name)
    data = {'id': str(doc_type.id), 'name': doc_type.name}
    return success_response("Document type created successfully", data, status.HTTP_201_CREATED)

@api_view(['PUT'])
@handle_exception
def update_document_type(request, type_id):
    """Update document type name."""
    doc_type = get_object_or_404(DocumentType, id=type_id)
    name = request.data.get('name', '').strip()
    
    if not name:
        return error_response("Name is required", status_code=status.HTTP_400_BAD_REQUEST)
    
    if DocumentType.objects.filter(name__iexact=name).exclude(id=type_id).exists():
        return error_response("Document type name already exists", status_code=status.HTTP_400_BAD_REQUEST)
    
    doc_type.name = name
    doc_type.save()
    
    data = {'id': str(doc_type.id), 'name': doc_type.name}
    return success_response("Document type updated successfully", data)

@api_view(['DELETE'])
@handle_exception
def delete_document_type(request, type_id):
    """Delete document type (only if no documents)."""
    doc_type = get_object_or_404(DocumentType, id=type_id)
    
    # Check for documents
    doc_count = doc_type.documents_set.filter(is_active=True).count()
    if doc_count > 0:
        return error_response(f"Cannot delete. {doc_count} documents use this type.", status_code=status.HTTP_400_BAD_REQUEST)
    
    doc_type.delete()
    return success_response("Document type deleted successfully")







