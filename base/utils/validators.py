"""
Type hints and validation utilities for API requests and responses
Provides consistent validation and error handling
"""

from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from functools import wraps
import logging
from uuid import UUID

from django.http import HttpRequest, JsonResponse
from rest_framework import status
from rest_framework.response import Response
from pydantic import BaseModel, ValidationError, Field, validator, field_validator

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# REQUEST VALIDATION SCHEMAS
# ============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration"""
    
    class Config:
        extra = 'forbid'  # Reject unknown fields
        use_enum_values = True


class ChatMessageRequest(BaseSchema):
    """Schema for chat message creation"""
    question: str = Field(default='', max_length=5000)
    session_id: Optional[str] = Field(None, pattern='^[0-9a-f-]*$')
    file_type: Optional[str] = Field(None, max_length=50)
    document_type: Optional[str] = Field(None, max_length=50)
    input_type: str = Field(default='text', pattern='^(text|voice|file)$')
    voice_file: Optional[Any] = Field(None)  # Allow voice file uploads
    
    class Config:
        extra = 'ignore'  # Ignore extra fields instead of forbid


    def __init__(self, **data):
        if (not data.get("file_type")) and data.get("document_type"):
            data["file_type"] = data["document_type"]
        super().__init__(**data)


class ChatSessionListParams(BaseSchema):
    """Schema for chat session list query parameters"""
    title: Optional[str] = Field(None, max_length=255)
    archived: bool = Field(default=False)
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ChatHistoryParams(BaseSchema):
    """Schema for chat history pagination"""
    take: int = Field(default=10, ge=1, le=100)
    skip: int = Field(default=0, ge=0)


class DocumentUploadRequest(BaseSchema):
    """Schema for document upload"""
    title: str = Field(..., min_length=1)
    document_type_id: Optional[UUID] = Field(None)
    type_id: Optional[UUID] = Field(None)
    department_id: Optional[UUID] = Field(None)
    expiry_date: Optional[str] = Field(None)
    effective_date: Optional[str] = Field(None)
    memos: Optional[str] = Field(None, max_length=5000)
    # Extra fields from frontend form (currently unused but accepted)
    code: Optional[str] = Field(None, max_length=50)
    version: Optional[str] = Field(None, max_length=50)
    unit: Optional[str] = Field(None, max_length=100)
    branch: Optional[str] = Field(None, max_length=100)
    
    @field_validator('type_id', 'department_id', 'document_type_id', mode='before')
    @classmethod
    def validate_uuids(cls, v):
        if v is None or v == '':
            return None
        if isinstance(v, UUID):
            return v
        if isinstance(v, str):
            try:
                return UUID(v)
            except (ValueError, TypeError):
                raise ValueError(f'Invalid UUID format')
        raise ValueError(f'UUID must be a string or UUID object')
    
    def __init__(self, **data):
        """Override init to copy document_type_id to type_id if needed"""
        # If type_id is not provided but document_type_id is, use it
        if 'type_id' not in data or data.get('type_id') is None:
            if 'document_type_id' in data and data['document_type_id'] is not None:
                data['type_id'] = data['document_type_id']
        super().__init__(**data)


class DocumentDeleteRequest(BaseSchema):
    """Schema for document delete query params"""
    hard: bool = Field(default=False)
    all_versions: bool = Field(default=False)


class DocumentListParams(BaseSchema):
    """Schema for document list query parameters"""
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    title: Optional[str] = Field(None, max_length=255)
    department_id: Optional[int] = Field(None, ge=1)
    type_id: Optional[int] = Field(None, ge=1)
    sort_by: Optional[str] = Field(None, pattern='^(created_at|updated_at|title)$')


class UserLoginRequest(BaseSchema):
    """Schema for user login"""
    username_or_email: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=255)


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class PaginationSchema(BaseSchema):
    """Pagination metadata"""
    page: int = Field(..., ge=1)
    page_size: int = Field(..., ge=1, le=100)
    total_count: int = Field(..., ge=0)
    total_pages: int = Field(..., ge=0)


class APIErrorSchema(BaseSchema):
    """Standard error response"""
    status: int
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class APISuccessSchema(BaseSchema):
    """Standard success response"""
    status: int
    message: str
    body: Optional[Any] = None
    pagination: Optional[PaginationSchema] = None


# ============================================================================
# VALIDATION DECORATORS
# ============================================================================

def validate_request_body(schema_class: type[BaseSchema]) -> Callable:
    """
    Decorator to validate request body against Pydantic schema
    
    Usage:
        @validate_request_body(ChatMessageRequest)
        def create_message(request, validated_data):
            ...
    """
    def decorator(view_func: Callable) -> Callable:
        @wraps(view_func)
        def wrapper(request: HttpRequest, *args, **kwargs):
            try:
                import json
                body = json.loads(request.body) if request.body else {}
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in request: {e}")
                return Response(
                    {
                        "status": status.HTTP_400_BAD_REQUEST,
                        "message": "Invalid JSON in request body",
                        "error_code": "INVALID_JSON"
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            try:
                validated_data = schema_class(**body)
                kwargs['validated_data'] = validated_data
            except ValidationError as e:
                logger.warning(f"Validation error: {e.errors()}")
                return Response(
                    {
                        "status": status.HTTP_400_BAD_REQUEST,
                        "message": "Validation error",
                        "error_code": "VALIDATION_ERROR",
                        "details": e.errors()
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            return view_func(request, *args, **kwargs)
        
        return wrapper
    return decorator


def validate_query_params(schema_class: type[BaseSchema]) -> Callable:
    """
    Decorator to validate query parameters against Pydantic schema
    
    Usage:
        class SearchParams(BaseSchema):
            q: Optional[str] = None
            limit: int = Field(default=10, le=100)
        
        @validate_query_params(SearchParams)
        def search(request, validated_params):
            ...
    """
    def decorator(view_func: Callable) -> Callable:
        @wraps(view_func)
        def wrapper(request: HttpRequest, *args, **kwargs):
            try:
                validated_params = schema_class(**request.GET.dict())
                kwargs['validated_params'] = validated_params
            except ValidationError as e:
                logger.warning(f"Query parameter validation error: {e.errors()}")
                return Response(
                    {
                        "status": status.HTTP_400_BAD_REQUEST,
                        "message": "Invalid query parameters",
                        "error_code": "INVALID_PARAMS",
                        "details": e.errors()
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            return view_func(request, *args, **kwargs)
        
        return wrapper
    return decorator


# ============================================================================
# RESPONSE BUILDERS
# ============================================================================

def success_response(
    message: str = "Success",
    body: Optional[Any] = None,
    status_code: int = status.HTTP_200_OK,
    pagination: Optional[Dict[str, int]] = None,
) -> Response:
    """Build standardized success response"""
    response_data = {
        "status": status_code,
        "message": message,
        "body": body,
    }
    
    if pagination:
        response_data["pagination"] = pagination
    
    return Response(response_data, status=status_code)


def error_response(
    message: str,
    error_code: Optional[str] = None,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    details: Optional[Dict[str, Any]] = None,
) -> Response:
    """Build standardized error response"""
    response_data = {
        "status": status_code,
        "message": message,
        "error_code": error_code,
    }
    
    if details:
        response_data["details"] = details
    
    return Response(response_data, status=status_code)


# ============================================================================
# COMMON VALIDATORS
# ============================================================================

class FileValidationSchema(BaseSchema):
    """Validate file uploads"""
    max_size_mb: int = 100
    allowed_types: List[str] = ['application/pdf']
    
    @validator('max_size_mb')
    def validate_max_size(cls, v: int) -> int:
        if v < 1 or v > 1000:
            raise ValueError('max_size_mb must be between 1 and 1000')
        return v


def validate_file_upload(
    file_obj: Any,
    max_size_mb: int = 100,
    allowed_types: Optional[List[str]] = None,
) -> tuple[bool, Optional[str]]:
    """
    Validate file upload
    
    Args:
        file_obj: Django UploadedFile object
        max_size_mb: Maximum file size in MB
        allowed_types: List of allowed MIME types
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if allowed_types is None:
        allowed_types = ['application/pdf', 'application/octet-stream']
    
    # Check file size
    max_bytes = max_size_mb * 1024 * 1024
    if file_obj.size > max_bytes:
        return False, f"File too large. Maximum size: {max_size_mb}MB"
    
    # Check file type
    if file_obj.content_type not in allowed_types:
        return False, f"File type not allowed. Allowed types: {', '.join(allowed_types)}"
    
    return True, None
