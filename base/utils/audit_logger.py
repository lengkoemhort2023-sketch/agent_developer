"""
Audit logging for sensitive operations and security events
"""

import logging
import json
from typing import Any, Dict, Optional
from datetime import datetime
from django.contrib.auth.models import User
from django.http import HttpRequest

logger = logging.getLogger(__name__)


class AuditLogger:
    """Log all sensitive operations for compliance and security"""
    
    class EventType:
        # User actions
        USER_LOGIN = "user.login"
        USER_LOGIN_FAILED = "user.login_failed"
        USER_LOGOUT = "user.logout"
        USER_CREATED = "user.created"
        USER_MODIFIED = "user.modified"
        USER_DELETED = "user.deleted"
        
        # Document operations
        DOCUMENT_UPLOADED = "document.uploaded"
        DOCUMENT_VIEWED = "document.viewed"
        DOCUMENT_DOWNLOADED = "document.downloaded"
        DOCUMENT_DELETED = "document.deleted"
        DOCUMENT_SHARED = "document.shared"
        
        # Chat operations
        CHAT_SESSION_CREATED = "chat.session_created"
        CHAT_MESSAGE_SENT = "chat.message_sent"
        CHAT_CLEARED = "chat.cleared"
        
        # Admin operations
        ADMIN_ACTION = "admin.action"
        CONFIG_CHANGED = "config.changed"
        
        # Security events
        PERMISSION_DENIED = "security.permission_denied"
        SUSPICIOUS_ACTIVITY = "security.suspicious_activity"
    
    @staticmethod
    def log_event(
        event_type: str,
        user: Optional[User] = None,
        request: Optional[HttpRequest] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status: str = "success",
    ) -> None:
        """
        Log a security-relevant event with full context.
        
        Args:
            event_type: Type of event (use EventType constants)
            user: User performing the action
            request: HTTP request object (for IP, user agent)
            resource_id: ID of affected resource
            resource_type: Type of affected resource (user, document, chat, etc)
            details: Additional context dict
            status: success, failed, denied
        """
        event_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "status": status,
            "user": {
                "id": user.id if user else None,
                "username": user.username if user else None,
            } if user else None,
            "request": {
                "ip": request.META.get('REMOTE_ADDR') if request else None,
                "user_agent": request.META.get('HTTP_USER_AGENT') if request else None,
                "path": request.path if request else None,
                "method": request.method if request else None,
            } if request else None,
            "resource": {
                "type": resource_type,
                "id": resource_id,
            } if resource_type or resource_id else None,
            "details": details or {},
        }
        
        # Log as JSON for easy parsing
        log_level = logging.WARNING if status != "success" else logging.INFO
        logger.log(log_level, json.dumps(event_log))
    
    @staticmethod
    def log_user_login(user: User, request: HttpRequest, ldap_auth: bool = False) -> None:
        """Log successful user login"""
        AuditLogger.log_event(
            AuditLogger.EventType.USER_LOGIN,
            user=user,
            request=request,
            details={"auth_method": "ldap" if ldap_auth else "local"},
            status="success"
        )
    
    @staticmethod
    def log_login_failed(username: str, request: Optional[HttpRequest] = None) -> None:
        """Log failed login attempt"""
        AuditLogger.log_event(
            AuditLogger.EventType.USER_LOGIN_FAILED,
            request=request,
            details={"attempted_user": username},
            status="failed"
        )
    
    @staticmethod
    def log_document_upload(
        user: User,
        request: HttpRequest,
        document_id: str,
        filename: str,
        file_size: int
    ) -> None:
        """Log document upload"""
        AuditLogger.log_event(
            AuditLogger.EventType.DOCUMENT_UPLOADED,
            user=user,
            request=request,
            resource_id=document_id,
            resource_type="document",
            details={
                "filename": filename,
                "size_bytes": file_size,
            }
        )
    
    @staticmethod
    def log_document_delete(
        user: User,
        request: HttpRequest,
        document_id: str,
        filename: str
    ) -> None:
        """Log document deletion"""
        AuditLogger.log_event(
            AuditLogger.EventType.DOCUMENT_DELETED,
            user=user,
            request=request,
            resource_id=document_id,
            resource_type="document",
            details={"filename": filename}
        )
    
    @staticmethod
    def log_permission_denied(
        user: Optional[User],
        request: Optional[HttpRequest],
        resource_type: str,
        resource_id: str,
        reason: str
    ) -> None:
        """Log permission denied events"""
        AuditLogger.log_event(
            AuditLogger.EventType.PERMISSION_DENIED,
            user=user,
            request=request,
            resource_id=resource_id,
            resource_type=resource_type,
            details={"reason": reason},
            status="denied"
        )
    
    @staticmethod
    def log_suspicious_activity(
        user: Optional[User],
        request: Optional[HttpRequest],
        description: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log suspicious activity for security review"""
        AuditLogger.log_event(
            AuditLogger.EventType.SUSPICIOUS_ACTIVITY,
            user=user,
            request=request,
            details={
                "description": description,
                **(details or {})
            },
            status="warning"
        )
