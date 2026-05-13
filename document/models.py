from django.db import models
from django.conf import settings
from django.urls import reverse
from docs_type.models import DocumentType
from department.models import Department
import uuid
import os
from decimal import Decimal
from .utils import delete_file_from_rag, get_pdf_version_path, ensure_pdf_version_exists
import logging

logger = logging.getLogger(__name__)

class Document(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    title = models.TextField()
    description = models.TextField(blank=True)
    effective_date = models.DateField(null=True, blank=True, default=None)
    expiry_date = models.DateField(null=True, blank=True)
    published_date = models.DateField(null=True, blank=True)
    code = models.CharField(max_length=50, null=True, blank=True)
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True, blank=True)
    type = models.ForeignKey(DocumentType, on_delete=models.SET_NULL, null=True, blank=True)
    
    # Versioning
    version = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal('1.0'))
    parent_document = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, related_name='versions')
    is_latest_version = models.BooleanField(default=True)
    total_versions = models.IntegerField(default=1)
    
    # File info
    original_filename = models.CharField(max_length=255)
    stored_filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    file_size = models.BigIntegerField()
    
    FILE_FORMATS = [
        ('pdf', 'PDF Document'),
        ('doc', 'Word Document (.doc)'),
        ('docx', 'Word Document (.docx)'),
        ('txt', 'Text File'),
    ]
    file_format = models.CharField(max_length=10, choices=FILE_FORMATS, default='pdf')

    # Status
    is_active = models.BooleanField(default=True)
    publisher_id = models.CharField(max_length=255, null=True, blank=True, help_text="User ID or username")
    publisher_name = models.CharField(max_length=255, null=True, blank=True, default="N/A", help_text="Human readable publisher name")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_vector_processed = models.BooleanField(default=False, help_text="Indicates if the document has been processed for vector DB")
    class Meta:
        db_table = 'documents'
        ordering = ['-created_at']
        app_label = 'document'
        permissions = [
            ("can_view_document", "Can view Document"),
            ("can_download_document", "Can download Document"),
            ("can_upload_document", "Can upload Document"),
            ("can_search_document", "Can search Document")
        ]
    def __str__(self):
        return f"{self.title} (v{self.version})"
    
    @property
    def file_url(self):
        """Get the URL to access the file."""
        return reverse("document:download_document", args=[self.id]) if self.file_path else None

    @property
    def full_file_path(self):
        """Get the full file system path."""
        return os.path.join(settings.MEDIA_ROOT, self.file_path) if self.file_path else None

    @property
    def pdf_view_url(self):
        """Get the URL to access the PDF version for viewing."""
        return reverse("document:view_document_pdf", args=[self.id]) if self.file_path else None

    @property
    def pdf_view_path(self):
        """Get the full file system path to the PDF version."""
        return get_pdf_version_path(self)

    def ensure_pdf_for_viewing(self):
        """Ensure a PDF version exists for viewing and return the URL."""
        try:
            if ensure_pdf_version_exists(self):
                return self.pdf_view_url
            return None
        except Exception as e:
            logger.error(f"Error creating PDF for viewing document {self.id}: {str(e)}")
            return None
    
    @property
    def file_size_display(self):
        """Get human readable file size."""
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    @property
    def root_document(self):
        """Get the original (first) version in the full parent chain."""
        root = self
        while root.parent_document_id:
            root = root.parent_document
        return root

    def get_family_ids(self):
        """Return all document IDs in this version family (root + descendants)."""
        root_id = self.root_document.id
        family_ids = {root_id}
        frontier = {root_id}

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
    
    def get_latest_version(self):
        """Get the latest version."""
        family_ids = self.get_family_ids()
        return Document.objects.filter(
            id__in=family_ids,
            is_latest_version=True
        ).order_by('-version', '-created_at').first()

    def get_all_versions(self):
        """Get all versions of this document family (active only)."""
        family_ids = self.get_family_ids()
        return Document.objects.filter(
            id__in=family_ids,
            is_active=True
        ).order_by('-version', '-created_at')
    
    def create_new_version(self, **file_data):
        """Create a new version with updated file data."""
        # Mark current as not latest
        self.is_latest_version = False
        self.save()
        
        # Create new version
        return Document.objects.create(
            title=self.title,
            description=self.description,
            department=self.department,
            type=self.type,
            effective_date=self.effective_date,
            version=self.version + 1,
            parent_document=self.get_latest_version() or self.root_document,
            is_latest_version=True,
            **file_data
        )
    
    def delete(self, *args, **kwargs):
        """Delete file from disk and RAG system when document is deleted."""
        # Delete from RAG system first
        try:
            file_type_name = self.type.name if self.type else "unknown"
            delete_file_from_rag(file_id=str(self.id), file_type=file_type_name)
            logger.info(f"Successfully deleted RAG data for document ID: {self.id}")
        except Exception as e:
            logger.error(f"Error deleting RAG data for document ID: {self.id}: {e}")

        # Then delete file from disk
        if self.full_file_path and os.path.exists(self.full_file_path):
            try:
                os.remove(self.full_file_path)
            except OSError as e:
                logger.error(f"Error deleting file from disk for document ID: {self.id}: {e}")
        
        super().delete(*args, **kwargs)


class DocumentIndexingJob(models.Model):
    STATUS_QUEUED = "queued"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    STAGE_QUEUED = "queued"
    STAGE_DEACTIVATING_PREVIOUS_VERSIONS = "deactivating_previous_versions"
    STAGE_EXTRACTING = "extracting"
    STAGE_INDEXING = "indexing"
    STAGE_FINALIZING = "finalizing"
    STAGE_COMPLETED = "completed"
    STAGE_FAILED = "failed"

    STATUS_CHOICES = [
        (STATUS_QUEUED, "Queued"),
        (STATUS_RUNNING, "Running"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_FAILED, "Failed"),
    ]

    STAGE_CHOICES = [
        (STAGE_QUEUED, "Queued"),
        (STAGE_DEACTIVATING_PREVIOUS_VERSIONS, "Deactivating Previous Versions"),
        (STAGE_EXTRACTING, "Extracting"),
        (STAGE_INDEXING, "Indexing"),
        (STAGE_FINALIZING, "Finalizing"),
        (STAGE_COMPLETED, "Completed"),
        (STAGE_FAILED, "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="indexing_jobs"
    )
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default=STATUS_QUEUED
    )
    stage = models.TextField(
        choices=STAGE_CHOICES, default=STAGE_QUEUED
    )
    source_path = models.CharField(max_length=1000, blank=True, default="")
    error_message = models.TextField(blank=True, default="")
    attempts = models.PositiveIntegerField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "document_indexing_jobs"
        ordering = ["-created_at"]

    def __str__(self):
        return f"Indexing Job {self.id} for {self.document_id}"
