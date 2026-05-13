from typing import Optional

from rest_framework import serializers
from datetime import date

from department.models import Department
from docs_type.models import DocumentType

from .models import Document, DocumentIndexingJob
from .utils import format_document_date, validate_document_date_range


def get_indexing_status_label(
    document: Document,
    latest_job: Optional[DocumentIndexingJob],
) -> str:
    if document.is_vector_processed:
        return "Ready"

    if latest_job and latest_job.status == DocumentIndexingJob.STATUS_FAILED:
        return "Failed"

    if (
        latest_job
        and latest_job.stage == DocumentIndexingJob.STAGE_DEACTIVATING_PREVIOUS_VERSIONS
    ):
        return "Updating Index"

    return "Processing"


def get_indexing_message(
    document: Document,
    latest_job: Optional[DocumentIndexingJob],
) -> Optional[str]:
    if document.is_vector_processed:
        return "Document is ready to ask question."

    if latest_job and latest_job.status == DocumentIndexingJob.STATUS_FAILED:
        return latest_job.error_message or "Document indexing failed."

    if not latest_job:
        return "Document is queued for indexing."

    stage_messages = {
        DocumentIndexingJob.STAGE_QUEUED: "Document is queued for indexing.",
        DocumentIndexingJob.STAGE_DEACTIVATING_PREVIOUS_VERSIONS: (
            "Deactivating previous version vectors before indexing the new version."
        ),
        DocumentIndexingJob.STAGE_EXTRACTING: "Extracting document content.",
        DocumentIndexingJob.STAGE_INDEXING: "Indexing document content.",
        DocumentIndexingJob.STAGE_FINALIZING: "Finalizing document indexing.",
        DocumentIndexingJob.STAGE_COMPLETED: "Document is ready to ask question.",
    }
    return stage_messages.get(latest_job.stage, "Document is processing on the server.")


class DocumentListSerializer(serializers.ModelSerializer):
    department = serializers.SerializerMethodField()
    document_type = serializers.SerializerMethodField()
    file_size = serializers.CharField(source='file_size_display')
    status = serializers.SerializerMethodField()
    memos = serializers.CharField(source='description')
    parent_document_id = serializers.SerializerMethodField()
    root_document_id = serializers.SerializerMethodField()
    effective_date = serializers.SerializerMethodField()
    expiry_date = serializers.SerializerMethodField()
    indexing_stage = serializers.SerializerMethodField()
    indexing_error = serializers.SerializerMethodField()
    indexing_message = serializers.SerializerMethodField()

    class Meta:
        model = Document
        fields = [
            'id', 'title', 'version', 'department', 'document_type', 'file_size',
            'publisher_id', 'publisher_name', 'created_at', 'status', 'memos', 'effective_date', 'expiry_date',
            'is_latest_version', 'parent_document_id', 'root_document_id', 'total_versions',
            'is_vector_processed', 'indexing_stage', 'indexing_error', 'indexing_message'
        ]

    def _get_latest_indexing_job(self, obj) -> Optional[DocumentIndexingJob]:
        prefetched_jobs = getattr(obj, "prefetched_indexing_jobs", None)
        if prefetched_jobs is not None:
            return prefetched_jobs[0] if prefetched_jobs else None
        return obj.indexing_jobs.order_by("-created_at").first()

    def get_department(self, obj):
        return obj.department.name if obj.department else None

    def get_document_type(self, obj):
        return obj.type.name if obj.type else None

    def get_parent_document_id(self, obj):
        return str(obj.parent_document.id) if obj.parent_document else None

    def get_root_document_id(self, obj):
        root_doc = obj.root_document
        return str(root_doc.id) if root_doc else str(obj.id)

    def get_status(self, obj):
        latest_job = self._get_latest_indexing_job(obj)
        return get_indexing_status_label(obj, latest_job)

    def get_indexing_stage(self, obj):
        if obj.is_vector_processed:
            return DocumentIndexingJob.STAGE_COMPLETED

        latest_job = self._get_latest_indexing_job(obj)
        return latest_job.stage if latest_job else DocumentIndexingJob.STAGE_QUEUED

    def get_indexing_error(self, obj):
        latest_job = self._get_latest_indexing_job(obj)
        if latest_job and latest_job.status == DocumentIndexingJob.STATUS_FAILED:
            return latest_job.error_message or None
        return None

    def get_indexing_message(self, obj):
        latest_job = self._get_latest_indexing_job(obj)
        return get_indexing_message(obj, latest_job)

    def get_effective_date(self, obj):
        return format_document_date(obj.effective_date)

    def get_expiry_date(self, obj):
        return format_document_date(obj.expiry_date)


class DocumentSuggestionSerializer(serializers.ModelSerializer):
    document_type = serializers.SerializerMethodField()
    file_name = serializers.CharField(source="original_filename")

    class Meta:
        model = Document
        fields = ["id", "title", "document_type", "file_name"]

    def get_document_type(self, obj):
        return obj.type.name if obj.type else None

class DocumentDetailSerializer(serializers.ModelSerializer):
    file_url = serializers.ReadOnlyField()
    file_size_display = serializers.ReadOnlyField()
    pdf_view_url = serializers.ReadOnlyField()
    version_count = serializers.ReadOnlyField()
    root_document = serializers.PrimaryKeyRelatedField(read_only=True)
    latest_version = serializers.SerializerMethodField()
    department_name = serializers.SerializerMethodField()
    type_name = serializers.SerializerMethodField()
    effective_date = serializers.SerializerMethodField()
    expiry_date = serializers.SerializerMethodField()
    memos = serializers.CharField(source='description')

    class Meta:
        model = Document
        fields = [
            'id', 'title', 'description', 'effective_date', 'expiry_date', 'memos',
            'department', 'department_name', 'type', 'type_name', 'version', 'parent_document', 'is_latest_version',
            'original_filename', 'stored_filename', 'file_path', 'file_url', 'pdf_view_url',
            'file_size', 'file_size_display', 'file_format', 'is_active',
            'publisher_id', 'publisher_name', 'created_at', 'updated_at',
            'version_count', 'root_document', 'latest_version', 'code', 'is_vector_processed'
        ]
        read_only_fields = [
            'id', 'created_at', 'updated_at', 'version_count', 'root_document', 'latest_version', 'file_url', 'file_size_display', 'pdf_view_url', 'department_name', 'type_name'
        ]

    def get_latest_version(self, obj):
        latest = obj.get_latest_version()
        if not latest or latest.pk == obj.pk:
            return None
        return DocumentDetailSerializer(latest).data

    def get_department_name(self, obj):
        return obj.department.name if obj.department else None

    def get_type_name(self, obj):
        return obj.type.name if obj.type else None

    def get_effective_date(self, obj):
        return format_document_date(obj.effective_date)

    def get_expiry_date(self, obj):
        return format_document_date(obj.expiry_date)

class DocumentUpdateSerializer(serializers.ModelSerializer):
    department_id = serializers.UUIDField(write_only=True, required=False)
    type_id = serializers.UUIDField(write_only=True, required=False)
    version = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    effective_date = serializers.DateField(required=False, input_formats=['%Y-%m-%d', '%d-%m-%Y'])
    expiry_date = serializers.DateField(required=False, input_formats=['%Y-%m-%d', '%d-%m-%Y'])

    class Meta:
        model = Document
        fields = ['title', 'description', 'department_id', 'type_id', 'is_active', 'version', 'effective_date', 'expiry_date']

    def validate(self, attrs):
        effective_date = attrs.get('effective_date', self.instance.effective_date if self.instance else None)
        expiry_date = attrs.get('expiry_date', self.instance.expiry_date if self.instance else None)

        try:
            validate_document_date_range(effective_date, expiry_date)
        except ValueError as exc:
            raise serializers.ValidationError({'expiry_date': str(exc)})

        return attrs

    def update(self, instance, validated_data):
        if 'title' in validated_data:
            instance.title = validated_data['title'].strip()
        if 'description' in validated_data:
            instance.description = validated_data['description']
        if 'is_active' in validated_data:
            instance.is_active = validated_data['is_active']
        if 'department_id' in validated_data:
            try:
                department = Department.objects.get(id=validated_data['department_id'])
                instance.department = department
            except Department.DoesNotExist:
                raise serializers.ValidationError({'department_id': 'Invalid department'})
        if 'type_id' in validated_data:
            try:
                doc_type = DocumentType.objects.get(id=validated_data['type_id'])
                instance.type = doc_type
            except DocumentType.DoesNotExist:
                raise serializers.ValidationError({'type_id': 'Invalid document type'})
        if 'version' in validated_data:
            instance.version = validated_data['version']
        if 'effective_date' in validated_data:
            instance.effective_date = validated_data['effective_date']
        if 'expiry_date' in validated_data:
            instance.expiry_date = validated_data['expiry_date']
        instance.save()
        return instance


