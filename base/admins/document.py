from django.contrib import admin
from document.models import Document
from document.tasks import deactivate_document
from base.application.document_indexing import enqueue_document_indexing

def process_selected_documents(modeladmin, request, queryset):
    unprocessed_docs = queryset.filter(is_vector_processed=False)
    for doc in unprocessed_docs:
        enqueue_document_indexing(doc)
    if not unprocessed_docs:
        modeladmin.message_user(request, "No unprocessed documents selected.", level='warning')
    else:
        modeladmin.message_user(request, f"Triggered processing for {unprocessed_docs.count()} document(s).")

process_selected_documents.short_description = "Process selected documents to vector DB"


class DocumentAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'type', 'department', 'is_vector_processed', 'is_active', 'created_at')
    actions = [process_selected_documents]
    readonly_fields = ('is_vector_processed',)
    list_filter = ('type', 'department', 'is_vector_processed', 'is_active', 'created_at')
    search_fields = ('title', 'description', 'code')
    ordering = ('-created_at',)
    def delete_queryset(self, request, queryset):
        # Trigger Celery deactivation before deletion
        for doc in queryset:
            deactivate_document.delay(str(doc.id), str(doc.type.name))
        self.message_user(request, f"Deleted {queryset.count()} document(s) and triggered deactivation.")
        super().delete_queryset(request, queryset)

    def get_actions(self, request):
        actions = super().get_actions(request)
        # Remove any custom delete action if present
        if 'delete_selected_documents' in actions:
            del actions['delete_selected_documents']
        return actions

try:
    admin.site.register(Document, DocumentAdmin)
except admin.sites.AlreadyRegistered:
    pass






