from django.urls import path
from . import views
from .views import DocumentDetailView

app_name = 'document'

urlpatterns = [
    # CSRF token endpoint
    path('csrf-token/', views.get_csrf_token, name='get_csrf_token'),
    
    # Upload endpoints
    path('upload/', views.upload_document, name='upload_document'),

    # Document management
    # list?active=true&department_id=<id>&type_id=<id>&title=<title>&sort=<field>
    path('list/', views.list_documents, name='list_documents'),
    path('suggestions/', views.list_document_suggestions, name='list_document_suggestions'),
    path('<uuid:pk>/download/', views.download_document, name='download_document'),
    path('<uuid:pk>/download-token/', views.get_download_token, name='get_download_token'),
    path('<uuid:pk>/index-status/', views.get_document_index_status, name='get_document_index_status'),
    path('<uuid:pk>/pdf-url/', views.get_document_pdf_url, name='get_document_pdf_url'),
    path('<uuid:pk>/preview/', views.view_document_preview, name='view_document_preview'),
    path('<uuid:pk>/preview/pages/<int:page_number>/', views.view_document_preview_page, name='view_document_preview_page'),
    path('v/<uuid:pk>/', views.view_document_pdf, name='view_document_pdf'),
    path('<uuid:pk>/update/', views.update_document, name='update_document'),
    path('<uuid:pk>/update-version/', views.update_document_version, name='update_document_version'),
    path('<uuid:pk>/versions/', views.get_document_versions, name='get_document_versions'),
    path('<uuid:pk>/', DocumentDetailView.as_view(), name='document_detail'),

    path('<uuid:pk>/upload-version/', views.upload_new_version, name='upload_new_version'),
    path('<uuid:pk>/delete/', views.delete_document, name='delete_document'),
    # ?hard=true
    # ?hard=true&all_versions=true
    path('<uuid:pk>/restore/', views.restore_document, name='restore_document'),
    path('years/', views.get_unique_years, name='get_unique_years'),

    # Archive/Trash endpoints
    path('trash/', views.list_inactive_documents, name='list_trash'),
]
