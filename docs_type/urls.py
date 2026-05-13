from django.urls import path
from . import views

app_name = 'docs_type'

urlpatterns = [
    path('', views.list_document_types, name='list_document_types'),
    path('create/', views.create_document_type, name='create_document_type'),
    path('<uuid:type_id>/update/', views.update_document_type, name='update_document_type'),
    path('<uuid:type_id>/delete/', views.delete_document_type, name='delete_document_type'),
]







