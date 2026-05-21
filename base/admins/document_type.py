from django.contrib import admin
from docs_type.models import DocumentType

try:
    admin.site.register(DocumentType)
except admin.sites.AlreadyRegistered:
    pass







