from django.db import models
import uuid

class DocumentType(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, unique=True)
    presentation = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'document_types'
        app_label = 'docs_type'
    
    def save(self, *args, **kwargs):
        # Normalize name: lower case and replace spaces with underscores
        if self.presentation:
            self.name = self.presentation.lower().replace(' ', '_')
        super().save(*args, **kwargs)
    
    def __str__(self):
        return self.presentation







