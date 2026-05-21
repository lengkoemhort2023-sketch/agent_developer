from django.db import migrations

def copy_name_to_presentation(apps, schema_editor):
    DocumentType = apps.get_model('docs_type', 'DocumentType')
    for doc_type in DocumentType.objects.all():
        doc_type.presentation = doc_type.name
        doc_type.save(update_fields=['presentation'])

class Migration(migrations.Migration):
    dependencies = [
        ('docs_type', '0002_documenttype_presentation_alter_documenttype_name'),
    ]

    operations = [
        migrations.RunPython(copy_name_to_presentation),
    ]







