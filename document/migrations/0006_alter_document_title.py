# Generated manually to alter title max_length

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('document', '0005_remove_document_pdf_preview_path'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='title',
            field=models.CharField(max_length=1000),
        ),
    ]







