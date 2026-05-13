from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):
    dependencies = [
        ("document", "0006_alter_document_title"),
    ]

    operations = [
        migrations.CreateModel(
            name="DocumentIndexingJob",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("status", models.CharField(choices=[("queued", "Queued"), ("running", "Running"), ("completed", "Completed"), ("failed", "Failed")], default="queued", max_length=20)),
                ("stage", models.CharField(choices=[("queued", "Queued"), ("extracting", "Extracting"), ("indexing", "Indexing"), ("finalizing", "Finalizing"), ("completed", "Completed"), ("failed", "Failed")], default="queued", max_length=20)),
                ("source_path", models.CharField(blank=True, default="", max_length=1000)),
                ("error_message", models.TextField(blank=True, default="")),
                ("attempts", models.PositiveIntegerField(default=0)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("document", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="indexing_jobs", to="document.document")),
            ],
            options={
                "db_table": "document_indexing_jobs",
                "ordering": ["-created_at"],
            },
        ),
    ]
