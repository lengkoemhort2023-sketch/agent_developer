import django.db.models.deletion
import uuid
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0013_chatsession_archiving"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="MessageFeedback",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                (
                    "message_id_str",
                    models.CharField(
                        blank=True, db_index=True, max_length=36, null=True
                    ),
                ),
                (
                    "rating",
                    models.IntegerField(choices=[(1, "Positive"), (-1, "Negative")]),
                ),
                ("feedback_text", models.TextField(blank=True, null=True)),
                ("question", models.TextField(blank=True, null=True)),
                ("answer", models.TextField(blank=True, null=True)),
                (
                    "user_name",
                    models.CharField(blank=True, max_length=255, null=True),
                ),
                ("user_email", models.EmailField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "message",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="feedbacks",
                        to="chat.chatmessage",
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="feedbacks",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "db_table": "message_feedbacks",
                "ordering": ["-created_at"],
            },
        ),
        migrations.AlterUniqueTogether(
            name="messagefeedback",
            unique_together={("message_id_str", "user")},
        ),
    ]
