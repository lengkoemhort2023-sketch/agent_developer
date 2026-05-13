from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0012_chatmessage_suggestions"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatsession",
            name="archived_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="chatsession",
            name="is_archived",
            field=models.BooleanField(default=False),
        ),
        migrations.AddIndex(
            model_name="chatsession",
            index=models.Index(
                fields=["user", "is_archived", "-updated_at"],
                name="chat_sessio_user_id_64cd64_idx",
            ),
        ),
    ]
