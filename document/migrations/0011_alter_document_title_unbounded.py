from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("document", "0010_alter_documentindexingjob_stage_unbounded"),
    ]

    operations = [
        migrations.AlterField(
            model_name="document",
            name="title",
            field=models.TextField(),
        ),
    ]
