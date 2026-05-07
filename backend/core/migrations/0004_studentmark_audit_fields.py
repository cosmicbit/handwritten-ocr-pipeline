from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0003_alter_teacherpdfupload_teacher"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name="studentmark",
            name="update_source",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        migrations.AddField(
            model_name="studentmark",
            name="updated_at",
            field=models.DateTimeField(auto_now=True, blank=True, null=True),
        ),
        migrations.AddField(
            model_name="studentmark",
            name="updated_by",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="updated_student_marks",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
