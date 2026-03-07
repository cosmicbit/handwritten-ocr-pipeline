from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0009_subject_unique_per_mapping"),
    ]

    operations = [
        migrations.AddField(
            model_name="department",
            name="institution",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="departments",
                to="core.institution",
            ),
        ),
        migrations.AddConstraint(
            model_name="department",
            constraint=models.UniqueConstraint(
                fields=("institution", "name"),
                name="core_unique_department_per_institution",
            ),
        ),
    ]
