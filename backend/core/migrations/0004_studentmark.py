from django.db import migrations, models
import django.db.models.deletion
from django.db.models import F, Q


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0003_institution"),
    ]

    operations = [
        migrations.CreateModel(
            name="StudentMark",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("total_mark", models.PositiveIntegerField()),
                ("acquired_mark", models.PositiveIntegerField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "student",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="marks",
                        to="core.student",
                    ),
                ),
                (
                    "subject",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="student_marks",
                        to="core.subject",
                    ),
                ),
            ],
            options={
                "constraints": [
                    models.UniqueConstraint(
                        fields=("subject", "student"),
                        name="core_unique_subject_student_mark",
                    ),
                    models.CheckConstraint(
                        condition=Q(acquired_mark__lte=F("total_mark")),
                        name="core_acquired_mark_lte_total_mark",
                    ),
                ]
            },
        ),
    ]
