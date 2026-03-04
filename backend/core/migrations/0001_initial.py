# Generated manually to move selected RBAC tables into core app state.
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("rbac", "0002_teacherinstitution"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],
            state_operations=[
                migrations.CreateModel(
                    name="StudentUnderTeacher",
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
                        ("created_at", models.DateTimeField(auto_now_add=True)),
                        (
                            "student",
                            models.ForeignKey(
                                on_delete=django.db.models.deletion.CASCADE,
                                related_name="teacher_links",
                                to=settings.AUTH_USER_MODEL,
                            ),
                        ),
                        (
                            "teacher",
                            models.ForeignKey(
                                on_delete=django.db.models.deletion.CASCADE,
                                related_name="student_links",
                                to=settings.AUTH_USER_MODEL,
                            ),
                        ),
                    ],
                    options={
                        "db_table": "rbac_studentunderteacher",
                        "indexes": [
                            models.Index(fields=["teacher"], name="rbac_teacher_idx"),
                            models.Index(fields=["student"], name="rbac_student_idx"),
                        ],
                        "constraints": [
                            models.UniqueConstraint(
                                fields=("teacher", "student"),
                                name="rbac_unique_teacher_student",
                            ),
                            models.CheckConstraint(
                                condition=models.Q(
                                    ("teacher", models.F("student")),
                                    _negated=True,
                                ),
                                name="rbac_teacher_student_not_same",
                            ),
                        ],
                    },
                ),
                migrations.CreateModel(
                    name="TeacherInstitution",
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
                        ("created_at", models.DateTimeField(auto_now_add=True)),
                        (
                            "institution",
                            models.ForeignKey(
                                on_delete=django.db.models.deletion.CASCADE,
                                related_name="teacher_institution_links",
                                to=settings.AUTH_USER_MODEL,
                            ),
                        ),
                        (
                            "teacher",
                            models.ForeignKey(
                                on_delete=django.db.models.deletion.CASCADE,
                                related_name="institution_links",
                                to=settings.AUTH_USER_MODEL,
                            ),
                        ),
                    ],
                    options={
                        "db_table": "rbac_teacherinstitution",
                        "indexes": [
                            models.Index(fields=["teacher"], name="rbac_ti_teacher_idx"),
                            models.Index(fields=["institution"], name="rbac_ti_inst_idx"),
                        ],
                        "constraints": [
                            models.UniqueConstraint(
                                fields=("teacher", "institution"),
                                name="rbac_teacher_institution_uq",
                            ),
                            models.CheckConstraint(
                                condition=models.Q(
                                    ("teacher", models.F("institution")),
                                    _negated=True,
                                ),
                                name="rbac_teacher_institution_chk",
                            ),
                        ],
                    },
                ),
            ],
        )
    ]
