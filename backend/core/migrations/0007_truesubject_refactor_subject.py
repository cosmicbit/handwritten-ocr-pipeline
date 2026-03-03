from django.db import migrations, models
import django.db.models.deletion


def forwards_copy_subjects_to_true_subject(apps, schema_editor):
    Subject = apps.get_model("core", "Subject")
    TrueSubject = apps.get_model("core", "TrueSubject")

    for subject in Subject.objects.all().iterator():
        true_subject, _ = TrueSubject.objects.get_or_create(
            code=subject.code,
            defaults={"name": subject.name},
        )
        subject.true_subject_id = true_subject.id
        subject.save(update_fields=["true_subject"])


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0006_teacher_institution_user_onetoone"),
    ]

    operations = [
        migrations.CreateModel(
            name="TrueSubject",
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
                ("name", models.CharField(max_length=100)),
                ("code", models.CharField(max_length=10, unique=True)),
                ("created_at", models.DateField(auto_now_add=True)),
            ],
        ),
        migrations.AddField(
            model_name="subject",
            name="true_subject",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="subjects",
                to="core.truesubject",
            ),
        ),
        migrations.RunPython(forwards_copy_subjects_to_true_subject, noop_reverse),
        migrations.AlterField(
            model_name="subject",
            name="true_subject",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="subjects",
                to="core.truesubject",
            ),
        ),
        migrations.RemoveField(
            model_name="subject",
            name="code",
        ),
        migrations.RemoveField(
            model_name="subject",
            name="name",
        ),
        migrations.AddConstraint(
            model_name="subject",
            constraint=models.UniqueConstraint(
                fields=("true_subject", "semester", "department", "teacher"),
                name="core_unique_subject_mapping",
            ),
        ),
    ]
