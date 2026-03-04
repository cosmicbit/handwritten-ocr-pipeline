from django.db import migrations, models
from django.db.models import Count


def assert_no_duplicates_before_constraint(apps, schema_editor):
    Subject = apps.get_model("core", "Subject")
    duplicates = (
        Subject.objects.values("institution_id", "true_subject_id", "semester", "department_id")
        .annotate(c=Count("id"))
        .filter(c__gt=1)
    )

    dup = list(duplicates[:5])
    if dup:
        raise RuntimeError(
            "Cannot enforce unique subject mapping because duplicates exist for "
            "(institution, true_subject, semester, department). "
            f"Sample duplicates: {dup}"
        )


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0008_subject_institution_fk"),
    ]

    operations = [
        migrations.RunPython(assert_no_duplicates_before_constraint, migrations.RunPython.noop),
        migrations.RemoveConstraint(
            model_name="subject",
            name="core_unique_subject_mapping",
        ),
        migrations.AddConstraint(
            model_name="subject",
            constraint=models.UniqueConstraint(
                fields=("institution", "true_subject", "semester", "department"),
                name="core_unique_subject_mapping",
            ),
        ),
    ]
