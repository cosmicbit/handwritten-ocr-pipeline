from django.db import migrations, models
import django.db.models.deletion


def backfill_subject_institution(apps, schema_editor):
    Subject = apps.get_model("core", "Subject")
    Institution = apps.get_model("core", "Institution")
    teacher_link_model = Institution.teachers.through

    teacher_to_institution = {}
    for link in teacher_link_model.objects.all().order_by("institution_id"):
        teacher_id = getattr(link, "teacher_id")
        institution_id = getattr(link, "institution_id")
        teacher_to_institution.setdefault(teacher_id, institution_id)

    missing_subject_ids = []
    for subject in Subject.objects.all().iterator():
        institution_id = teacher_to_institution.get(subject.teacher_id)
        if not institution_id:
            missing_subject_ids.append(subject.id)
            continue
        subject.institution_id = institution_id
        subject.save(update_fields=["institution"])

    if missing_subject_ids:
        raise RuntimeError(
            "Could not backfill Subject.institution for subject ids: "
            + ", ".join(str(i) for i in missing_subject_ids)
            + ". Ensure each subject's teacher is linked to an Institution before running this migration."
        )


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0007_truesubject_refactor_subject"),
    ]

    operations = [
        migrations.AddField(
            model_name="subject",
            name="institution",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="subjects",
                to="core.institution",
            ),
        ),
        migrations.RunPython(backfill_subject_institution, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="subject",
            name="institution",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="subjects",
                to="core.institution",
            ),
        ),
        migrations.RemoveConstraint(
            model_name="subject",
            name="core_unique_subject_mapping",
        ),
        migrations.AddConstraint(
            model_name="subject",
            constraint=models.UniqueConstraint(
                fields=("institution", "true_subject", "semester", "department", "teacher"),
                name="core_unique_subject_mapping",
            ),
        ),
    ]
