from django.db import migrations, models


ROLE_GROUPS = ("student", "teacher", "institution")


def forwards(apps, schema_editor):
    User = apps.get_model("auth2", "User")
    Group = apps.get_model("auth", "Group")

    groups_by_name = {}
    for group_name in ROLE_GROUPS:
        group, _ = Group.objects.get_or_create(name=group_name)
        groups_by_name[group_name] = group

    legacy_institution = Group.objects.filter(name="Institution").first()
    if legacy_institution:
        for user in legacy_institution.user_set.all():
            user.groups.add(groups_by_name["institution"])

    for user in User.objects.all():
        group_names = set(user.groups.values_list("name", flat=True))

        if "institution" in group_names or "Institution" in group_names:
            role = "institution"
        elif "teacher" in group_names:
            role = "teacher"
        elif "student" in group_names or "user" in group_names:
            role = "student"
        else:
            role = "student"

        user.role = role
        user.save(update_fields=["role"])
        user.groups.add(groups_by_name[role])


def backwards(apps, schema_editor):
    # Keep existing role data and groups when rolling back.
    return


class Migration(migrations.Migration):

    dependencies = [
        ("auth2", "0007_alter_user_language"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="role",
            field=models.CharField(
                choices=[
                    ("institution", "Institution"),
                    ("teacher", "Teacher"),
                    ("student", "Student"),
                ],
                default="student",
                max_length=20,
            ),
        ),
        migrations.RunPython(forwards, backwards),
    ]
