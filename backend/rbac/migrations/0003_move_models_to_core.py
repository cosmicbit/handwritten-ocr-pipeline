from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("rbac", "0002_teacherinstitution"),
        ("core", "0001_initial"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],
            state_operations=[
                migrations.DeleteModel(name="TeacherInstitution"),
                migrations.DeleteModel(name="StudentUnderTeacher"),
            ],
        )
    ]
