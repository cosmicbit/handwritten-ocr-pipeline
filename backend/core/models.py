from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import F, Q

class Department(models.Model):
    name = models.CharField(max_length=200)
   

class Teacher(models.Model):

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="teacher_user",
    )
    department = models.ForeignKey(
        Department,
        on_delete=models.CASCADE,
        related_name="teacher_department"
    )
    created_at = models.DateField(auto_created=True)



class Subject(models.Model):

    class Semester(models.IntegerChoices):
        ONE = 1, 'ONE'
        TWO = 2, 'TWO'
        THREE = 3, 'THREE'
        FOUR = 4, 'FOUR'
        FIVE = 5, 'FIVE'
        SIX = 6, 'SIX'
        SEVEN = 7, 'SEVEN'
        EIGHT = 8, 'EIGHT'

    name = models.CharField(max_length=100)
    code = models.CharField(
        max_length=10,
        unique=True   # ensures unique subject code
    )
    semester = models.IntegerField(
        choices=Semester.choices
    )
    department = models.ForeignKey(
        Department, 
        on_delete=models.CASCADE,
        related_name="subject_department"
    )
    teacher = models.ForeignKey(
        Teacher,
        on_delete=models.CASCADE,
        related_name="subjects"
    )
    created_at = models.DateField(auto_now_add=True)


 
class Student(models.Model):
   
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="student_user",
    )
    subjects = models.ManyToManyField(
        Subject,
        related_name='students'
    )
    department = models.ForeignKey(
        Department, 
        on_delete=models.CASCADE,
        related_name="department_student"
    )
    created_at = models.DateField(auto_now_add=True)

class StudentUnderTeacher(models.Model):
    student = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="teacher_links",
    )
    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="student_links",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "rbac_studentunderteacher"
        constraints = [
            models.UniqueConstraint(
                fields=["teacher", "student"],
                name="rbac_unique_teacher_student",
            ),
            models.CheckConstraint(
                condition=~Q(teacher=F("student")),
                name="rbac_teacher_student_not_same",
            ),
        ]
        indexes = [
            models.Index(fields=["teacher"], name="rbac_teacher_idx"),
            models.Index(fields=["student"], name="rbac_student_idx"),
        ]

    def __str__(self):
        return f"teacher={self.teacher_id} student={self.student_id}"

    def clean(self):
        if self.teacher_id and getattr(self.teacher, "role", None) != "teacher":
            raise ValidationError({"teacher": "Selected user is not a teacher."})
        if self.student_id and getattr(self.student, "role", None) != "student":
            raise ValidationError({"student": "Selected user is not a student."})

    def save(self, *args, **kwargs):
        self.full_clean()
        return super().save(*args, **kwargs)


class TeacherInstitution(models.Model):
    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="institution_links",
    )
    institution = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="teacher_institution_links",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "rbac_teacherinstitution"
        constraints = [
            models.UniqueConstraint(
                fields=["teacher", "institution"],
                name="rbac_teacher_institution_uq",
            ),
            models.CheckConstraint(
                condition=~Q(teacher=F("institution")),
                name="rbac_teacher_institution_chk",
            ),
        ]
        indexes = [
            models.Index(fields=["teacher"], name="rbac_ti_teacher_idx"),
            models.Index(fields=["institution"], name="rbac_ti_inst_idx"),
        ]

    def __str__(self):
        return f"teacher={self.teacher_id} institution={self.institution_id}"

    def clean(self):
        if self.teacher_id and getattr(self.teacher, "role", None) != "teacher":
            raise ValidationError({"teacher": "Selected user is not a teacher."})
        if self.institution_id and getattr(self.institution, "role", None) != "institution":
            raise ValidationError({"institution": "Selected user is not an institution."})

    def save(self, *args, **kwargs):
        self.full_clean()
        return super().save(*args, **kwargs)
