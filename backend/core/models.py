from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import F, Q
import uuid


class Institution(models.Model):
    name = models.CharField(max_length=200)

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE
    )

    teachers = models.ManyToManyField(
        "Teacher",
        related_name="teachers_in_institution"
    )

    students = models.ManyToManyField(
        "Student",
        related_name="students_in_institution"
    )

    created_at = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.name


class Department(models.Model):
    name = models.CharField(max_length=200)

    institution = models.ForeignKey(
        Institution,
        on_delete=models.CASCADE,
        related_name="departments",
        null=True,
        blank=True,
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["institution", "name"],
                name="core_unique_department_per_institution",
            )
        ]

    def __str__(self):
        return self.name


class Teacher(models.Model):

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="teacher_user",
    )

    department = models.ForeignKey(
        Department,
        on_delete=models.CASCADE,
        related_name="teacher_department"
    )

    created_at = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} ({self.department.name})"


class TrueSubject(models.Model):
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=10, unique=True)
    created_at = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.code} - {self.name}"


class Subject(models.Model):

    class Semester(models.IntegerChoices):
        ONE = 1, "ONE"
        TWO = 2, "TWO"
        THREE = 3, "THREE"
        FOUR = 4, "FOUR"
        FIVE = 5, "FIVE"
        SIX = 6, "SIX"
        SEVEN = 7, "SEVEN"
        EIGHT = 8, "EIGHT"

    true_subject = models.ForeignKey(
        TrueSubject,
        on_delete=models.CASCADE,
        related_name="subjects",
    )

    semester = models.IntegerField(choices=Semester.choices)

    department = models.ForeignKey(
        Department,
        on_delete=models.CASCADE,
        related_name="subject_department"
    )

    institution = models.ForeignKey(
        Institution,
        on_delete=models.CASCADE,
        related_name="subjects",
    )

    teacher = models.ForeignKey(
        Teacher,
        on_delete=models.CASCADE,
        related_name="subjects"
    )

    created_at = models.DateField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["institution", "true_subject", "semester", "department"],
                name="core_unique_subject_mapping",
            )
        ]

    def __str__(self):
        return f"{self.true_subject.name} - Sem {self.semester}"


class Student(models.Model):

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="student_user",
    )

    subjects = models.ManyToManyField(
        Subject,
        related_name="students"
    )

    department = models.ForeignKey(
        Department,
        on_delete=models.CASCADE,
        related_name="department_student"
    )

    created_at = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} ({self.department.name})"


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
                check=~Q(teacher=F("student")),
                name="rbac_teacher_student_not_same",
            ),
        ]

        indexes = [
            models.Index(fields=["teacher"], name="rbac_teacher_idx"),
            models.Index(fields=["student"], name="rbac_student_idx"),
        ]

    def __str__(self):
        return f"teacher={self.teacher_id} student={self.student_id}"


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
                check=~Q(teacher=F("institution")),
                name="rbac_teacher_institution_chk",
            ),
        ]

    def __str__(self):
        return f"teacher={self.teacher_id} institution={self.institution_id}"


class StudentMark(models.Model):

    subject = models.ForeignKey(
        Subject,
        on_delete=models.CASCADE,
        related_name="student_marks",
    )

    student = models.ForeignKey(
        Student,
        on_delete=models.CASCADE,
        related_name="marks",
    )

    total_mark = models.PositiveIntegerField()
    acquired_mark = models.PositiveIntegerField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["subject", "student"],
                name="core_unique_subject_student_mark",
            ),

            models.CheckConstraint(
                check=Q(acquired_mark__lte=F("total_mark")),
                name="core_acquired_mark_lte_total_mark",
            ),
        ]

    def __str__(self):
        return f"{self.student_id} {self.acquired_mark}/{self.total_mark}"


def teacher_pdf_upload_path(_, __):
    return f"uploads/{uuid.uuid4().hex}.pdf"


def safe_teacher_text_upload_path(_, __):
    return f"safe/teacher/{uuid.uuid4().hex}.txt"


def safe_student_text_upload_path(_, __):
    return f"safe/student/{uuid.uuid4().hex}.txt"


class TeacherPDFUpload(models.Model):

    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="teacher_pdf_uploads",
    )

    subject = models.ForeignKey(
        Subject,
        on_delete=models.CASCADE,
        related_name="student_pdf_uploads",
        null=True,
        blank=True,
    )

    student = models.ForeignKey(
        Student,
        on_delete=models.CASCADE,
        related_name="uploaded_pdfs",
        null=True,
        blank=True,
    )

    file = models.FileField(upload_to=teacher_pdf_upload_path)

    original_filename = models.CharField(max_length=255)

    extracted_text = models.TextField(blank=True, default="")

    extracted_text_file = models.FileField(
        upload_to=safe_student_text_upload_path,
        blank=True,
        null=True
    )

    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)


def teacher_answer_key_upload_path(_, __):
    return f"uploads/answer-keys/{uuid.uuid4().hex}.pdf"


class TeacherSubjectAnswerKey(models.Model):

    teacher = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="teacher_answer_keys",
    )

    subject = models.ForeignKey(
        Subject,
        on_delete=models.CASCADE,
        related_name="answer_keys",
    )

    file = models.FileField(upload_to=teacher_answer_key_upload_path)

    original_filename = models.CharField(max_length=255)

    extracted_text = models.TextField(blank=True, default="")

    extracted_text_file = models.FileField(
        upload_to=safe_teacher_text_upload_path,
        blank=True,
        null=True,
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["subject"],
                name="core_unique_answer_key_per_subject",
            )
        ]

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)