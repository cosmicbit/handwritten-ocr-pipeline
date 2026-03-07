from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.db.models import F, Q
import uuid

class Department(models.Model):
    name = models.CharField(max_length=200)
    institution = models.ForeignKey(
        "Institution",
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
        username = self.user.username if self.user_id and self.user else f"user_id={self.user_id}"
        department_name = self.department.name if self.department_id and self.department else f"department_id={self.department_id}"
        return f"{username} ({department_name})"


class TrueSubject(models.Model):
    name = models.CharField(max_length=100)
    code = models.CharField(max_length=10, unique=True)
    created_at = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.code} - {self.name}"



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

    true_subject = models.ForeignKey(
        TrueSubject,
        on_delete=models.CASCADE,
        related_name="subjects",
    )
    semester = models.IntegerField(
        choices=Semester.choices
    )
    department = models.ForeignKey(
        Department, 
        on_delete=models.CASCADE,
        related_name="subject_department"
    )
    institution = models.ForeignKey(
        "Institution",
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
        subject_name = self.true_subject.name if self.true_subject_id and self.true_subject else f"true_subject_id={self.true_subject_id}"
        department_name = self.department.name if self.department_id and self.department else f"department_id={self.department_id}"
        return f"{subject_name} - Sem {self.semester} ({department_name})"


 
class Student(models.Model):
   
    user = models.OneToOneField(
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

    def __str__(self):
        username = self.user.username if self.user_id and self.user else f"user_id={self.user_id}"
        department_name = self.department.name if self.department_id and self.department else f"department_id={self.department_id}"
        return f"{username} ({department_name})"


class Institution(models.Model):
    name = models.CharField(max_length=200)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE
    )
    teachers = models.ManyToManyField(
        Teacher, 
        related_name="teachers_in_institution"
    )
    students = models.ManyToManyField(
        Student, 
        related_name="students_in_instituation"
    )
    created_at = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.name


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
        update_fields = kwargs.get("update_fields")
        # Background task updates extracted text fields only; relation validation
        # is already enforced at upload time and can fail after business-rule changes.
        if update_fields:
            update_fields_set = set(update_fields)
            if update_fields_set.issubset({"extracted_text", "extracted_text_file"}):
                return super().save(*args, **kwargs)
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
                condition=Q(acquired_mark__lte=F("total_mark")),
                name="core_acquired_mark_lte_total_mark",
            ),
        ]

    def __str__(self):
        return f"student={self.student_id} subject={self.subject_id} marks={self.acquired_mark}/{self.total_mark}"


def _teacher_pdf_upload_path(_instance, _filename):
    return f"uploads/{uuid.uuid4().hex}.pdf"


def _safe_teacher_text_upload_path(_instance, _filename):
    return f"safe/teacher/{uuid.uuid4().hex}.txt"


def _safe_student_text_upload_path(_instance, _filename):
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
    file = models.FileField(upload_to=_teacher_pdf_upload_path)
    original_filename = models.CharField(max_length=255)
    extracted_text = models.TextField(blank=True, default="")
    extracted_text_file = models.FileField(upload_to=_safe_student_text_upload_path, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def clean(self):
        if self.teacher_id and getattr(self.teacher, "role", None) != "teacher":
            raise ValidationError({"teacher": "Selected user is not a teacher."})
        if self.subject_id and self.teacher_id:
            subject_teacher_user_id = Subject.objects.filter(id=self.subject_id).values_list("teacher__user_id", flat=True).first()
            if subject_teacher_user_id and subject_teacher_user_id != self.teacher_id:
                raise ValidationError({"subject": "Subject does not belong to this teacher."})
        if self.subject_id and self.student_id:
            pair = Subject.objects.filter(id=self.subject_id).values_list("department_id", "teacher__user_id").first()
            if pair:
                subject_department_id, subject_teacher_user_id = pair
                student_department_id = Student.objects.filter(id=self.student_id).values_list("department_id", flat=True).first()
                if subject_teacher_user_id and subject_teacher_user_id != self.teacher_id:
                    raise ValidationError({"subject": "Subject does not belong to this teacher."})
                if student_department_id != subject_department_id:
                    raise ValidationError({"student": "Student and subject must belong to the same department."})

    def save(self, *args, **kwargs):
        self.full_clean()
        return super().save(*args, **kwargs)


def _teacher_answer_key_upload_path(_instance, _filename):
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
    file = models.FileField(upload_to=_teacher_answer_key_upload_path)
    original_filename = models.CharField(max_length=255)
    extracted_text = models.TextField(blank=True, default="")
    extracted_text_file = models.FileField(upload_to=_safe_teacher_text_upload_path, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["subject"],
                name="core_unique_answer_key_per_subject",
            ),
        ]

    def clean(self):
        if self.teacher_id and getattr(self.teacher, "role", None) != "teacher":
            raise ValidationError({"teacher": "Selected user is not a teacher."})

    def save(self, *args, **kwargs):
        self.full_clean()
        #update
        return super().save(*args, **kwargs)
    
