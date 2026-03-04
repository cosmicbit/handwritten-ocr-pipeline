from django.test import TestCase
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError

from core.models import StudentUnderTeacher, TeacherInstitution


User = get_user_model()


class StudentUnderTeacherModelTests(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user(
            username="teacher1",
            email="teacher1@example.com",
            password="Pass@12345",
            role="teacher",
        )
        self.student = User.objects.create_user(
            username="student1",
            email="student1@example.com",
            password="Pass@12345",
            role="student",
        )

    def test_valid_teacher_student_link_is_saved(self):
        link = StudentUnderTeacher.objects.create(teacher=self.teacher, student=self.student)
        self.assertIsNotNone(link.id)

    def test_invalid_roles_are_rejected(self):
        with self.assertRaises(ValidationError):
            StudentUnderTeacher.objects.create(teacher=self.student, student=self.teacher)

    def test_same_user_cannot_be_teacher_and_student(self):
        with self.assertRaises(ValidationError):
            StudentUnderTeacher.objects.create(teacher=self.teacher, student=self.teacher)


class TeacherInstitutionModelTests(TestCase):
    def setUp(self):
        self.teacher = User.objects.create_user(
            username="teacher2",
            email="teacher2@example.com",
            password="Pass@12345",
            role="teacher",
        )
        self.institution = User.objects.create_user(
            username="institution1",
            email="institution1@example.com",
            password="Pass@12345",
            role="institution",
        )
        self.student = User.objects.create_user(
            username="student2",
            email="student2@example.com",
            password="Pass@12345",
            role="student",
        )

    def test_valid_teacher_institution_link_is_saved(self):
        link = TeacherInstitution.objects.create(teacher=self.teacher, institution=self.institution)
        self.assertIsNotNone(link.id)

    def test_invalid_roles_are_rejected(self):
        with self.assertRaises(ValidationError):
            TeacherInstitution.objects.create(teacher=self.student, institution=self.institution)

        with self.assertRaises(ValidationError):
            TeacherInstitution.objects.create(teacher=self.teacher, institution=self.student)
