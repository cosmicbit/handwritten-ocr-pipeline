import shutil
import tempfile

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from auth2.jwt_utils import create_jwt_token
from .models import (
    Department,
    Institution,
    Student,
    StudentMark,
    StudentUnderTeacher,
    Subject,
    Teacher,
    TeacherPDFUpload,
    TeacherSubjectAnswerKey,
    TrueSubject,
)

User = get_user_model()


class TeacherUploadPDFTests(TestCase):
    def setUp(self):
        self.media_root = tempfile.mkdtemp(prefix="core-test-media-")
        self.addCleanup(lambda: shutil.rmtree(self.media_root, ignore_errors=True))

        self.teacher = User.objects.create_user(
            username="teacher_pdf",
            email="teacher_pdf@example.com",
            password="Pass@12345",
            role="teacher",
        )
        self.student = User.objects.create_user(
            username="student_pdf",
            email="student_pdf@example.com",
            password="Pass@12345",
            role="student",
        )
        self.institution_user = User.objects.create_user(
            username="institution_pdf",
            email="institution_pdf@example.com",
            password="Pass@12345",
            role="institution",
        )
        self.institution = Institution.objects.get(user=self.institution_user)
        self.department = Department.objects.create(name="CSE", institution=self.institution)
        self.teacher_profile = Teacher.objects.get(user=self.teacher)
        self.teacher_profile.department = self.department
        self.teacher_profile.save(update_fields=["department"])
        self.student_profile = Student.objects.get(user=self.student)
        self.student_profile.department = self.department
        self.student_profile.save(update_fields=["department"])
        self.institution.teachers.add(self.teacher_profile)
        self.institution.students.add(self.student_profile)
        self.subject = Subject.objects.create(
            true_subject=TrueSubject.objects.create(name="Algorithms", code="CS201"),
            semester=1,
            department=self.department,
            institution=self.institution,
            teacher=self.teacher_profile,
        )
        self.student_profile.subjects.add(self.subject)
        StudentUnderTeacher.objects.get_or_create(teacher=self.teacher, student=self.student)
        self.url = "/core/teacher/pdfs/upload"

    def _auth_headers(self, user):
        token = create_jwt_token(user)
        return {"HTTP_AUTHORIZATION": f"Bearer {token}"}

    def test_non_teacher_cannot_upload_pdf(self):
        with override_settings(MEDIA_ROOT=self.media_root):
            pdf_file = SimpleUploadedFile(
                "sample.pdf",
                b"%PDF-1.4\n%EOF",
                content_type="application/pdf",
            )
            response = self.client.post(
                self.url,
                {
                    "subject_id": str(self.subject.id),
                    "student_id": str(self.student_profile.id),
                    "pdf": pdf_file,
                },
                **self._auth_headers(self.student),
            )
            self.assertEqual(response.status_code, 403)

    def test_rejects_non_pdf_file(self):
        with override_settings(MEDIA_ROOT=self.media_root):
            text_file = SimpleUploadedFile(
                "sample.txt",
                b"hello world",
                content_type="text/plain",
            )
            response = self.client.post(
                self.url,
                {
                    "subject_id": str(self.subject.id),
                    "student_id": str(self.student_profile.id),
                    "pdf": text_file,
                },
                **self._auth_headers(self.teacher),
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(TeacherPDFUpload.objects.count(), 0)

    def test_teacher_can_upload_pdf_with_unique_name(self):
        with override_settings(MEDIA_ROOT=self.media_root):
            pdf_file = SimpleUploadedFile(
                "question-paper.pdf",
                b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF",
                content_type="application/pdf",
            )
            response = self.client.post(
                self.url,
                {
                    "subject_id": str(self.subject.id),
                    "student_id": str(self.student_profile.id),
                    "pdf": pdf_file,
                },
                **self._auth_headers(self.teacher),
            )
            self.assertEqual(response.status_code, 201)

            upload = TeacherPDFUpload.objects.get()
            self.assertEqual(upload.original_filename, "question-paper.pdf")
            self.assertEqual(upload.subject_id, self.subject.id)
            self.assertEqual(upload.student_id, self.student_profile.id)
            self.assertTrue(upload.file.name.startswith("uploads/"))
            self.assertTrue(upload.file.name.endswith(".pdf"))
            self.assertNotEqual(upload.file.name, "uploads/question-paper.pdf")


class TeacherAnswerKeyUploadTests(TestCase):
    def setUp(self):
        self.media_root = tempfile.mkdtemp(prefix="core-answerkey-test-media-")
        self.addCleanup(lambda: shutil.rmtree(self.media_root, ignore_errors=True))

        self.teacher_user = User.objects.create_user(
            username="teacher_answer_key",
            email="teacher_answer_key@example.com",
            password="Pass@12345",
            role="teacher",
        )
        self.other_teacher_user = User.objects.create_user(
            username="teacher_other_answer_key",
            email="teacher_other_answer_key@example.com",
            password="Pass@12345",
            role="teacher",
        )
        institution_user = User.objects.create_user(
            username="institution_answer_key",
            email="institution_answer_key@example.com",
            password="Pass@12345",
            role="institution",
        )

        self.institution = Institution.objects.get(user=institution_user)
        self.department = Department.objects.create(name="CSE", institution=self.institution)
        self.teacher = Teacher.objects.get(user=self.teacher_user)
        self.teacher.department = self.department
        self.teacher.save(update_fields=["department"])
        self.other_teacher = Teacher.objects.get(user=self.other_teacher_user)
        self.other_teacher.department = self.department
        self.other_teacher.save(update_fields=["department"])

        self.institution.teachers.add(self.teacher, self.other_teacher)

        self.true_subject = TrueSubject.objects.create(name="Mathematics", code="MATH101")
        self.subject = Subject.objects.create(
            true_subject=self.true_subject,
            semester=1,
            department=self.department,
            institution=self.institution,
            teacher=self.teacher,
        )
        self.other_subject = Subject.objects.create(
            true_subject=TrueSubject.objects.create(name="Physics", code="PHY101"),
            semester=1,
            department=self.department,
            institution=self.institution,
            teacher=self.other_teacher,
        )
        self.url = "/core/teacher/subjects/answer-key/upload"

    def _auth_headers(self, user):
        token = create_jwt_token(user)
        return {"HTTP_AUTHORIZATION": f"Bearer {token}"}

    def test_teacher_can_upload_answer_key_once_for_subject(self):
        with override_settings(MEDIA_ROOT=self.media_root):
            pdf_file = SimpleUploadedFile(
                "answer-key.pdf",
                b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF",
                content_type="application/pdf",
            )
            response = self.client.post(
                self.url,
                {"subject_id": str(self.subject.id), "pdf": pdf_file},
                **self._auth_headers(self.teacher_user),
            )
            self.assertEqual(response.status_code, 201)
            self.assertEqual(TeacherSubjectAnswerKey.objects.count(), 1)

            second_file = SimpleUploadedFile(
                "answer-key-v2.pdf",
                b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF",
                content_type="application/pdf",
            )
            second_response = self.client.post(
                self.url,
                {"subject_id": str(self.subject.id), "pdf": second_file},
                **self._auth_headers(self.teacher_user),
            )
            self.assertEqual(second_response.status_code, 200)
            self.assertEqual(TeacherSubjectAnswerKey.objects.count(), 1)
            payload = second_response.json()["message"]
            self.assertTrue(payload.get("updated"))
            self.assertEqual(payload.get("original_filename"), "answer-key-v2.pdf")

    def test_teacher_cannot_upload_answer_key_for_other_teacher_subject(self):
        with override_settings(MEDIA_ROOT=self.media_root):
            pdf_file = SimpleUploadedFile(
                "other-answer-key.pdf",
                b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF",
                content_type="application/pdf",
            )
            response = self.client.post(
                self.url,
                {"subject_id": str(self.other_subject.id), "pdf": pdf_file},
                **self._auth_headers(self.teacher_user),
            )
            self.assertEqual(response.status_code, 403)
            self.assertEqual(TeacherSubjectAnswerKey.objects.count(), 0)


class StudentMarksApiTests(TestCase):
    def setUp(self):
        self.student_user = User.objects.create_user(
            username="student_marks",
            email="student_marks@example.com",
            password="Pass@12345",
            role="student",
        )
        self.teacher_user = User.objects.create_user(
            username="teacher_marks",
            email="teacher_marks@example.com",
            password="Pass@12345",
            role="teacher",
        )
        institution_user = User.objects.create_user(
            username="institution_marks",
            email="institution_marks@example.com",
            password="Pass@12345",
            role="institution",
        )

        self.student = Student.objects.get(user=self.student_user)
        self.teacher = Teacher.objects.get(user=self.teacher_user)
        self.institution = Institution.objects.get(user=institution_user)
        self.department = Department.objects.create(name="ECE", institution=self.institution)
        self.student.department = self.department
        self.student.save(update_fields=["department"])
        self.teacher.department = self.department
        self.teacher.save(update_fields=["department"])

        self.sem1_subject = Subject.objects.create(
            true_subject=TrueSubject.objects.create(name="Signals", code="EC201"),
            semester=1,
            department=self.department,
            institution=self.institution,
            teacher=self.teacher,
        )
        self.sem2_subject = Subject.objects.create(
            true_subject=TrueSubject.objects.create(name="Networks", code="EC301"),
            semester=2,
            department=self.department,
            institution=self.institution,
            teacher=self.teacher,
        )
        self.student.subjects.add(self.sem1_subject, self.sem2_subject)

        StudentMark.objects.create(subject=self.sem1_subject, student=self.student, total_mark=10, acquired_mark=8)
        StudentMark.objects.create(subject=self.sem2_subject, student=self.student, total_mark=10, acquired_mark=7)

        self.marks_url = "/core/student/marks"
        self.options_url = "/core/student/marks/options"

    def _auth_headers(self, user):
        token = create_jwt_token(user)
        return {"HTTP_AUTHORIZATION": f"Bearer {token}"}

    def test_student_marks_supports_semester_and_subject_filters(self):
        response = self.client.get(
            self.marks_url,
            {"semester": 1, "subject_id": self.sem1_subject.id},
            **self._auth_headers(self.student_user),
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()["message"]
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["subject_id"], self.sem1_subject.id)
        self.assertEqual(payload[0]["subject__semester"], 1)

    def test_student_marks_options_returns_semesters_and_subjects(self):
        response = self.client.get(self.options_url, **self._auth_headers(self.student_user))
        self.assertEqual(response.status_code, 200)
        payload = response.json()["message"]
        self.assertEqual(payload["semesters"], [1, 2])
        self.assertEqual(payload["subject_count"], 2)

    def test_student_marks_rejects_invalid_semester(self):
        response = self.client.get(
            self.marks_url,
            {"semester": "invalid"},
            **self._auth_headers(self.student_user),
        )
        self.assertEqual(response.status_code, 400)


class TeacherStudentsMarkFlagTests(TestCase):
    def setUp(self):
        self.teacher_user = User.objects.create_user(
            username="teacher_flag",
            email="teacher_flag@example.com",
            password="Pass@12345",
            role="teacher",
        )
        self.student_marked_user = User.objects.create_user(
            username="student_marked",
            email="student_marked@example.com",
            password="Pass@12345",
            role="student",
        )
        self.student_unmarked_user = User.objects.create_user(
            username="student_unmarked",
            email="student_unmarked@example.com",
            password="Pass@12345",
            role="student",
        )
        institution_user = User.objects.create_user(
            username="institution_flag",
            email="institution_flag@example.com",
            password="Pass@12345",
            role="institution",
        )

        self.teacher = Teacher.objects.get(user=self.teacher_user)
        self.student_marked = Student.objects.get(user=self.student_marked_user)
        self.student_unmarked = Student.objects.get(user=self.student_unmarked_user)
        self.institution = Institution.objects.get(user=institution_user)
        self.department = Department.objects.create(name="MECH", institution=self.institution)
        self.teacher.department = self.department
        self.teacher.save(update_fields=["department"])
        self.student_marked.department = self.department
        self.student_marked.save(update_fields=["department"])
        self.student_unmarked.department = self.department
        self.student_unmarked.save(update_fields=["department"])

        self.subject = Subject.objects.create(
            true_subject=TrueSubject.objects.create(name="Thermo", code="ME201"),
            semester=3,
            department=self.department,
            institution=self.institution,
            teacher=self.teacher,
        )
        self.subject.students.add(self.student_marked, self.student_unmarked)

        StudentMark.objects.create(subject=self.subject, student=self.student_marked, total_mark=10, acquired_mark=9)
        self.url = "/core/teacher/students"

    def _auth_headers(self, user):
        token = create_jwt_token(user)
        return {"HTTP_AUTHORIZATION": f"Bearer {token}"}

    def test_teacher_students_includes_mark_completed_flag(self):
        response = self.client.get(self.url, **self._auth_headers(self.teacher_user))
        self.assertEqual(response.status_code, 200)
        payload = response.json()["message"]

        self.assertEqual(payload["subject_count"], 1)
        students = payload["subjects"][0]["students"]
        marks_by_student_id = {row["id"]: row["mark_completed"] for row in students}

        self.assertTrue(marks_by_student_id[self.student_marked.id])
        self.assertFalse(marks_by_student_id[self.student_unmarked.id])
