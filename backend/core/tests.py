import shutil
import tempfile

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from auth2.jwt_utils import create_jwt_token
from .models import Department, Institution, Subject, Teacher, TeacherPDFUpload, TeacherSubjectAnswerKey, TrueSubject

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
            response = self.client.post(self.url, {"pdf": pdf_file}, **self._auth_headers(self.student))
            self.assertEqual(response.status_code, 403)

    def test_rejects_non_pdf_file(self):
        with override_settings(MEDIA_ROOT=self.media_root):
            text_file = SimpleUploadedFile(
                "sample.txt",
                b"hello world",
                content_type="text/plain",
            )
            response = self.client.post(self.url, {"pdf": text_file}, **self._auth_headers(self.teacher))
            self.assertEqual(response.status_code, 400)
            self.assertEqual(TeacherPDFUpload.objects.count(), 0)

    def test_teacher_can_upload_pdf_with_unique_name(self):
        with override_settings(MEDIA_ROOT=self.media_root):
            pdf_file = SimpleUploadedFile(
                "question-paper.pdf",
                b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF",
                content_type="application/pdf",
            )
            response = self.client.post(self.url, {"pdf": pdf_file}, **self._auth_headers(self.teacher))
            self.assertEqual(response.status_code, 201)

            upload = TeacherPDFUpload.objects.get()
            self.assertEqual(upload.original_filename, "question-paper.pdf")
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
            self.assertEqual(second_response.status_code, 409)
            self.assertEqual(TeacherSubjectAnswerKey.objects.count(), 1)

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
