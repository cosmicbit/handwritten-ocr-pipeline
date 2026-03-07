import os

from django.db import IntegrityError
from django.db.models import Q

from auth2.models import Role, User
from core.models import (
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
#from core.engine.main import DEFAULT_STUDENT_PDF, DEFAULT_TEACHER_PDF, run_pipeline


class CoreServiceError(Exception):
    def __init__(self, message, status, extra=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.extra = extra or {}

    def to_response_body(self):
        payload = {"error": self.message}
        payload.update(self.extra)
        return payload


class CoreService:
    @staticmethod
    def _get_institution_for_user(user):
        return Institution.objects.filter(user=user).first()

    @staticmethod
    def _get_department_subjects(institution, department):
        return list(
            Subject.objects.filter(
                institution=institution,
                department=department,
            ).order_by("id")
        )

    def institution_search_users(self, role, q):
        if role not in {Role.TEACHER, Role.STUDENT}:
            raise CoreServiceError("role must be either 'teacher' or 'student'", 400)

        queryset = User.objects.filter(role=role)
        if q:
            queryset = queryset.filter(
                Q(username__icontains=q)
                | Q(email__icontains=q)
                | Q(first_name__icontains=q)
                | Q(last_name__icontains=q)
            )

        users = list(queryset.values("id", "username", "email", "first_name", "last_name", "role")[:50])
        return users, 200

    def institution_add_teacher(self, user, teacher_user_id, department_id):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        teacher_user = User.objects.filter(id=teacher_user_id, role=Role.TEACHER).first()
        if not teacher_user:
            raise CoreServiceError("Teacher user not found", 404)

        department = Department.objects.filter(id=department_id, institution=institution).first()
        if not department:
            raise CoreServiceError("Department not found", 404)

        teacher, _ = Teacher.objects.get_or_create(user=teacher_user, defaults={"department": department})
        if teacher.department_id != department.id:
            teacher.department = department
            teacher.save(update_fields=["department"])

        institution.teachers.add(teacher)

        return {
            "institution_id": institution.id,
            "teacher_id": teacher.id,
            "teacher_user_id": teacher_user.id,
            "department_id": department.id,
        }, 200

    def institution_add_student(self, user, student_user_id, department_id):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        student_user = User.objects.filter(id=student_user_id, role=Role.STUDENT).first()
        if not student_user:
            raise CoreServiceError("Student user not found", 404)

        department = Department.objects.filter(id=department_id, institution=institution).first()
        if not department:
            raise CoreServiceError("Department not found", 404)

        subjects = self._get_department_subjects(institution, department)

        student, _ = Student.objects.get_or_create(user=student_user, defaults={"department": department})
        if student.department_id != department.id:
            student.department = department
            student.save(update_fields=["department"])
        student.subjects.set(subjects)

        institution.students.add(student)

        return {
            "institution_id": institution.id,
            "student_id": student.id,
            "student_user_id": student_user.id,
            "department_id": department.id,
            "subject_ids": list(student.subjects.values_list("id", flat=True).order_by("id")),
        }, 200

    def institution_create_teacher(self, user, username, email, password, department_id):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        department = Department.objects.filter(id=department_id, institution=institution).first()
        if not department:
            raise CoreServiceError("Department not found", 404)

        try:
            teacher_user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                role=Role.TEACHER,
            )
        except IntegrityError:
            raise CoreServiceError("username or email already exists", 409)

        teacher = Teacher.objects.get(user=teacher_user)
        if teacher.department_id != department.id:
            teacher.department = department
            teacher.save(update_fields=["department"])

        institution.teachers.add(teacher)

        return {
            "user_id": teacher_user.id,
            "teacher_id": teacher.id,
            "institution_id": institution.id,
            "department_id": department.id,
            "role": teacher_user.role,
        }, 201

    def institution_create_student(self, user, username, email, password, department_id):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        department = Department.objects.filter(id=department_id, institution=institution).first()
        if not department:
            raise CoreServiceError("Department not found", 404)

        subjects = self._get_department_subjects(institution, department)

        try:
            student_user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                role=Role.STUDENT,
            )
        except IntegrityError:
            raise CoreServiceError("username or email already exists", 409)

        student = Student.objects.get(user=student_user)
        if student.department_id != department.id:
            student.department = department
            student.save(update_fields=["department"])
        student.subjects.set(subjects)

        institution.students.add(student)

        return {
            "user_id": student_user.id,
            "student_id": student.id,
            "institution_id": institution.id,
            "department_id": department.id,
            "subject_ids": list(student.subjects.values_list("id", flat=True).order_by("id")),
            "role": student_user.role,
        }, 201

    def institution_departments(self, user):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        departments = list(Department.objects.filter(institution=institution).values("id", "name").order_by("name"))
        return departments, 200

    def institution_members(self, user, role, q, page, page_size, offset):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        if role and role not in {"teacher", "student"}:
            raise CoreServiceError("role must be either 'teacher' or 'student'", 400)

        teachers_qs = institution.teachers.select_related("user", "department").all().order_by("id")
        students_qs = institution.students.select_related("user", "department").all().order_by("id")

        if q:
            teachers_qs = teachers_qs.filter(
                Q(user__username__icontains=q)
                | Q(user__email__icontains=q)
                | Q(user__first_name__icontains=q)
                | Q(user__last_name__icontains=q)
                | Q(department__name__icontains=q)
            )
            students_qs = students_qs.filter(
                Q(user__username__icontains=q)
                | Q(user__email__icontains=q)
                | Q(user__first_name__icontains=q)
                | Q(user__last_name__icontains=q)
                | Q(department__name__icontains=q)
            )

        teacher_total = teachers_qs.count()
        student_total = students_qs.count()

        teachers_page = teachers_qs[offset : offset + page_size]
        students_page = students_qs[offset : offset + page_size]

        teachers = [
            {
                "id": teacher.id,
                "user_id": teacher.user_id,
                "username": teacher.user.username,
                "email": teacher.user.email,
                "first_name": teacher.user.first_name,
                "last_name": teacher.user.last_name,
                "department_id": teacher.department_id,
                "department_name": teacher.department.name if teacher.department else None,
                "role": "teacher",
            }
            for teacher in teachers_page
        ]
        students = [
            {
                "id": student.id,
                "user_id": student.user_id,
                "username": student.user.username,
                "email": student.user.email,
                "first_name": student.user.first_name,
                "last_name": student.user.last_name,
                "department_id": student.department_id,
                "department_name": student.department.name if student.department else None,
                "role": "student",
            }
            for student in students_page
        ]

        if role == "teacher":
            return {
                "teachers": teachers,
                "count": len(teachers),
                "total": teacher_total,
                "offset": offset,
                "page": page,
                "page_size": page_size,
                "has_more": offset + len(teachers) < teacher_total,
            }, 200

        if role == "student":
            return {
                "students": students,
                "count": len(students),
                "total": student_total,
                "offset": offset,
                "page": page,
                "page_size": page_size,
                "has_more": offset + len(students) < student_total,
            }, 200

        return {
            "teachers": teachers,
            "students": students,
            "teacher_count": len(teachers),
            "student_count": len(students),
            "teacher_total": teacher_total,
            "student_total": student_total,
            "offset": offset,
            "page": page,
            "page_size": page_size,
            "teacher_has_more": offset + len(teachers) < teacher_total,
            "student_has_more": offset + len(students) < student_total,
        }, 200

    def institution_add_department(self, user, name):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        department, created = Department.objects.get_or_create(name=name, institution=institution)
        return {
            "id": department.id,
            "name": department.name,
            "created": created,
        }, 201 if created else 200

    def institution_add_subject(self, user, true_subject_id, semester, department_id, teacher_id):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        department = Department.objects.filter(id=department_id, institution=institution).first()
        if not department:
            raise CoreServiceError("Department not found", 404)

        teacher = Teacher.objects.filter(id=teacher_id).first()
        if not teacher:
            raise CoreServiceError("Teacher not found", 404)

        if not institution.teachers.filter(id=teacher.id).exists():
            raise CoreServiceError("Teacher is not part of your institution", 403)

        true_subject = TrueSubject.objects.filter(id=true_subject_id).first()
        if not true_subject:
            raise CoreServiceError("TrueSubject not found", 404)

        try:
            semester_value = int(semester)
        except (TypeError, ValueError):
            raise CoreServiceError("Invalid semester value", 400)

        if semester_value not in {choice[0] for choice in Subject.Semester.choices}:
            raise CoreServiceError("Invalid semester value", 400)

        subject, created = Subject.objects.get_or_create(
            institution=institution,
            true_subject=true_subject,
            semester=semester_value,
            department=department,
            defaults={"teacher": teacher},
        )

        if not created:
            raise CoreServiceError(
                "Subject already set for this institution/department/semester",
                409,
                {"assigned_teacher_id": subject.teacher_id},
            )

        return {
            "id": subject.id,
            "true_subject_id": subject.true_subject_id,
            "name": subject.true_subject.name,
            "code": subject.true_subject.code,
            "institution_id": subject.institution_id,
            "semester": subject.semester,
            "department_id": subject.department_id,
            "teacher_id": subject.teacher_id,
        }, 201

    def institution_update_teacher(self, user, teacher_id, department_id):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        teacher = institution.teachers.filter(id=teacher_id).first()
        if not teacher:
            raise CoreServiceError("Teacher not found in your institution", 404)

        department = Department.objects.filter(id=department_id, institution=institution).first()
        if not department:
            raise CoreServiceError("Department not found", 404)

        teacher.department = department
        teacher.save(update_fields=["department"])

        return {
            "teacher_id": teacher.id,
            "department_id": teacher.department_id,
        }, 200

    def institution_remove_teacher(self, user, teacher_id):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        teacher = institution.teachers.filter(id=teacher_id).first()
        if not teacher:
            raise CoreServiceError("Teacher not found in your institution", 404)

        assigned_subjects_count = Subject.objects.filter(institution=institution, teacher=teacher).count()
        if assigned_subjects_count > 0:
            raise CoreServiceError(
                "Teacher has assigned subjects in this institution. Reassign or remove them first.",
                409,
                {"assigned_subjects_count": assigned_subjects_count},
            )

        institution.teachers.remove(teacher)
        return {
            "teacher_id": teacher.id,
            "removed": True,
        }, 200

    def institution_update_student(self, user, student_id, department_id):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        student = institution.students.filter(id=student_id).first()
        if not student:
            raise CoreServiceError("Student not found in your institution", 404)

        department = Department.objects.filter(id=department_id, institution=institution).first()
        if not department:
            raise CoreServiceError("Department not found", 404)

        student.department = department
        student.save(update_fields=["department"])

        return {
            "student_id": student.id,
            "department_id": student.department_id,
        }, 200

    def institution_remove_student(self, user, student_id):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        student = institution.students.filter(id=student_id).first()
        if not student:
            raise CoreServiceError("Student not found in your institution", 404)

        institution.students.remove(student)
        return {
            "student_id": student.id,
            "removed": True,
        }, 200

    def institution_update_department(self, user, department_id, name):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        department = Department.objects.filter(id=department_id, institution=institution).first()
        if not department:
            raise CoreServiceError("Department not found", 404)

        department.name = name
        try:
            department.save(update_fields=["name"])
        except IntegrityError:
            raise CoreServiceError("Department with this name already exists in your institution", 409)

        return {
            "department_id": department.id,
            "name": department.name,
        }, 200

    def institution_remove_department(self, user, department_id):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        department = Department.objects.filter(id=department_id, institution=institution).first()
        if not department:
            raise CoreServiceError("Department not found", 404)

        teacher_count = Teacher.objects.filter(department=department).count()
        student_count = Student.objects.filter(department=department).count()
        subject_count = Subject.objects.filter(institution=institution, department=department).count()

        if teacher_count > 0 or student_count > 0 or subject_count > 0:
            raise CoreServiceError(
                "Department is in use. Reassign/remove linked records first.",
                409,
                {
                    "teacher_count": teacher_count,
                    "student_count": student_count,
                    "subject_count": subject_count,
                },
            )

        department.delete()
        return {
            "department_id": department_id,
            "removed": True,
        }, 200

    def institution_update_subject_assignment(self, user, subject_id, teacher_id):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        subject = Subject.objects.filter(id=subject_id, institution=institution).first()
        if not subject:
            raise CoreServiceError("Subject not found", 404)

        teacher = institution.teachers.filter(id=teacher_id).first()
        if not teacher:
            raise CoreServiceError("Teacher not found in your institution", 404)

        subject.teacher = teacher
        subject.save(update_fields=["teacher"])

        return {
            "subject_id": subject.id,
            "teacher_id": subject.teacher_id,
        }, 200

    def institution_remove_subject(self, user, subject_id):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        subject = Subject.objects.filter(id=subject_id, institution=institution).first()
        if not subject:
            raise CoreServiceError("Subject not found", 404)

        subject.delete()
        return {
            "subject_id": subject_id,
            "removed": True,
        }, 200

    def institution_true_subjects(self, q):
        queryset = TrueSubject.objects.all().order_by("code")
        if q:
            queryset = queryset.filter(Q(name__icontains=q) | Q(code__icontains=q))
        return list(queryset.values("id", "name", "code")), 200

    def institution_subjects(self, user, q, page, page_size, offset):
        institution = self._get_institution_for_user(user)
        if not institution:
            raise CoreServiceError("Institution profile not found", 404)

        queryset = (
            Subject.objects.select_related("true_subject", "department", "teacher", "teacher__user")
            .filter(institution=institution)
            .order_by("id")
        )

        if q:
            queryset = queryset.filter(
                Q(true_subject__name__icontains=q)
                | Q(true_subject__code__icontains=q)
                | Q(department__name__icontains=q)
                | Q(teacher__user__username__icontains=q)
                | Q(teacher__user__email__icontains=q)
            )

        total = queryset.count()
        page_items = queryset[offset : offset + page_size]

        subjects = [
            {
                "id": subject.id,
                "true_subject_id": subject.true_subject_id,
                "name": subject.true_subject.name,
                "code": subject.true_subject.code,
                "semester": subject.semester,
                "institution_id": subject.institution_id,
                "department_id": subject.department_id,
                "department_name": subject.department.name if subject.department else None,
                "teacher_id": subject.teacher_id,
                "teacher_user_id": subject.teacher.user_id if subject.teacher else None,
                "teacher_username": subject.teacher.user.username if subject.teacher and subject.teacher.user else None,
            }
            for subject in page_items
        ]

        return {
            "subjects": subjects,
            "count": len(subjects),
            "total": total,
            "offset": offset,
            "page": page,
            "page_size": page_size,
            "has_more": offset + len(subjects) < total,
        }, 200

    def teacher_upload_pdf(self, user, uploaded_file, build_absolute_uri):
        if not uploaded_file:
            raise CoreServiceError("pdf file is required in multipart form-data", 400)

        extension = os.path.splitext(uploaded_file.name)[1].lower()
        if extension != ".pdf":
            raise CoreServiceError("Only PDF files are allowed", 400)

        file_header = uploaded_file.read(5)
        uploaded_file.seek(0)
        if file_header != b"%PDF-":
            raise CoreServiceError("Invalid PDF file", 400)

        upload = TeacherPDFUpload(
            teacher=user,
            original_filename=uploaded_file.name,
        )
        upload.file.save(uploaded_file.name, uploaded_file, save=False)
        upload.save()

        return {
            "id": upload.id,
            "teacher_user_id": upload.teacher_id,
            "original_filename": upload.original_filename,
            "stored_filename": os.path.basename(upload.file.name),
            "file_path": upload.file.name,
            "file_url": build_absolute_uri(upload.file.url),
        }, 201

    def teacher_upload_answer_key(self, user, subject_id, uploaded_file, build_absolute_uri):
        subject = Subject.objects.select_related("teacher", "teacher__user").filter(id=subject_id).first()
        if not subject:
            raise CoreServiceError("Subject not found", 404)

        if not subject.teacher or subject.teacher.user_id != user.id:
            raise CoreServiceError("You can upload answer key only for your own subject", 403)

        if TeacherSubjectAnswerKey.objects.filter(subject=subject).exists():
            raise CoreServiceError("Answer key already uploaded for this subject", 409)

        if not uploaded_file:
            raise CoreServiceError("pdf file is required in multipart form-data", 400)

        extension = os.path.splitext(uploaded_file.name)[1].lower()
        if extension != ".pdf":
            raise CoreServiceError("Only PDF files are allowed", 400)

        file_header = uploaded_file.read(5)
        uploaded_file.seek(0)
        if file_header != b"%PDF-":
            raise CoreServiceError("Invalid PDF file", 400)

        answer_key = TeacherSubjectAnswerKey(
            teacher=user,
            subject=subject,
            original_filename=uploaded_file.name,
        )
        answer_key.file.save(uploaded_file.name, uploaded_file, save=False)

        try:
            answer_key.save()
        except IntegrityError:
            raise CoreServiceError("Answer key already uploaded for this subject", 409)

        return {
            "id": answer_key.id,
            "teacher_user_id": answer_key.teacher_id,
            "subject_id": answer_key.subject_id,
            "original_filename": answer_key.original_filename,
            "stored_filename": os.path.basename(answer_key.file.name),
            "file_path": answer_key.file.name,
            "file_url": build_absolute_uri(answer_key.file.url),
        }, 201

    def teacher_assign_student(self, user, student_user_id):
        student_user = User.objects.filter(id=student_user_id, role=Role.STUDENT).first()
        if not student_user:
            raise CoreServiceError("Student user not found", 404)

        link, created = StudentUnderTeacher.objects.get_or_create(
            teacher=user,
            student=student_user,
        )

        return {
            "id": link.id,
            "teacher_user_id": link.teacher_id,
            "student_user_id": link.student_id,
            "created": created,
        }, 201 if created else 200

    def teacher_students(self, user, q):
        teacher = Teacher.objects.filter(user=user).first()
        if not teacher:
            raise CoreServiceError("Teacher profile not found", 404)

        subjects_qs = (
            Subject.objects.filter(teacher=teacher)
            .select_related("true_subject", "department", "institution")
            .prefetch_related("students__user", "students__department")
            .order_by("id")
        )

        subjects_payload = []
        unique_student_ids = set()
        for subject in subjects_qs:
            students_payload = []
            for student in subject.students.all():
                if q:
                    haystack = " ".join(
                        [
                            (student.user.username or ""),
                            (student.user.email or ""),
                            (student.user.first_name or ""),
                            (student.user.last_name or ""),
                            (student.department.name if student.department else ""),
                            (subject.true_subject.name if subject.true_subject else ""),
                            (subject.true_subject.code if subject.true_subject else ""),
                        ]
                    ).lower()
                    if q not in haystack:
                        continue

                students_payload.append(
                    {
                        "id": student.id,
                        "user_id": student.user_id,
                        "username": student.user.username,
                        "email": student.user.email,
                        "first_name": student.user.first_name,
                        "last_name": student.user.last_name,
                        "department_id": student.department_id,
                        "department_name": student.department.name if student.department else None,
                        "role": "student",
                    }
                )
                unique_student_ids.add(student.id)

            if q and not students_payload:
                continue

            subjects_payload.append(
                {
                    "subject_id": subject.id,
                    "true_subject_id": subject.true_subject_id,
                    "subject_name": subject.true_subject.name if subject.true_subject else None,
                    "subject_code": subject.true_subject.code if subject.true_subject else None,
                    "semester": subject.semester,
                    "department_id": subject.department_id,
                    "department_name": subject.department.name if subject.department else None,
                    "institution_id": subject.institution_id,
                    "students": students_payload,
                    "student_count": len(students_payload),
                }
            )

        return {
            "subjects": subjects_payload,
            "subject_count": len(subjects_payload),
            "student_count": len(unique_student_ids),
        }, 200

    def student_marks(self, user):
        student = Student.objects.filter(user=user).first()
        if not student:
            raise CoreServiceError("Student profile not found", 404)

        marks = (
            StudentMark.objects.filter(student=student)
            .select_related("subject", "subject__true_subject")
            .values(
                "id",
                "subject_id",
                "subject__true_subject__name",
                "subject__true_subject__code",
                "total_mark",
                "acquired_mark",
                "created_at",
            )
        )

        return list(marks), 200

    def trigger_engine_model(self, teacher_pdf_path=None, student_pdf_path=None, marks=None):
    #    #teacher_path = teacher_pdf_path or DEFAULT_TEACHER_PDF
    #     student_path = student_pdf_path or DEFAULT_STUDENT_PDF

    #     if not os.path.isfile(teacher_path):
    #         raise CoreServiceError("Teacher PDF file not found", 404, {"teacher_pdf_path": teacher_path})
    #     if not os.path.isfile(student_path):
    #         raise CoreServiceError("Student PDF file not found", 404, {"student_pdf_path": student_path})

    #     if marks is not None:
    #         if not isinstance(marks, list) or not marks:
    #             raise CoreServiceError("marks must be a non-empty list", 400)
    #         if not all(isinstance(mark, (int, float)) for mark in marks):
    #             raise CoreServiceError("marks must contain only numbers", 400)

    #     try:
    #         teacher_answers, every_student_answers, every_student_scores = run_pipeline(
    #             teacher_pdf_path=teacher_path,
    #             student_pdf_path=student_path,
    #             marks=marks,
    #         )
    #     except Exception as exc:
    #         raise CoreServiceError("Model execution failed", 500, {"details": str(exc)})

    #     student_answers = every_student_answers[0] if every_student_answers else []
    #     student_scores = every_student_scores[0] if every_student_scores else []
    #     return {
    #         "teacher_pdf_path": teacher_path,
    #         "student_pdf_path": student_path,
    #         "teacher_answers_count": len(teacher_answers),
    #         "student_answers_count": len(student_answers),
    #         "scores": student_scores,
    #         "total_score": sum(student_scores) if student_scores else 0,
    #     }, 200
        return {
            "teacher_pdf_path": None
        }
