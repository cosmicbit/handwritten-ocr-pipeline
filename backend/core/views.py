import json
import os

from django.db.models import Q
from django.db import IntegrityError
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from auth2.models import Role, User
from rbac.permissions import has_permission

from .models import Department, Institution, Student, StudentMark, StudentUnderTeacher, Subject, Teacher, TeacherPDFUpload, TeacherSubjectAnswerKey, TrueSubject


def _parse_json(request):
    try:
        return json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def _require_role(request, role_name):
    user = getattr(request, "user", None)
    if not user:
        return JsonResponse({"error": "Authentication required"}, status=401)
    if str(getattr(user, "role", "")).lower() != role_name:
        return JsonResponse({"error": f"{role_name.title()} role required"}, status=403)
    return None


def _get_institution_for_user(user):
    return Institution.objects.filter(user=user).first()


def _get_department_subjects(institution, department):
    return list(
        Subject.objects.filter(
            institution=institution,
            department=department,
        ).order_by("id")
    )


@require_GET
def test(req):
    return JsonResponse({"success": "ok"}, status=200)


@csrf_exempt
@require_GET
def institution_search_users(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    role = str(request.GET.get("role", "")).lower().strip()
    if role not in {Role.TEACHER, Role.STUDENT}:
        return JsonResponse({"error": "role must be either 'teacher' or 'student'"}, status=400)

    q = request.GET.get("q", "").strip()

    queryset = User.objects.filter(role=role)
    if q:
        queryset = queryset.filter(
            Q(username__icontains=q)
            | Q(email__icontains=q)
            | Q(first_name__icontains=q)
            | Q(last_name__icontains=q)
        )

    users = list(queryset.values("id", "username", "email", "first_name", "last_name", "role")[:50])
    return JsonResponse({"message": users}, status=200)


@csrf_exempt
@require_POST
def institution_add_teacher(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    teacher_user_id = data.get("teacher_user_id")
    department_id = data.get("department_id")
    if not teacher_user_id or not department_id:
        return JsonResponse({"error": "teacher_user_id and department_id are required"}, status=400)

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    teacher_user = User.objects.filter(id=teacher_user_id, role=Role.TEACHER).first()
    if not teacher_user:
        return JsonResponse({"error": "Teacher user not found"}, status=404)

    department = Department.objects.filter(id=department_id, institution=institution).first()
    if not department:
        return JsonResponse({"error": "Department not found"}, status=404)

    teacher, _ = Teacher.objects.get_or_create(user=teacher_user, defaults={"department": department})
    if teacher.department_id != department.id:
        teacher.department = department
        teacher.save(update_fields=["department"])

    institution.teachers.add(teacher)

    return JsonResponse(
        {
            "message": {
                "institution_id": institution.id,
                "teacher_id": teacher.id,
                "teacher_user_id": teacher_user.id,
                "department_id": department.id,
            }
        },
        status=200,
    )


@csrf_exempt
@require_POST
def institution_add_student(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    student_user_id = data.get("student_user_id")
    department_id = data.get("department_id")
    if not student_user_id or not department_id:
        return JsonResponse({"error": "student_user_id and department_id are required"}, status=400)

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    student_user = User.objects.filter(id=student_user_id, role=Role.STUDENT).first()
    if not student_user:
        return JsonResponse({"error": "Student user not found"}, status=404)

    department = Department.objects.filter(id=department_id, institution=institution).first()
    if not department:
        return JsonResponse({"error": "Department not found"}, status=404)
    subjects = _get_department_subjects(institution, department)

    student, _ = Student.objects.get_or_create(user=student_user, defaults={"department": department})
    if student.department_id != department.id:
        student.department = department
        student.save(update_fields=["department"])
    student.subjects.set(subjects)

    institution.students.add(student)

    return JsonResponse(
        {
            "message": {
                "institution_id": institution.id,
                "student_id": student.id,
                "student_user_id": student_user.id,
                "department_id": department.id,
                "subject_ids": list(student.subjects.values_list("id", flat=True).order_by("id")),
            }
        },
        status=200,
    )


@csrf_exempt
@require_POST
def institution_create_teacher(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = data.get("password")
    department_id = data.get("department_id")

    if not all([username, email, password, department_id]):
        return JsonResponse(
            {"error": "username, email, password, and department_id are required"},
            status=400,
        )

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    department = Department.objects.filter(id=department_id, institution=institution).first()
    if not department:
        return JsonResponse({"error": "Department not found"}, status=404)

    try:
        teacher_user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            role=Role.TEACHER,
        )
    except IntegrityError:
        return JsonResponse({"error": "username or email already exists"}, status=409)

    teacher = Teacher.objects.get(user=teacher_user)
    if teacher.department_id != department.id:
        teacher.department = department
        teacher.save(update_fields=["department"])

    institution.teachers.add(teacher)

    return JsonResponse(
        {
            "message": {
                "user_id": teacher_user.id,
                "teacher_id": teacher.id,
                "institution_id": institution.id,
                "department_id": department.id,
                "role": teacher_user.role,
            }
        },
        status=201,
    )


@csrf_exempt
@require_POST
def institution_create_student(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = data.get("password")
    department_id = data.get("department_id")

    if not all([username, email, password, department_id]):
        return JsonResponse(
            {"error": "username, email, password, and department_id are required"},
            status=400,
        )

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    department = Department.objects.filter(id=department_id, institution=institution).first()
    if not department:
        return JsonResponse({"error": "Department not found"}, status=404)
    subjects = _get_department_subjects(institution, department)

    try:
        student_user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            role=Role.STUDENT,
        )
    except IntegrityError:
        return JsonResponse({"error": "username or email already exists"}, status=409)

    student = Student.objects.get(user=student_user)
    if student.department_id != department.id:
        student.department = department
        student.save(update_fields=["department"])
    student.subjects.set(subjects)

    institution.students.add(student)

    return JsonResponse(
        {
            "message": {
                "user_id": student_user.id,
                "student_id": student.id,
                "institution_id": institution.id,
                "department_id": department.id,
                "subject_ids": list(student.subjects.values_list("id", flat=True).order_by("id")),
                "role": student_user.role,
            }
        },
        status=201,
    )


@csrf_exempt
@require_GET
def institution_departments(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    departments = list(
        Department.objects.filter(institution=institution).values("id", "name").order_by("name")
    )
    return JsonResponse({"message": departments}, status=200)


@csrf_exempt
@require_GET
def institution_members(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    role = str(request.GET.get("role", "")).strip().lower()
    q = str(request.GET.get("q", "")).strip()
    page_size_raw = request.GET.get("page_size", 20)
    page_raw = request.GET.get("page", 1)
    offset_raw = request.GET.get("offset")

    try:
        page_size = int(page_size_raw)
        page = int(page_raw)
        if offset_raw is None:
            offset = (page - 1) * page_size
        else:
            offset = int(offset_raw)
    except (TypeError, ValueError):
        return JsonResponse({"error": "page, page_size, and offset must be integers"}, status=400)

    if page_size <= 0 or page <= 0 or offset < 0:
        return JsonResponse({"error": "page and page_size must be > 0, offset must be >= 0"}, status=400)

    if role and role not in {"teacher", "student"}:
        return JsonResponse({"error": "role must be either 'teacher' or 'student'"}, status=400)

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
        return JsonResponse(
            {
                "message": {
                    "teachers": teachers,
                    "count": len(teachers),
                    "total": teacher_total,
                    "offset": offset,
                    "page": page,
                    "page_size": page_size,
                    "has_more": offset + len(teachers) < teacher_total,
                }
            },
            status=200,
        )
    if role == "student":
        return JsonResponse(
            {
                "message": {
                    "students": students,
                    "count": len(students),
                    "total": student_total,
                    "offset": offset,
                    "page": page,
                    "page_size": page_size,
                    "has_more": offset + len(students) < student_total,
                }
            },
            status=200,
        )

    return JsonResponse(
        {
            "message": {
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
            }
        },
        status=200,
    )


@csrf_exempt
@require_POST
def institution_add_department(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    name = str(data.get("name", "")).strip()
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    department, created = Department.objects.get_or_create(
        name=name,
        institution=institution,
    )
    return JsonResponse(
        {
            "message": {
                "id": department.id,
                "name": department.name,
                "created": created,
            }
        },
        status=201 if created else 200,
    )


@csrf_exempt
@require_POST
def institution_add_subject(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    true_subject_id = data.get("true_subject_id")
    semester = data.get("semester")
    department_id = data.get("department_id")
    teacher_id = data.get("teacher_id")

    if not all([true_subject_id, semester, department_id, teacher_id]):
        return JsonResponse(
            {"error": "true_subject_id, semester, department_id, and teacher_id are required"},
            status=400,
        )

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    department = Department.objects.filter(id=department_id, institution=institution).first()
    if not department:
        return JsonResponse({"error": "Department not found"}, status=404)

    teacher = Teacher.objects.filter(id=teacher_id).first()
    if not teacher:
        return JsonResponse({"error": "Teacher not found"}, status=404)
    if not institution.teachers.filter(id=teacher.id).exists():
        return JsonResponse({"error": "Teacher is not part of your institution"}, status=403)

    true_subject = TrueSubject.objects.filter(id=true_subject_id).first()
    if not true_subject:
        return JsonResponse({"error": "TrueSubject not found"}, status=404)

    if int(semester) not in {choice[0] for choice in Subject.Semester.choices}:
        return JsonResponse({"error": "Invalid semester value"}, status=400)

    subject, created = Subject.objects.get_or_create(
        institution=institution,
        true_subject=true_subject,
        semester=int(semester),
        department=department,
        defaults={
            "teacher": teacher,
        },
    )

    if not created:
        return JsonResponse(
            {
                "error": "Subject already set for this institution/department/semester",
                "assigned_teacher_id": subject.teacher_id,
            },
            status=409,
        )

    return JsonResponse(
        {
            "message": {
                "id": subject.id,
                "true_subject_id": subject.true_subject_id,
                "name": subject.true_subject.name,
                "code": subject.true_subject.code,
                "institution_id": subject.institution_id,
                "semester": subject.semester,
                "department_id": subject.department_id,
                "teacher_id": subject.teacher_id,
            }
        },
        status=201,
    )


@csrf_exempt
@require_POST
def institution_update_teacher(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    teacher_id = data.get("teacher_id")
    department_id = data.get("department_id")
    if not teacher_id or not department_id:
        return JsonResponse({"error": "teacher_id and department_id are required"}, status=400)

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    teacher = institution.teachers.filter(id=teacher_id).first()
    if not teacher:
        return JsonResponse({"error": "Teacher not found in your institution"}, status=404)

    department = Department.objects.filter(id=department_id, institution=institution).first()
    if not department:
        return JsonResponse({"error": "Department not found"}, status=404)

    teacher.department = department
    teacher.save(update_fields=["department"])

    return JsonResponse(
        {
            "message": {
                "teacher_id": teacher.id,
                "department_id": teacher.department_id,
            }
        },
        status=200,
    )


@csrf_exempt
@require_POST
def institution_remove_teacher(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    teacher_id = data.get("teacher_id")
    if not teacher_id:
        return JsonResponse({"error": "teacher_id is required"}, status=400)

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    teacher = institution.teachers.filter(id=teacher_id).first()
    if not teacher:
        return JsonResponse({"error": "Teacher not found in your institution"}, status=404)

    assigned_subjects_count = Subject.objects.filter(institution=institution, teacher=teacher).count()
    if assigned_subjects_count > 0:
        return JsonResponse(
            {
                "error": "Teacher has assigned subjects in this institution. Reassign or remove them first.",
                "assigned_subjects_count": assigned_subjects_count,
            },
            status=409,
        )

    institution.teachers.remove(teacher)
    return JsonResponse(
        {"message": {"teacher_id": teacher.id, "removed": True}},
        status=200,
    )


@csrf_exempt
@require_POST
def institution_update_student(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    student_id = data.get("student_id")
    department_id = data.get("department_id")
    if not student_id or not department_id:
        return JsonResponse({"error": "student_id and department_id are required"}, status=400)

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    student = institution.students.filter(id=student_id).first()
    if not student:
        return JsonResponse({"error": "Student not found in your institution"}, status=404)

    department = Department.objects.filter(id=department_id, institution=institution).first()
    if not department:
        return JsonResponse({"error": "Department not found"}, status=404)

    student.department = department
    student.save(update_fields=["department"])

    return JsonResponse(
        {
            "message": {
                "student_id": student.id,
                "department_id": student.department_id,
            }
        },
        status=200,
    )


@csrf_exempt
@require_POST
def institution_remove_student(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    student_id = data.get("student_id")
    if not student_id:
        return JsonResponse({"error": "student_id is required"}, status=400)

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    student = institution.students.filter(id=student_id).first()
    if not student:
        return JsonResponse({"error": "Student not found in your institution"}, status=404)

    institution.students.remove(student)
    return JsonResponse(
        {"message": {"student_id": student.id, "removed": True}},
        status=200,
    )


@csrf_exempt
@require_POST
def institution_update_department(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    department_id = data.get("department_id")
    name = str(data.get("name", "")).strip()
    if not department_id or not name:
        return JsonResponse({"error": "department_id and name are required"}, status=400)

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    department = Department.objects.filter(id=department_id, institution=institution).first()
    if not department:
        return JsonResponse({"error": "Department not found"}, status=404)

    department.name = name
    try:
        department.save(update_fields=["name"])
    except IntegrityError:
        return JsonResponse(
            {"error": "Department with this name already exists in your institution"},
            status=409,
        )

    return JsonResponse(
        {"message": {"department_id": department.id, "name": department.name}},
        status=200,
    )


@csrf_exempt
@require_POST
def institution_remove_department(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    department_id = data.get("department_id")
    if not department_id:
        return JsonResponse({"error": "department_id is required"}, status=400)

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    department = Department.objects.filter(id=department_id, institution=institution).first()
    if not department:
        return JsonResponse({"error": "Department not found"}, status=404)

    teacher_count = Teacher.objects.filter(department=department).count()
    student_count = Student.objects.filter(department=department).count()
    subject_count = Subject.objects.filter(institution=institution, department=department).count()
    if teacher_count > 0 or student_count > 0 or subject_count > 0:
        return JsonResponse(
            {
                "error": "Department is in use. Reassign/remove linked records first.",
                "teacher_count": teacher_count,
                "student_count": student_count,
                "subject_count": subject_count,
            },
            status=409,
        )

    department.delete()
    return JsonResponse(
        {"message": {"department_id": department_id, "removed": True}},
        status=200,
    )


@csrf_exempt
@require_POST
def institution_update_subject_assignment(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    subject_id = data.get("subject_id")
    teacher_id = data.get("teacher_id")
    if not subject_id or not teacher_id:
        return JsonResponse({"error": "subject_id and teacher_id are required"}, status=400)

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    subject = Subject.objects.filter(id=subject_id, institution=institution).first()
    if not subject:
        return JsonResponse({"error": "Subject not found"}, status=404)

    teacher = institution.teachers.filter(id=teacher_id).first()
    if not teacher:
        return JsonResponse({"error": "Teacher not found in your institution"}, status=404)

    subject.teacher = teacher
    subject.save(update_fields=["teacher"])

    return JsonResponse(
        {
            "message": {
                "subject_id": subject.id,
                "teacher_id": subject.teacher_id,
            }
        },
        status=200,
    )


@csrf_exempt
@require_POST
def institution_remove_subject(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    subject_id = data.get("subject_id")
    if not subject_id:
        return JsonResponse({"error": "subject_id is required"}, status=400)

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    subject = Subject.objects.filter(id=subject_id, institution=institution).first()
    if not subject:
        return JsonResponse({"error": "Subject not found"}, status=404)

    subject.delete()
    return JsonResponse(
        {"message": {"subject_id": subject_id, "removed": True}},
        status=200,
    )


@csrf_exempt
@require_GET
def institution_true_subjects(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    q = str(request.GET.get("q", "")).strip()
    queryset = TrueSubject.objects.all().order_by("code")
    if q:
        queryset = queryset.filter(Q(name__icontains=q) | Q(code__icontains=q))

    payload = list(queryset.values("id", "name", "code"))
    return JsonResponse({"message": payload}, status=200)


@csrf_exempt
@require_GET
def institution_subjects(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    institution = _get_institution_for_user(request.user)
    if not institution:
        return JsonResponse({"error": "Institution profile not found"}, status=404)

    q = str(request.GET.get("q", "")).strip()
    page_size_raw = request.GET.get("page_size", 20)
    page_raw = request.GET.get("page", 1)
    offset_raw = request.GET.get("offset")

    try:
        page_size = int(page_size_raw)
        page = int(page_raw)
        if offset_raw is None:
            offset = (page - 1) * page_size
        else:
            offset = int(offset_raw)
    except (TypeError, ValueError):
        return JsonResponse({"error": "page, page_size, and offset must be integers"}, status=400)

    if page_size <= 0 or page <= 0 or offset < 0:
        return JsonResponse({"error": "page and page_size must be > 0, offset must be >= 0"}, status=400)

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

    return JsonResponse(
        {
            "message": {
                "subjects": subjects,
                "count": len(subjects),
                "total": total,
                "offset": offset,
                "page": page,
                "page_size": page_size,
                "has_more": offset + len(subjects) < total,
            }
        },
        status=200,
    )


@csrf_exempt
@require_POST
@has_permission()
def teacher_upload_pdf(request):
    auth_error = _require_role(request, "teacher")
    if auth_error:
        return auth_error

    uploaded_file = request.FILES.get("pdf")
    if not uploaded_file:
        return JsonResponse({"error": "pdf file is required in multipart form-data"}, status=400)

    extension = os.path.splitext(uploaded_file.name)[1].lower()
    if extension != ".pdf":
        return JsonResponse({"error": "Only PDF files are allowed"}, status=400)

    file_header = uploaded_file.read(5)
    uploaded_file.seek(0)
    if file_header != b"%PDF-":
        return JsonResponse({"error": "Invalid PDF file"}, status=400)

    upload = TeacherPDFUpload(
        teacher=request.user,
        original_filename=uploaded_file.name,
    )
    upload.file.save(uploaded_file.name, uploaded_file, save=False)
    upload.save()

    return JsonResponse(
        {
            "message": {
                "id": upload.id,
                "teacher_user_id": upload.teacher_id,
                "original_filename": upload.original_filename,
                "stored_filename": os.path.basename(upload.file.name),
                "file_path": upload.file.name,
                "file_url": request.build_absolute_uri(upload.file.url),
            }
        },
        status=201,
    )


@csrf_exempt
@require_POST
@has_permission()
def teacher_upload_answer_key(request):
    auth_error = _require_role(request, "teacher")
    if auth_error:
        return auth_error

    subject_id = request.POST.get("subject_id")
    if not subject_id:
        return JsonResponse({"error": "subject_id is required"}, status=400)

    subject = (
        Subject.objects.select_related("teacher", "teacher__user")
        .filter(id=subject_id)
        .first()
    )
    if not subject:
        return JsonResponse({"error": "Subject not found"}, status=404)

    if not subject.teacher or subject.teacher.user_id != request.user.id:
        return JsonResponse({"error": "You can upload answer key only for your own subject"}, status=403)

    if TeacherSubjectAnswerKey.objects.filter(subject=subject).exists():
        return JsonResponse({"error": "Answer key already uploaded for this subject"}, status=409)

    uploaded_file = request.FILES.get("pdf")
    if not uploaded_file:
        return JsonResponse({"error": "pdf file is required in multipart form-data"}, status=400)

    extension = os.path.splitext(uploaded_file.name)[1].lower()
    if extension != ".pdf":
        return JsonResponse({"error": "Only PDF files are allowed"}, status=400)

    file_header = uploaded_file.read(5)
    uploaded_file.seek(0)
    if file_header != b"%PDF-":
        return JsonResponse({"error": "Invalid PDF file"}, status=400)

    answer_key = TeacherSubjectAnswerKey(
        teacher=request.user,
        subject=subject,
        original_filename=uploaded_file.name,
    )
    answer_key.file.save(uploaded_file.name, uploaded_file, save=False)

    try:
        answer_key.save()
    except IntegrityError:
        return JsonResponse({"error": "Answer key already uploaded for this subject"}, status=409)

    return JsonResponse(
        {
            "message": {
                "id": answer_key.id,
                "teacher_user_id": answer_key.teacher_id,
                "subject_id": answer_key.subject_id,
                "original_filename": answer_key.original_filename,
                "stored_filename": os.path.basename(answer_key.file.name),
                "file_path": answer_key.file.name,
                "file_url": request.build_absolute_uri(answer_key.file.url),
            }
        },
        status=201,
    )


@csrf_exempt
@require_POST
def teacher_assign_student(request):
    auth_error = _require_role(request, "teacher")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    student_user_id = data.get("student_user_id")
    if not student_user_id:
        return JsonResponse({"error": "student_user_id is required"}, status=400)

    student_user = User.objects.filter(id=student_user_id, role=Role.STUDENT).first()
    if not student_user:
        return JsonResponse({"error": "Student user not found"}, status=404)

    link, created = StudentUnderTeacher.objects.get_or_create(
        teacher=request.user,
        student=student_user,
    )

    return JsonResponse(
        {
            "message": {
                "id": link.id,
                "teacher_user_id": link.teacher_id,
                "student_user_id": link.student_id,
                "created": created,
            }
        },
        status=201 if created else 200,
    )


@csrf_exempt
@require_GET
def teacher_students(request):
    auth_error = _require_role(request, "teacher")
    if auth_error:
        return auth_error

    teacher = Teacher.objects.filter(user=request.user).first()
    if not teacher:
        return JsonResponse({"error": "Teacher profile not found"}, status=404)

    q = str(request.GET.get("q", "")).strip().lower()
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

    return JsonResponse(
        {
            "message": {
                "subjects": subjects_payload,
                "subject_count": len(subjects_payload),
                "student_count": len(unique_student_ids),
            }
        },
        status=200,
    )


@csrf_exempt
@require_GET
def student_marks(request):
    auth_error = _require_role(request, "student")
    if auth_error:
        return auth_error

    student = Student.objects.filter(user=request.user).first()
    if not student:
        return JsonResponse({"error": "Student profile not found"}, status=404)

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

    return JsonResponse({"message": list(marks)}, status=200)
