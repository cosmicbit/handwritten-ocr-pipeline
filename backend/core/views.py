import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from core.tasks import run_engine_task


from rbac.permissions import has_permission

from .services.core_service import CoreService, CoreServiceError

core_service = CoreService()
logger = logging.getLogger("engine.realtime")


def _parse_json(request):
    try:
        return json.loads(request.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _require_role(request, role_name):
    user = getattr(request, "user", None)
    if not user:
        return JsonResponse({"error": "Authentication required"}, status=401)
    if str(getattr(user, "role", "")).lower() != role_name:
        return JsonResponse({"error": f"{role_name.title()} role required"}, status=403)
    return None


def _service_response(service_method, *args, **kwargs):
    try:
        message, status = service_method(*args, **kwargs)
        return JsonResponse({"message": message}, status=status)
    except CoreServiceError as exc:
        return JsonResponse(exc.to_response_body(), status=exc.status)


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
    q = request.GET.get("q", "").strip()
    return _service_response(core_service.institution_search_users, role, q)


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

    return _service_response(
        core_service.institution_add_teacher,
        request.user,
        teacher_user_id,
        department_id,
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

    return _service_response(
        core_service.institution_add_student,
        request.user,
        student_user_id,
        department_id,
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

    return _service_response(
        core_service.institution_create_teacher,
        request.user,
        username,
        email,
        password,
        department_id,
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

    return _service_response(
        core_service.institution_create_student,
        request.user,
        username,
        email,
        password,
        department_id,
    )


@csrf_exempt
@require_GET
def institution_departments(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    return _service_response(core_service.institution_departments, request.user)


@csrf_exempt
@require_GET
def institution_members(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

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

    return _service_response(
        core_service.institution_members,
        request.user,
        role,
        q,
        page,
        page_size,
        offset,
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

    return _service_response(core_service.institution_add_department, request.user, name)


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

    return _service_response(
        core_service.institution_add_subject,
        request.user,
        true_subject_id,
        semester,
        department_id,
        teacher_id,
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

    return _service_response(core_service.institution_update_teacher, request.user, teacher_id, department_id)


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

    return _service_response(core_service.institution_remove_teacher, request.user, teacher_id)


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

    return _service_response(core_service.institution_update_student, request.user, student_id, department_id)


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

    return _service_response(core_service.institution_remove_student, request.user, student_id)


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

    return _service_response(core_service.institution_update_department, request.user, department_id, name)


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

    return _service_response(core_service.institution_remove_department, request.user, department_id)


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

    return _service_response(
        core_service.institution_update_subject_assignment,
        request.user,
        subject_id,
        teacher_id,
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

    return _service_response(core_service.institution_remove_subject, request.user, subject_id)


@csrf_exempt
@require_GET
def institution_true_subjects(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

    q = str(request.GET.get("q", "")).strip()
    return _service_response(core_service.institution_true_subjects, q)


@csrf_exempt
@require_GET
def institution_subjects(request):
    auth_error = _require_role(request, "institution")
    if auth_error:
        return auth_error

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

    return _service_response(
        core_service.institution_subjects,
        request.user,
        q,
        page,
        page_size,
        offset,
    )


@csrf_exempt
@require_POST
@has_permission()
def teacher_upload_pdf(request):
    auth_error = _require_role(request, "teacher")
    if auth_error:
        return auth_error

    subject_id = request.POST.get("subject_id")
    student_id = request.POST.get("student_id")
    print(subject_id, student_id)
    if not subject_id or not student_id:
        return JsonResponse({"error": "subject_id and student_id are required"}, status=400)

    uploaded_file = request.FILES.get("pdf")
    return _service_response(
        core_service.teacher_upload_pdf,
        request.user,
        subject_id,
        student_id,
        uploaded_file,
        request.build_absolute_uri,
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

    uploaded_file = request.FILES.get("pdf")
    return _service_response(
        core_service.teacher_upload_answer_key,
        request.user,
        subject_id,
        uploaded_file,
        request.build_absolute_uri,
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

    return _service_response(core_service.teacher_assign_student, request.user, student_user_id)


@csrf_exempt
@require_GET
def teacher_students(request):
    auth_error = _require_role(request, "teacher")
    if auth_error:
        return auth_error

    q = str(request.GET.get("q", "")).strip().lower()
    return _service_response(core_service.teacher_students, request.user, q)


@csrf_exempt
@require_GET
def student_marks(request):
    auth_error = _require_role(request, "student")
    if auth_error:
        return auth_error

    semester = request.GET.get("semester")
    subject_id = request.GET.get("subject_id")
    return _service_response(core_service.student_marks, request.user, semester, subject_id)


@csrf_exempt
@require_GET
def student_mark_options(request):
    auth_error = _require_role(request, "student")
    if auth_error:
        return auth_error

    semester = request.GET.get("semester")
    return _service_response(core_service.student_mark_options, request.user, semester)


@csrf_exempt
@require_POST
@has_permission()
def engine_trigger(request):
    auth_error = _require_role(request, "teacher")
    if auth_error:
        return auth_error

    data = _parse_json(request)
    if data is None:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    subject_id = data.get("subject_id")
    student_id = data.get("student_id")
    marks = data.get("marks")

    if not subject_id or not student_id:
        return JsonResponse({"error": "subject_id and student_id are required"}, status=400)

    try:
        context, _status = core_service.trigger_engine_model(request.user, subject_id, student_id)
    except CoreServiceError as exc:
        return JsonResponse(exc.to_response_body(), status=exc.status)

    task = run_engine_task.delay(
        teacher_pdf_path=context["teacher_pdf_path"],
        student_pdf_path=context["student_pdf_path"],
        marks=marks,
        subject_id=context["subject_id"],
        student_id=context["student_id"],
        teacher_answer_key_upload_id=context["teacher_answer_key_upload_id"],
        student_pdf_upload_id=context["student_pdf_upload_id"],
    )
    logger.info(
        "trigger_accepted task_id=%s user_id=%s subject_id=%s student_id=%s ws_path=%s",
        task.id,
        getattr(request.user, "id", None),
        context["subject_id"],
        context["student_id"],
        f"/ws/engine/status/{task.id}/",
    )
    return JsonResponse(
        {
            "message": "Engine started",
            "task_id": task.id,
            "ws_path": f"/ws/engine/status/{task.id}/",
            "subject_id": context["subject_id"],
            "student_id": context["student_id"],
            "teacher_answer_key_upload_id": context["teacher_answer_key_upload_id"],
            "student_pdf_upload_id": context["student_pdf_upload_id"],
        },
        status=202,
    )
   

from celery.result import AsyncResult

@require_GET
def engine_status(request, task_id):
    result = AsyncResult(task_id)
    info = result.info if isinstance(result.info, dict) else {}
    return JsonResponse({
        "task_id": task_id,
        "state": result.state,
        "stage": info.get("stage"),
        "progress": info.get("progress"),
        "message": info.get("message"),
        "result": result.result if result.successful() else None,
        "error": str(result.result) if result.failed() else None,
    })
