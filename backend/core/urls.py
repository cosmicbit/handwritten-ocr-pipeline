from django.urls import path

from . import views

urlpatterns = [
    path("test", view=views.test, name="core_test"),
    path("institution/users/search", views.institution_search_users, name="institution_search_users"),
    path("institution/departments", views.institution_departments, name="institution_departments"),
    path("institution/members", views.institution_members, name="institution_members"),
    path("institution/teachers/add", views.institution_add_teacher, name="institution_add_teacher"),
    path("institution/students/add", views.institution_add_student, name="institution_add_student"),
    path("institution/teachers/create", views.institution_create_teacher, name="institution_create_teacher"),
    path("institution/students/create", views.institution_create_student, name="institution_create_student"),
    path("institution/teachers/update", views.institution_update_teacher, name="institution_update_teacher"),
    path("institution/students/update", views.institution_update_student, name="institution_update_student"),
    path("institution/teachers/remove", views.institution_remove_teacher, name="institution_remove_teacher"),
    path("institution/students/remove", views.institution_remove_student, name="institution_remove_student"),
    path("institution/departments/add", views.institution_add_department, name="institution_add_department"),
    path("institution/departments/update", views.institution_update_department, name="institution_update_department"),
    path("institution/departments/remove", views.institution_remove_department, name="institution_remove_department"),
    path("institution/true-subjects", views.institution_true_subjects, name="institution_true_subjects"),
    path("institution/subjects", views.institution_subjects, name="institution_subjects"),
    path("institution/subjects/set", views.institution_add_subject, name="institution_set_subject"),
    path("institution/subjects/add", views.institution_add_subject, name="institution_add_subject"),
    path(
        "institution/subjects/update-assignment",
        views.institution_update_subject_assignment,
        name="institution_update_subject_assignment",
    ),
    path("institution/subjects/remove", views.institution_remove_subject, name="institution_remove_subject"),
    path("teacher/pdfs/upload", views.teacher_upload_pdf, name="teacher_upload_pdf"),
    path("teacher/subjects/answer-key/upload", views.teacher_upload_answer_key, name="teacher_upload_answer_key"),
    path("teacher/students/assign", views.teacher_assign_student, name="teacher_assign_student"),
    path("teacher/students", views.teacher_students, name="teacher_students"),
    path("student/marks", views.student_marks, name="student_marks"),
    path("student/marks/options", views.student_mark_options, name="student_mark_options"),
    path("engine/trigger", views.engine_trigger, name="engine_trigger"),
    path("engine/status/<str:task_id>", views.engine_status, name="engine_status"),
]
