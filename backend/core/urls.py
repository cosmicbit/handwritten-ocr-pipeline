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
    path("institution/departments/add", views.institution_add_department, name="institution_add_department"),
    path("institution/true-subjects", views.institution_true_subjects, name="institution_true_subjects"),
    path("institution/subjects", views.institution_subjects, name="institution_subjects"),
    path("institution/subjects/set", views.institution_add_subject, name="institution_set_subject"),
    path("institution/subjects/add", views.institution_add_subject, name="institution_add_subject"),
    path("teacher/students/assign", views.teacher_assign_student, name="teacher_assign_student"),
    path("student/marks", views.student_marks, name="student_marks"),
]
