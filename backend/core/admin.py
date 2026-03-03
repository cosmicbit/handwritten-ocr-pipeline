from django.contrib import admin
from .models import (
    Department,
    Student,
    StudentMark,
    StudentUnderTeacher,
    Subject,
    Teacher,
    TeacherInstitution,
    Institution,
    TrueSubject,
)


@admin.register(TrueSubject)
class TrueSubjectAdmin(admin.ModelAdmin):
    list_display = ("id", "code", "name", "created_at")
    search_fields = ("code", "name")
    list_filter = ("created_at",)


@admin.register(StudentUnderTeacher)
class StudentUnderTeacherAdmin(admin.ModelAdmin):
    list_display = ("id", "teacher", "student", "created_at")
    search_fields = ("teacher__username", "student__username")
    list_filter = ("created_at",)


@admin.register(TeacherInstitution)
class TeacherInstitutionAdmin(admin.ModelAdmin):
    list_display = ("id", "teacher", "institution", "created_at")
    search_fields = ("teacher__username", "institution__username")
    list_filter = ("created_at",)


@admin.register(Department)
class DepartmentAdmin(admin.ModelAdmin):
    list_display = ("id", "name")
    search_fields = ("name",)


@admin.register(Teacher)
class TeacherAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "department", "created_at")
    search_fields = ("user__username", "department__name")
    list_filter = ("department", "created_at")


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ("id", "true_subject", "institution", "semester", "department", "teacher", "created_at")
    search_fields = (
        "true_subject__name",
        "true_subject__code",
        "institution__name",
        "department__name",
        "teacher__user__username",
    )
    list_filter = ("institution", "semester", "department", "created_at")


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "department", "created_at")
    search_fields = ("user__username", "department__name")
    list_filter = ("department", "created_at")


@admin.register(StudentMark)
class StudentMarkAdmin(admin.ModelAdmin):
    list_display = ("id", "student", "subject", "acquired_mark", "total_mark", "created_at")
    search_fields = ("student__user__username", "subject__true_subject__name", "subject__true_subject__code")
    list_filter = ("subject", "created_at")


admin.site.register(Institution)
