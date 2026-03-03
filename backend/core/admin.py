from django.contrib import admin
from .models import StudentUnderTeacher, TeacherInstitution


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
