from django.contrib import admin
from django.contrib.auth import get_user_model
from django.db import models
from django.db.models import F, Q
User = get_user_model()

from django.contrib.auth.models import Permission

# Register your models here.
@admin.register(User)
class UserModel(admin.ModelAdmin):
    list_display = ("id", "username", "email")

admin.site.register(Permission)