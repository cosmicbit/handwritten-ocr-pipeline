from django.contrib import admin
from . import models
from .websocket.service import WebSocketService

ws_service = WebSocketService()


class NotifyOnAdminChangeMixin:
    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        action = "updated" if change else "created"
        ws_service.notify_model_change(
            instance=obj,
            action=action,
            actor_id=request.user.id,
        )

    def delete_model(self, request, obj):
        ws_service.notify_model_change(
            instance=obj,
            action="deleted",
            actor_id=request.user.id,
        )
        super().delete_model(request, obj)


@admin.register(models.Notification)
class NotificationAdmin(NotifyOnAdminChangeMixin, admin.ModelAdmin):
    list_display = ("id", "title", "type", "created_at")

    def save_model(self, request, obj, form, change):
        if not change and not obj.created_by_id:
            obj.created_by_id = request.user.id
        super().save_model(request, obj, form, change)


@admin.register(models.NotificationType)
class NotificationTypeAdmin(NotifyOnAdminChangeMixin, admin.ModelAdmin):
    list_display = ("id", "name")


@admin.register(models.NotificationRead)
class NotificationReadAdmin(NotifyOnAdminChangeMixin, admin.ModelAdmin):
    list_display = ("id", "notification", "user", "read_at")


@admin.register(models.GroupNotification)
class GroupNotificationAdmin(NotifyOnAdminChangeMixin, admin.ModelAdmin):
    list_display = ("id", "group", "notification", "status")


@admin.register(models.UserNotification)
class UserNotificationAdmin(NotifyOnAdminChangeMixin, admin.ModelAdmin):
    list_display = ("id", "user", "notification", "created_at")
