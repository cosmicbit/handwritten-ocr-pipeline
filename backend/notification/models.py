from django.db import models
from django.conf import settings
from django.contrib.auth.models import Group

# Create your models here.


class NotificationType(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name

class Notification(models.Model):
    title = models.CharField(max_length=255,blank=True, null=True)
    type = models.ForeignKey(NotificationType, on_delete=models.CASCADE)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_notifications",
    )
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title or f"Notification {self.id}"
    
class GroupNotification(models.Model):
    class Status(models.TextChoices):
        ACTIVE = 'ACTIVE', 'Active'
        DISABLED = 'DISABLED', 'Disabled'
        EXPIRED = 'EXPIRED', 'Expired'

    group = models.ForeignKey(Group, on_delete=models.CASCADE)
    notification = models.ForeignKey(Notification, on_delete=models.CASCADE)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE
    )

    class Meta:
        unique_together = ('group', 'notification')

    def __str__(self):
        group_name = self.group.name if self.group_id and self.group else f"group_id={self.group_id}"
        return f"{group_name} -> {self.notification}"

class NotificationRead(models.Model):
    notification = models.ForeignKey(Notification, on_delete=models.CASCADE)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    read_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('notification', 'user')

    def __str__(self):
        username = self.user.username if self.user_id and self.user else f"user_id={self.user_id}"
        return f"{username} read {self.notification}"


class UserNotification(models.Model):
    notification = models.ForeignKey(Notification, on_delete=models.CASCADE)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("notification", "user")

    def __str__(self):
        username = self.user.username if self.user_id and self.user else f"user_id={self.user_id}"
        return f"{username} -> {self.notification}"
