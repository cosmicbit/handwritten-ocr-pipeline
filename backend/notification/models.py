from django.db import models
from django.conf import settings
from django.contrib.auth.models import Group

# Create your models here.


class NotificationType(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(null=True, blank=True)

class Notification(models.Model):
    title = models.CharField(max_length=255,blank=True, null=True)
    type = models.ForeignKey(NotificationType, on_delete=models.CASCADE)
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
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

class NotificationRead(models.Model):
    notification = models.ForeignKey(Notification, on_delete=models.CASCADE)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    read_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('notification', 'user')
