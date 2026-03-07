from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.contrib.auth import get_user_model

from ..models import GroupNotification, Notification, NotificationRead, NotificationType, UserNotification


class WebSocketService:
    def __init__(self):
        self.channel_layer = get_channel_layer()
        self.user_model = get_user_model()

    def notify_users(self, user_ids, payload):
        if not self.channel_layer:
            return

        for user_id in set(user_ids):
            async_to_sync(self.channel_layer.group_send)(
                f"user_{user_id}",
                {
                    "type": "notify",
                    "payload": payload,
                },
            )

    def notify_model_change(self, instance, action, actor_id=None):
        payload = {
            "event": "db_change",
            "action": action,
            "model": instance.__class__.__name__,
            "object_id": instance.pk,
            "actor_id": actor_id,
        }
        target_user_ids = self._get_target_user_ids(instance)
        self.notify_users(target_user_ids, payload)

    def _get_target_user_ids(self, instance):
        if isinstance(instance, GroupNotification):
            return list(
                self.user_model.objects.filter(groups=instance.group)
                .distinct()
                .values_list("id", flat=True)
            )

        if isinstance(instance, Notification):
            group_ids = GroupNotification.objects.filter(
                notification=instance,
                status=GroupNotification.Status.ACTIVE,
            ).values_list("group_id", flat=True)
            return list(
                self.user_model.objects.filter(groups__id__in=group_ids)
                .distinct()
                .values_list("id", flat=True)
            )

        if isinstance(instance, NotificationRead):
            return [instance.user_id]

        if isinstance(instance, UserNotification):
            return [instance.user_id]

        if isinstance(instance, NotificationType):
            return self._get_admin_user_ids()

        return self._get_admin_user_ids()

    def _get_admin_user_ids(self):
        return list(
            self.user_model.objects.filter(is_staff=True)
            .distinct()
            .values_list("id", flat=True)
        )
