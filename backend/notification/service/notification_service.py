from ..models import *
from collections import defaultdict
from ..types.types import NotificationReadStatus
from django.contrib.auth.models import User

class NotificationService():

    def __init__(self):
        pass

    def check_notification_already_read(self, notification_id, user_id) -> bool:
        return NotificationRead.objects.filter(notification_id = notification_id, user_id = user_id).exists()

    def get_notifications_for_group(self, group_id):
        group_notifications = (
            GroupNotification.objects
            .filter(group_id=group_id, status=GroupNotification.Status.ACTIVE)
            .select_related("notification")
        )

        notification_ids = [gn.notification_id for gn in group_notifications]

        reads = (
            NotificationRead.objects
            .filter(notification_id__in=notification_ids)
            .select_related("user")
        )

        read_map = defaultdict(list)

        for r in reads:
            read_map[r.notification_id].append(r.user)
        
        result = []

        for gn in group_notifications:
            result.append({
                "notification_id": gn.notification_id,
                "message": gn.notification.message,
                "created_at": gn.notification.created_at,
                "read_by": read_map.get(gn.notification_id, [])
            })

        return result

    def update_notification_read(self, notification_id, user_id) -> NotificationReadStatus:
        try:
            if self.check_notification_already_read(notification_id=notification_id, user_id=user_id):
                return NotificationReadStatus.ALREADY_EXISTS
            
            notification_read = NotificationRead.objects.create(
                notification_id = notification_id,
                user_id = user_id
            )

            if notification_read:
                return NotificationReadStatus.CREATED
            else: 
                return NotificationReadStatus.FAILED
                
        except Exception as e:
            return NotificationReadStatus.FAILED
        
    def create_notification(self, data):
        notification = Notification.objects.create(
            message = data.get("message", ""),
        )

        group_ids = data.get("group_ids", [])
        for group_id in group_ids:
            GroupNotification.objects.create(
                group_id = group_id,
                notification_id = notification.id
            )