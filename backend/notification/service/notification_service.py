from ..models import *
from ..types.types import NotificationReadStatus
from django.db import IntegrityError
from ..websocket.service import WebSocketService
import logging

logger = logging.getLogger("notification.realtime")

class NotificationService():

    def __init__(self):
        self.ws_service = WebSocketService()

    def check_notification_already_read(self, notification_id, user_id) -> bool:
        return NotificationRead.objects.filter(notification_id = notification_id, user_id = user_id).exists()

    def get_notifications_for_group(self, group_id, user_id=None):
        group_notifications = (
            GroupNotification.objects
            .filter(group_id=group_id, status=GroupNotification.Status.ACTIVE)
            .select_related("notification")
        )

        notification_ids = [gn.notification_id for gn in group_notifications]
        seen_at_map = {}

        if user_id:
            reads = NotificationRead.objects.filter(
                notification_id__in=notification_ids,
                user_id=user_id,
            ).values_list("notification_id", "read_at")
            seen_at_map = {notification_id: read_at for notification_id, read_at in reads}

        creator_read_map = {}
        if user_id:
            creator_notification_ids = [
                gn.notification_id
                for gn in group_notifications
                if gn.notification.created_by_id == user_id
            ]
            if creator_notification_ids:
                creator_reads = NotificationRead.objects.filter(
                    notification_id__in=creator_notification_ids
                ).values_list("notification_id", "user_id")
                for notification_id, reader_user_id in creator_reads:
                    creator_read_map.setdefault(notification_id, []).append(reader_user_id)
        
        result = []

        for gn in group_notifications:
            can_view_seen_by = bool(user_id and gn.notification.created_by_id == user_id)
            result.append({
                "notification_id": gn.notification_id,
                "title": gn.notification.title,
                "message": gn.notification.message,
                "created_at": gn.notification.created_at,
                "is_seen": gn.notification_id in seen_at_map,
                "seen_at": seen_at_map.get(gn.notification_id),
                "can_view_seen_by": can_view_seen_by,
                "read_by_user_ids": creator_read_map.get(gn.notification_id, []) if can_view_seen_by else [],
            })

        return result

    def get_notifications_created_by_user(self, user_id):
        notifications = Notification.objects.filter(created_by_id=user_id).select_related("type")
        notification_ids = list(notifications.values_list("id", flat=True))

        seen_reads = NotificationRead.objects.filter(
            notification_id__in=notification_ids,
            user_id=user_id,
        ).values_list("notification_id", "read_at")
        seen_at_map = {notification_id: read_at for notification_id, read_at in seen_reads}

        creator_reads = NotificationRead.objects.filter(
            notification_id__in=notification_ids
        ).values_list("notification_id", "user_id")
        creator_read_map = {}
        for notification_id, reader_user_id in creator_reads:
            creator_read_map.setdefault(notification_id, []).append(reader_user_id)

        result = []
        for notification in notifications:
            result.append({
                "notification_id": notification.id,
                "title": notification.title,
                "message": notification.message,
                "created_at": notification.created_at,
                "is_seen": notification.id in seen_at_map,
                "seen_at": seen_at_map.get(notification.id),
                "can_view_seen_by": True,
                "read_by_user_ids": creator_read_map.get(notification.id, []),
            })

        return result

    def get_notifications_for_user(self, user_id):
        user_notifications = (
            UserNotification.objects
            .filter(user_id=user_id)
            .select_related("notification", "notification__type")
            .order_by("-notification__created_at")
        )

        notification_ids = [un.notification_id for un in user_notifications]
        seen_reads = NotificationRead.objects.filter(
            notification_id__in=notification_ids,
            user_id=user_id,
        ).values_list("notification_id", "read_at")
        seen_at_map = {notification_id: read_at for notification_id, read_at in seen_reads}

        result = []
        for un in user_notifications:
            notification = un.notification
            result.append({
                "notification_id": notification.id,
                "title": notification.title,
                "message": notification.message,
                "created_at": notification.created_at,
                "is_seen": notification.id in seen_at_map,
                "seen_at": seen_at_map.get(notification.id),
                "can_view_seen_by": bool(notification.created_by_id == user_id),
                "read_by_user_ids": [user_id] if notification.id in seen_at_map and notification.created_by_id == user_id else [],
            })
        return result

    def _user_can_access_notification(self, notification_id, user_id) -> bool:
        has_direct_access = UserNotification.objects.filter(
            notification_id=notification_id,
            user_id=user_id,
        ).exists()
        if has_direct_access:
            return True

        has_group_access = GroupNotification.objects.filter(
            notification_id=notification_id,
            status=GroupNotification.Status.ACTIVE,
            group__user__id=user_id,
        ).exists()
        if has_group_access:
            return True

        return Notification.objects.filter(id=notification_id, created_by_id=user_id).exists()

    def update_notification_read(self, notification_id, user_id):
        try:
            if not self._user_can_access_notification(notification_id=notification_id, user_id=user_id):
                return NotificationReadStatus.FAILED, None

            notification_read, created = NotificationRead.objects.get_or_create(
                notification_id=notification_id,
                user_id=user_id,
            )

            if created:
                self.ws_service.notify_model_change(
                    instance=notification_read,
                    action="created",
                    actor_id=user_id,
                )
                return NotificationReadStatus.CREATED, notification_read
            return NotificationReadStatus.ALREADY_EXISTS, notification_read
        except IntegrityError:
            return NotificationReadStatus.FAILED, None
        except Exception:
            return NotificationReadStatus.FAILED, None
        
    def _resolve_notification_type(self, type_name=None):
        if type_name:
            notification_type, _ = NotificationType.objects.get_or_create(name=type_name)
            return notification_type

        notification_type = NotificationType.objects.first()
        if notification_type:
            return notification_type

        notification_type, _ = NotificationType.objects.get_or_create(
            name="system",
            defaults={"description": "Auto-generated system notification type"},
        )
        return notification_type

    def create_notification(self, data, created_by_id=None):
        notification_type = self._resolve_notification_type(data.get("type_name"))
        notification = Notification.objects.create(
            title=data.get("title"),
            message = data.get("message", ""),
            type=notification_type,
            created_by_id=created_by_id,
        )
        logger.info(
            "notification_created notification_id=%s type=%s created_by_id=%s title=%s",
            notification.id,
            notification_type.name if notification_type else None,
            created_by_id,
            notification.title,
        )

        group_ids = data.get("group_ids", [])
        for group_id in group_ids:
            GroupNotification.objects.create(
                group_id = group_id,
                notification_id = notification.id
            )
            logger.info(
                "group_notification_created notification_id=%s group_id=%s",
                notification.id,
                group_id,
            )

        user_ids = data.get("user_ids", [])
        logger.info(
            "notification_target_users notification_id=%s user_ids=%s",
            notification.id,
            sorted(set(user_ids)),
        )
        for user_id in set(user_ids):
            user_notification = UserNotification.objects.create(
                user_id=user_id,
                notification_id=notification.id,
            )
            logger.info(
                "user_notification_created notification_id=%s user_notification_id=%s user_id=%s",
                notification.id,
                user_notification.id,
                user_id,
            )
            self.ws_service.notify_model_change(
                instance=user_notification,
                action="created",
                actor_id=created_by_id,
            )

        if group_ids:
            self.ws_service.notify_model_change(
                instance=notification,
                action="created",
                actor_id=created_by_id,
            )

        return notification
