from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from rbac.permissions import has_permission
from .service.notification_service import NotificationService
from auth2.services.user_service import UserService
import json
from .types.types import NotificationReadStatus


APPLICATION_NAME="notification"

PERMISSION_FOR_VIEWING_NOTIFICATION = [ 
    f"{APPLICATION_NAME}.view_notificationread",
    f"{APPLICATION_NAME}.view_notificationtype",
    f"{APPLICATION_NAME}.view_notification",
    f"{APPLICATION_NAME}.view_groupnotification"
]

PERMISSION_FOR_CHANGING_NOTIFICATION = [ 
    f"{APPLICATION_NAME}.change_notificationread",
]

notificationService = NotificationService()
userService = UserService()


def _dedupe_notifications(notifications):
    deduped = {}
    for item in notifications:
        deduped[item["notification_id"]] = item
    return sorted(
        deduped.values(),
        key=lambda x: x.get("created_at"),
        reverse=True,
    )

@csrf_exempt
@require_GET
@has_permission()
def get_notification_for_user(req):
    groups = userService.get_group(req.user)
    notifications = []
    for group in groups:
        notifications.extend(
            notificationService.get_notifications_for_group(
                group_id=group.id,
                user_id=req.user.id,
            )
        )
    return JsonResponse({
        'message': _dedupe_notifications(notifications),
    })

@csrf_exempt
@require_GET
@has_permission()
def get_notification_for_group(req):
    groups = userService.get_group(req.user)
    notifications = []
    for group in groups:
        notifications.extend(
            notificationService.get_notifications_for_group(
                group_id=group.id,
                user_id=req.user.id,
            )
        )
    return JsonResponse({
        'message': _dedupe_notifications(notifications),
    })

@csrf_exempt
@require_POST
@has_permission()
def update_notification_read(req, nid):
    updated, notification_read = notificationService.update_notification_read(nid, req.user.id)
    if updated == NotificationReadStatus.FAILED:
        return JsonResponse(
            {"message": updated.value},
            status=400,
        )

    return JsonResponse({
        "message": updated.value,
        "notification_id": nid,
        "user_id": req.user.id,
        "is_seen": True,
        "seen_at": notification_read.read_at if notification_read else None,
    })

@csrf_exempt
@require_POST
@has_permission()
def create_notification(req):
    data = json.loads(req.body)
    notificationService.create_notification(data, created_by_id=req.user.id)
    return JsonResponse({
        'message':'Notification has been created'
    })  
