from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET
from rbac.permissions import has_permission
from .service.notification_service import NotificationService
from auth2.services.user_service import UserService
import json


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

@csrf_exempt
@require_GET
@has_permission()
def get_notification_for_user(req):
    groups = userService.get_group(req.user)
    notifications = []
    for group in groups:
        notifications.append(notificationService.get_notifications_for_group(group_id=group.id))
    return JsonResponse({
        'message': notifications,
    })

@csrf_exempt
@require_GET
@has_permission()
def get_notification_for_group(req):
    groups = userService.get_group(req.user)
    notifications = []
    for group in groups:
        notifications.append(notificationService.get_notifications_for_group(group_id=group.id))
    return JsonResponse({
        'message': notifications,
    })

@csrf_exempt
@require_POST
@has_permission()
def update_notification_read(req, nid):
    updated = notificationService.update_notification_read(nid, req.user.id)
    return JsonResponse({
        'message':'Notification has been updated'
    })

@csrf_exempt
@require_POST
@has_permission()
def create_notification(req):
    data = json.loads(req.body)
    notificationService.create_notification(data)
    return JsonResponse({
        'message':'Notification has been created'
    })  

