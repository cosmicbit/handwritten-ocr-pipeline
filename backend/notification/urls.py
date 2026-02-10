from django.urls import path
from . import views

urlpatterns = [
    path('user/', view=views.get_notification_for_user, name="get_all_notification_for_user"),
    path('group/', view=views.get_notification_for_group, name="get_notification_for_group"),
    path('read/<int:nid>/', view=views.update_notification_read, name="update_notification_read")
]