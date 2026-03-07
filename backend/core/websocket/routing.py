from django.urls import re_path

from .consumer import EngineTaskConsumer

websocket_urlpatterns = [
    re_path(r"ws/engine/status/(?P<task_id>[^/]+)/$", EngineTaskConsumer.as_asgi()),
]
