from notification.websocket.routing import websocket_urlpatterns as notification_ws
from core.websocket.routing import websocket_urlpatterns as core_ws

websocket_urlpatterns = [
    *notification_ws,
    *core_ws,
]
