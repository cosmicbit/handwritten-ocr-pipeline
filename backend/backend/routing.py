from notification.websocket.routing import websocket_urlpatterns as notification_ws

websocket_urlpatterns = [
    *notification_ws,
]
