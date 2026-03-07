from channels.generic.websocket import AsyncJsonWebsocketConsumer
import logging

logger = logging.getLogger("engine.realtime")

class EngineTaskConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        user = self.scope.get("user")
        if not user or user.is_anonymous:
            await self.close(code=4001)
            return

        self.task_id = self.scope["url_route"]["kwargs"]["task_id"]
        self.group_name = f"engine_task_{self.task_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        logger.info(
            "ws_subscribed user_id=%s task_id=%s group=%s",
            getattr(user, "id", None),
            self.task_id,
            self.group_name,
        )
        await self.send_json(
            {
                "event": "connected",
                "task_id": self.task_id,
                "message": "Subscribed to engine task updates",
            }
        )

    async def disconnect(self, close_code):
        if hasattr(self, "group_name"):
            await self.channel_layer.group_discard(self.group_name, self.channel_name)
            logger.info("ws_unsubscribed task_id=%s group=%s code=%s", self.task_id, self.group_name, close_code)

    async def receive_json(self, content, **kwargs):
        # Keep the socket alive for frontend ping messages.
        if content.get("type") == "ping":
            await self.send_json({"type": "pong", "task_id": self.task_id})

    async def engine_status(self, event):
        payload = event["payload"]
        logger.info(
            "ws_send task_id=%s event=%s stage=%s progress=%s",
            self.task_id,
            payload.get("event"),
            payload.get("stage"),
            payload.get("progress"),
        )
        await self.send_json(payload)
