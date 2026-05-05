from channels.generic.websocket import AsyncJsonWebsocketConsumer
from asgiref.sync import sync_to_async
from celery.result import AsyncResult
import logging
from core.tasks import _get_stage_metadata, _normalize_stage_for_payload

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
                "task_id": self.task_id,
                "event": "connected",
                "stage": "connected",
                "progress": 0,
                "message": "Subscribed to engine task updates",
                **_get_stage_metadata("connected"),
            }
        )
        snapshot = await self._get_task_snapshot()
        if snapshot:
            logger.info(
                "ws_snapshot_send task_id=%s state=%s stage=%s progress=%s",
                self.task_id,
                snapshot.get("state"),
                snapshot.get("stage"),
                snapshot.get("progress"),
            )
            await self.send_json(snapshot)

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

    @sync_to_async
    def _get_task_snapshot(self):
        result = AsyncResult(self.task_id)
        info = result.info if isinstance(result.info, dict) else {}

        if result.state == "PENDING" and not info:
            return None

        payload = {
            "event": "snapshot",
            "task_id": self.task_id,
            "state": result.state,
            "stage": _normalize_stage_for_payload(info.get("stage") or "queued"),
            "progress": info.get("progress", 0),
            "message": info.get("message") or "Waiting for task progress",
        }
        payload.update(_get_stage_metadata(payload["stage"]))

        if result.successful() and isinstance(result.result, dict):
            payload.update(result.result)
            payload["event"] = "success"
        elif result.failed():
            payload["event"] = "failure"
            payload["message"] = payload["message"] or str(result.result)
            payload.update(_get_stage_metadata(payload["stage"]))

        return payload
