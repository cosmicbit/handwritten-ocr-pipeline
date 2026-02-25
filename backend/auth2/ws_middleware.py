from urllib.parse import parse_qs

import jwt
from channels.db import database_sync_to_async
from channels.middleware import BaseMiddleware
from django.conf import settings
from django.contrib.auth import get_user_model


@database_sync_to_async
def get_user_from_token(token: str):
    from django.contrib.auth.models import AnonymousUser

    User = get_user_model()
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return User.objects.get(id=payload.get("user_id"))
    except Exception:
        return AnonymousUser()


class JWTAuthMiddleware(BaseMiddleware):
    async def __call__(self, scope, receive, send):
        from django.contrib.auth.models import AnonymousUser

        token = None

        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization")
        if auth_header:
            value = auth_header.decode("utf-8")
            if value.startswith("Bearer "):
                token = value.split(" ", 1)[1]

        if not token:
            
            query = parse_qs(scope.get("query_string", b"").decode("utf-8"))
            token = (query.get("token") or [None])[0]

        scope["user"] = await get_user_from_token(token) if token else AnonymousUser()
        return await super().__call__(scope, receive, send)
