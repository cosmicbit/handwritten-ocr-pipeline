import jwt
import time
import logging
from django.conf import settings
from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.urls import resolve

User = get_user_model()
logger = logging.getLogger("auth.jwt")

class JWTAuthenticationMiddleware:

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):

        if request.path.startswith("/admin/") or request.path.startswith("/static/") or request.path.startswith("/media/"):
            return self.get_response(request)

        resolver_match = resolve(request.path)
        view_func = resolver_match.func

        if hasattr(view_func, "is_public"):
            return self.get_response(request)

        auth_header = request.headers.get("Authorization")
        request.user = None

        if not auth_header or not auth_header.startswith("Bearer "):
            logger.warning(
                "Missing JWT | path=%s ip=%s",
                request.path,
                self._get_ip(request),
            )
            return JsonResponse(
                {"error": "Missing token"},
                status=401
            )

        token = auth_header.split(" ")[1]
        print("Token:", token)

        try:
            print("Decoding JWT...")
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM],
            )

            user_id = payload.get("user_id")
            exp = payload.get("exp")

            print("Payload:", payload)

            if exp:
                now = int(time.time())
                remaining_seconds = exp - now

                logger.debug(
                    "JWT validated | user_id=%s path=%s expires_in=%ss exp=%s",
                    user_id,
                    request.path,
                    remaining_seconds,
                    exp,
                )

            request.user = User.objects.get(id=user_id)

        except jwt.ExpiredSignatureError as e:
            print(e)
            logger.info(
                "JWT expired | path=%s ip=%s",
                request.path,
                self._get_ip(request),
            )
            return JsonResponse(
                {"error": "Token expired"},
                status=401
            )

        except jwt.InvalidTokenError:
            logger.warning(
                "Invalid JWT | path=%s ip=%s",
                request.path,
                self._get_ip(request),
            )
            return JsonResponse(
                {"error": "Invalid token"},
                status=401
            )

        except User.DoesNotExist:
            logger.warning(
                "JWT user not found | user_id=%s path=%s",
                user_id,
                request.path,
            )
            return JsonResponse(
                {"error": "User not found"},
                status=401
            )

        return self.get_response(request)

    def _get_ip(self, request):
        return request.META.get("REMOTE_ADDR")
