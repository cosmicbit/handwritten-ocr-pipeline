from django.http import JsonResponse
from functools import wraps
from django.contrib.auth import get_user_model
import json

User = get_user_model()

def has_permission(permissions: list):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            user = request.user

            if not user or not user.is_authenticated:
                return JsonResponse(
                    {"error": "Authentication required"},
                    status=401
                )
            print(list(user.groups.all()))

            print("Permission needed : ", permissions)

            for permission in permissions:
                if not user.has_perm(permission):
                    return JsonResponse(
                        {"error": "Permission denied"},
                        status=403
                    )

            return view_func(request, *args, **kwargs)

        return wrapper
    return decorator


def check_permission(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        user = request.user

        if not user or not user.is_authenticated:
            return JsonResponse(
                {"error": "Authentication required"},
                status=401
            )

        try:
            data = json.loads(request.body)
            table_name_with_application = data["table_name"]
            application_name, table_name = table_name_with_application.split("_",1)

            required_permissions = [
                f"{application_name}.add_{table_name}",
                f"{application_name}.view_{table_name}",
                f"{application_name}.change_{table_name}",
                f"{application_name}.delete_{table_name}",
            ]

            for permission in required_permissions:
                if not user.has_perm(permission):
                    return JsonResponse(
                        {"error": "Permission denied"},
                        status=403
                    )

        except KeyError:
            return JsonResponse(
                {"error": "table_name is required"},
                status=400
            )
        except json.JSONDecodeError:
            return JsonResponse(
                {"error": "Invalid JSON"},
                status=400
            )

        return view_func(request, *args, **kwargs)

    return wrapper