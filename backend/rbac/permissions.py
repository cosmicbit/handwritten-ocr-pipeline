from django.http import JsonResponse
from functools import wraps
from django.contrib.auth import get_user_model
from .services.table_desc_service import TableDescriptionService
import json

tableDescriptionService=TableDescriptionService()
User = get_user_model()

def has_permission():
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
            permissions = [permission for group in user.groups.all() for permission in group.permissions.all()]
            print(permissions)
            if user.is_superuser:
                return view_func(request, *args, **kwargs)
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

        table_name_with_application = None


        if request.method == "GET":
            table_name_with_application = kwargs.get("table_name")


        else:
            try:
                data = json.loads(request.body.decode("utf-8"))
                table_name_with_application = data.get("table_name")
            except json.JSONDecodeError:
                return JsonResponse(
                    {"error": "Invalid JSON"},
                    status=400
                )

        if not table_name_with_application:
            return JsonResponse(
                {"error": "table_name is required"},
                status=400
            )

        # ✅ split app + model
        try:
            application_name, table_name = table_name_with_application.split("_", 1)
        except ValueError:
            return JsonResponse(
                {"error": "Invalid table_name format"},
                status=400
            )

        required_permissions = [
            f"{application_name}.view_{table_name}",
            f"{application_name}.add_{table_name}",
            f"{application_name}.change_{table_name}",
            f"{application_name}.delete_{table_name}",
        ]

        for permission in required_permissions:
            if not user.has_perm(permission):
                return JsonResponse(
                    {"error": "Permission denied"},
                    status=403
                )

        return view_func(request, *args, **kwargs)

    return wrapper


def is_super_admin(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        user = getattr(request, "user", None)

        if not user:
            return JsonResponse({"error": "Authentication required"}, status=401)

        if not ( user.groups.filter(name="admin").exists() or user.is_superuser ):
            return JsonResponse({"error": "Super admin required"}, status=403)

        return view_func(request, *args, **kwargs)
    return wrapper
