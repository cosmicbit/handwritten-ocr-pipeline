import jwt
from django.conf import settings
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.urls import resolve

class JWTAuthenticationMiddleware:
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        resolver_match = resolve(request.path)

        # Skip JWT if the view is tagged as public
        view_func = resolver_match.func
        if hasattr(view_func, "is_public"):
            return self.get_response(request)
        print(request.headers)
        auth_header = request.headers.get('Authorization')

        request.user = None

        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(" ")[1]

            try:
                payload = jwt.decode(
                    token,
                    settings.JWT_SECRET,
                    algorithms=[settings.JWT_ALGORITHM]
                )
                user_id = payload.get("user_id")
                request.user = User.objects.get(id=user_id)
            except jwt.ExpiredSignatureError:
                return JsonResponse({
                    "error": "Token expired"
                }, status=400)
            
            except jwt.InvalidTokenError:
                return JsonResponse({
                    "error" : "Invalid token"
                }, status=400)
            except User.DoesNotExist:
                return JsonResponse({
                    "error": "User not found"
                },status=400)
        else:
            return JsonResponse({
                "error": "Missing token"
            }, status=401)
            
        return self.get_response(request)