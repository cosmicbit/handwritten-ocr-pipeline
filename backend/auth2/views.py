from django.http import HttpResponse, JsonResponse
from django.contrib.auth import authenticate
from .jwt_utils import create_jwt_token 
from django.views.decorators.csrf import csrf_exempt
from django.db import connection
import logging
import json

from .commons.require_methods import require_post
from .decorator import public_view
from rbac.permissions import has_permission, check_permission
from .services.user_service import UserService
from .exceptions.user_already_exist import UserAlreadyExist
from .exceptions.no_default_group import NoDefaultGroup

logger = logging.getLogger(__name__)
userService=UserService()
    

@csrf_exempt
@public_view
def register(req):
    resp = require_post(req=req)
    if resp:
        return resp
    
    try:
        data = json.loads(req.body.decode("utf-8"))
    except json.JSONDecodeError as e:
        logger.exception(e)
        return JsonResponse({ 'error':'Invalid JSON format' }, status=400)
    try:
        response = userService.register(data=data)
        logger.info("New user registered ")
        return JsonResponse(response,status=201)
    except NoDefaultGroup as e:
        return JsonResponse({ 'error': str(e) }, status=400)
    except UserAlreadyExist as e:
        logger.exception(e)
        return JsonResponse({ 'error': str(e) },status=409)
    except Exception as e:
        logger.exception(e)
        return JsonResponse({ 'error': str(e) },status=400)
    
    
@csrf_exempt
@public_view
def login(req):
    resp = require_post(req)
    if resp:
        return resp

    try:
        data = json.loads(req.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    user = authenticate(
        req,
        username=data.get("username"),
        password=data.get("password")
    )

    print(req.user)

    if user is None:
        return JsonResponse({'error': 'Invalid credentials'}, status=401)

    token = create_jwt_token(user)
    print("Generated Token:", token)

    return JsonResponse({
        'message': {
            'token': token,
            'user': {
                'username': user.username,
                'is_superAdmin': user.is_superuser,
                'role': [group.name for group in userService.get_group(user=user)]
            }
        }
        
    }, status=200
    )

