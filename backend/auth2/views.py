from django.http import HttpResponse, JsonResponse
from django.contrib.auth import authenticate
from utils.jwt_utils import create_jwt_token 
from django.views.decorators.csrf import csrf_exempt
from django.db import connection
import logging
import json

from .commons.require_methods import require_post
from utils.decorator import public_view
from utils.permissions import has_permission, check_permission
from .services.user_service import UserService
from .exceptions.user_already_exist import UserAlreadyExist
from .exceptions.no_default_group import NoDefaultGroup

logger = logging.getLogger(__name__)
APPLICATION_NAME="auth2"



@has_permission([
    f"{APPLICATION_NAME}.add_user",
    f"{APPLICATION_NAME}.view_user"
    ])
@csrf_exempt
def test(req):
    return HttpResponse("hello from the server")

@csrf_exempt
@check_permission
def get_table_description(request):
    data = json.loads(request.body)
    table_name = f"{data['table_name']}"

    with connection.cursor() as cursor:
        columns = connection.introspection.get_table_description(
            cursor,
            table_name
        )

    return JsonResponse({
        "table": table_name,
        "columns": [
            {
                "name": col.name,
                "null": col.null_ok,
                "type": col.type_code,
            }
            for col in columns
        ]
    })


@csrf_exempt
@public_view
def getTableDesc(req):
    resp = require_post(req=req)
    if resp:
        return resp
    

@csrf_exempt
@public_view
def register(req, userService=UserService()):
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

    return JsonResponse({
        'message': 'Login successful',
        'token': token
    }, status=200
    )

