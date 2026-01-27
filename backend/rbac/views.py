from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .permissions import has_permission, check_permission, is_super_admin
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .services.table_desc_service import TableDescriptionService
from django.views.decorators.http import require_POST
from .commons import INVALID_JSON_ERROR, validate_json


import json
tableDescription=TableDescriptionService()

APPLICATION_NAME="auth2"

@csrf_exempt
@require_POST
def admin(req):
    JsonResponse({
        'message': 'hello world'
    },status=201)



@has_permission([
    f"{APPLICATION_NAME}.add_user",
    f"{APPLICATION_NAME}.view_user"
    ]
)
@require_POST
@csrf_exempt
def test(req):
    return JsonResponse({
        'message':'hello from server'
    },status=201)


@csrf_exempt
@require_POST
@check_permission
def get_table_description(request):
    data = validate_json(request=request)
    table_name = data['table_name']
    if not tableDescription.table_exists(table_name):
        return JsonResponse({'error':'No matching table found'},status=201
    )
    response = tableDescription.get_table_description(table_name=table_name)
    return JsonResponse(response, status=201)


@is_super_admin
@csrf_exempt
@require_POST
def get_tables(req):
    data = validate_json(request=req)
    response = tableDescription.get_all_tables()
    return JsonResponse({ 'message': response }, status=201)