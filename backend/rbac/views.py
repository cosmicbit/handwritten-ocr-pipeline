from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .permissions import has_permission, check_permission
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .services.table_desc_service import TableDescriptionService
from django.views.decorators.http import require_POST


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
    data = json.loads(request.body)
    try:
        table_name = data['table_name']
    except json.JSONDecodeError as e:
        return JsonResponse({
            'error':'Invalid JSON format'
        },status=401
    )

    if not tableDescription.table_exists(table_name):
        return JsonResponse({
            'error': 'No matching Table found'
        },status=201
    )
    response = tableDescription.get_table_description(table_name=table_name)
    return JsonResponse(response, status=201)