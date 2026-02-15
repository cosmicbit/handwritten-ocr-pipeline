from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .permissions import has_permission, check_permission, is_super_admin
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .services.table_desc_service import TableDescriptionService
from django.views.decorators.http import require_POST, require_GET
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



@has_permission()
@require_POST
@csrf_exempt
def test(req):
    return JsonResponse({
        'message':'hello from server'
    },status=201)


@csrf_exempt
@require_GET
@check_permission
def get_table_description(request,table_name):
    data = validate_json(request=request)
    if not tableDescription.table_exists(table_name):
        return JsonResponse({'error':'No matching table found'},status=201
    )
    response = tableDescription.get_table_description(table_name=table_name)
    return JsonResponse({'message':response}, status=201)


@is_super_admin
@csrf_exempt
@require_GET
def get_tables(req):
    
    response = tableDescription.get_all_tables()
    return JsonResponse({ 'message': response }, status=201)


require_GET
@csrf_exempt
@is_super_admin
def get_table_data(req, table_name):
    page = req.GET.get("page", 1)
    page_size = req.GET.get("pageSize", 20)
    print( page, page_size)
    response = tableDescription.get_table_data(table_name, int(page_size), int(page))
    
    return JsonResponse({'message':{ 'data': response, 'total': len(response), 'page': page, 'page_size': page_size }}, status=201)


@csrf_exempt
@is_super_admin
def update_table_data(req, table_name, id):
    if req.method != "PATCH":
        return JsonResponse({ 'error': 'Invalid request method' }, status=405)
    print(table_name, id,req.body)
    data = validate_json(request=req)
    try:
        tableDescription.update_table_data(table_name, id, data)
    except Exception as e:
        return JsonResponse({ 'error': str(e) }, status=400)
    return JsonResponse({'message':f"Successfully updated {table_name}"}, status=201)

@require_GET
@csrf_exempt
@is_super_admin
def get_groups(req):
    groups = tableDescription.get_groups()
    return JsonResponse({'message':groups}, status=201)


@require_GET
@csrf_exempt
@is_super_admin
def get_permissions_for_group(req, group_id):
    permissions = tableDescription.get_permissions_for_group(group_id)
    return JsonResponse({'message':permissions}, status=201)

@require_GET
@csrf_exempt
@is_super_admin
def get_all_permissions(req):
    permissions = tableDescription.get_all_permissions()
    return JsonResponse({'message':permissions}, status=201)


@require_POST
@csrf_exempt
@is_super_admin
def update_group_permissions(req, group_id, permission_id):
    data = validate_json(request=req)
    if isinstance(data, JsonResponse):
        return data

    assigned = data.get("assigned")
    result = tableDescription.set_group_permission_assignment(
        group_id=group_id,
        permission_id=permission_id,
        assigned=assigned,
    )

    if result.get("error"):
        status = 404 if result["error"] in ("Group not found", "Permission not found") else 400
        return JsonResponse(result, status=status)

    return JsonResponse(result, status=200)
