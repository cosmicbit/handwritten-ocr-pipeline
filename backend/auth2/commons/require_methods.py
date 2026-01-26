from django.http import JsonResponse

POST_METHOD_REQUIRED = "POST required"
GET_METHOD_REQUIRED = "GET required"
PUT_METHOD_REQUIRED = "PUT required"
PATCH_METHOD_REQUIRED = "PATCH required"
DELETE_METHOD_REQUIRED = "DELETE required"
UPDATE_METHOD_REQUIRED = "UPDATE required"

def require_post(req):
    if req.method != 'POST':
        return JsonResponse({ 'error': POST_METHOD_REQUIRED },status=400)
    return None

def require_get(req):
    if req.method != 'GET':
        return JsonResponse({ 'error': GET_METHOD_REQUIRED}, status=400)
    return None

