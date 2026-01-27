import json 
from django.http import JsonResponse

INVALID_JSON_ERROR={
            'error': 'Invalid JSON format'
        }

def validate_json(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError as e:
        return JsonResponse(INVALID_JSON_ERROR,status=401)
    return data