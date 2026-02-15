from django.http import JsonResponse
from .service.profile_service import ProfileService
from django.views.decorators.csrf import csrf_exempt
import json

profileService = ProfileService()

@csrf_exempt
def get_profile(req):
    user_id = req.user.id
    response = profileService.get_profile(user_id=user_id)
    if 'error' in response:
        return JsonResponse({
            'error': response['error']
        }, status=404)
    return JsonResponse({
        'message': response['message']
    }, status=200)

@csrf_exempt
def update_profile(req):
    user_id = req.user.id
    data = profileService.convert_formdata_to_dict(req)
    response = profileService.update_profile(user_id=user_id, data=data, files=req.FILES)
    if 'error' in response:
        return JsonResponse({
            'error': response['error']
        }, status=400)
    return JsonResponse({
        'message': response['message']
    }, status=200)
    
@csrf_exempt
def get_languages(req, offset, limit): 
    reponse = profileService.get_language(offset=offset, limit=limit)
    return JsonResponse({
        "message":reponse['message']
        }, status=200)

@csrf_exempt
def get_timezones(req, offset, limit): 
    reponse = profileService.get_timezone(offset=offset, limit=limit)
    return JsonResponse({
        "message":reponse['message']
        }, status=200)

@csrf_exempt
def get_locations(req, offset, limit): 
    reponse = profileService.get_location(offset=offset, limit=limit)
    return JsonResponse({
        "message":reponse['message']
        }, status=200)