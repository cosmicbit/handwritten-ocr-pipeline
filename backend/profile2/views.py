from django.http import JsonResponse
from .service.profile_service import ProfileService
from django.views.decorators.csrf import csrf_exempt

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
    