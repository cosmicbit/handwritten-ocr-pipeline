from django.http.response import JsonResponse

def test(req):
    return JsonResponse({
        'success': 'ok'
    },status=200)

