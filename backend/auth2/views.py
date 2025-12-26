from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.contrib.auth import authenticate, login
import json
from .utils.jwt_utils import create_jwt_token 
from django.contrib.auth.models import User
from .utils.decorator import public_view

# test api

def test(req):
    return HttpResponse("hello from the server")

@public_view
def regiter(req):
    if req.method != 'POST':
        return JsonResponse({
            'error':'POST required'
        }, status = 400)
    data = json.loads(req.body)
    username = data.get('username')
    password = data.get('password')

    if User.objects.filter(username=username).exists():
        return  JsonResponse({
            'error': 'User already exists'
        }, status=400)
    
    user = User.objects.create_user(username=username, password=password)
    return JsonResponse({
        'message': 'User registered'
    })

@public_view
def login(req):
    if req.method != 'POST':
        return JsonResponse({
            'message':'',
            'error': 'POST required'
        }, status = 400)
    data = json.loads(req.body)

    username = data.get('username')
    password = data.get('password')

    user = authenticate(req, username=username, password=password)

    if user is None:
        return JsonResponse({
            'message': '',
            'error': 'Invalid credentails'
        }, status = 400)
    
    token = create_jwt_token(user)

    return JsonResponse({
        'message': {
            'success' : 'Login successfull',
            'token': token,
            'error' : ''
        }
    })