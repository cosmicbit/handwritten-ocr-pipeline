from django.urls import path
from . import views

urlpatterns = [
    path('test', views.test, name='hello'),
    path('register', views.register, name='register'),
    path('login', views.login, name='login'),
    path('desc',views.get_table_description, name='description')
]