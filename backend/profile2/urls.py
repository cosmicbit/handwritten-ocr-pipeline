from django.urls import path
from . import views

urlpatterns = [
    path('', views.get_profile, name="get_profile"),
    path('update/', views.update_profile, name="update_profile"),
    path('languages/<int:offset>/<int:limit>/', views.get_languages, name="get_languages"),
    path('timezones/<int:offset>/<int:limit>/', views.get_timezones, name="get_timezones"),
    path('locations/<int:offset>/<int:limit>/', views.get_locations, name="get_locations"),

]