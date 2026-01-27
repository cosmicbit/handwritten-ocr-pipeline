from django.urls import path
from . import views

urlpatterns = [
    path('test', views.test, name='rbac_hello'),
    path('admin', views.admin, name='rbac_admin'),
    path('test', views.test, name='rbac_hello'),
    path('desc',views.get_table_description, name='description'),
    path('tables', views.get_tables, name="rbac_tables")
]