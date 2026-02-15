from django.urls import path
from . import views

urlpatterns = [
    path('test', views.test, name='rbac_hello'),
    path('admin', views.admin, name='rbac_admin'),
    path('test', views.test, name='rbac_hello'),
    path('tables/<str:table_name>/desc/',views.get_table_description, name='description'),
    path('tables', views.get_tables, name="rbac_tables"),
    path('tables/<str:table_name>/data', views.get_table_data, name="rbac_tables"),
    path('tables/<str:table_name>/data/<int:id>', views.update_table_data, name="rbac_update_table_data"),
    path('groups', views.get_groups, name="rbac_groups"),
    path('groups/<int:group_id>/permissions', views.get_permissions_for_group, name="rbac_group_permissions"),
    path('groups/<int:group_id>/permissions/<int:permission_id>/update', views.update_group_permissions, name="rbac_update_group_permissions"),
    path('permissions', views.get_all_permissions, name="rbac_all_permissions"),

]
