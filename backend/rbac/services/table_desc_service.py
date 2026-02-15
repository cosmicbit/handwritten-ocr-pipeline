from django.db import connection
from ..models import *
from auth2.models import *
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.contrib.auth.models import Permission

class TableDescriptionService:
    def __init__(self):
        pass
    

    def get_table_description(self,table_name: str):

        with connection.cursor() as cursor:
            columns = connection.introspection.get_table_description(
                cursor,
                table_name
            )

        return {
            "table": table_name,
            "columns": [
                {
                    "name": col.name,
                    "null": col.null_ok,
                    "type": col.type_code,
                }
                for col in columns
            ]
        }
    

    def table_exists(self, table_name: str) -> bool:
        return table_name in connection.introspection.table_names()
    
    
    def get_all_tables(self):
        return connection.introspection.table_names()
    
    
    def get_table_data(self, table_name, page_size=20, page=1):
        if not self.table_exists(table_name):
            return {'error':'No matching table found'}

        offset = (page - 1) * page_size
        with connection.cursor() as cursor:
            query = f"""
                SELECT *
                FROM {table_name} order by id
                LIMIT %s OFFSET %s
            """
            cursor.execute(query, [page_size, offset])
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]

        print(rows)
        return [dict(zip(columns, row)) for row in rows]
    
    from django.db import connection

    def update_table_data(self, table_name, row_id, data: dict):
        if not self.table_exists(table_name):
            return {"error": "No matching table found"}

        if not data:
            return {"error": "No data to update"}

        columns = data.keys()

        # column1 = %s, column2 = %s
        set_clause = ", ".join([f"{col} = %s" for col in columns])

        values = list(data.values())
        values.append(row_id)

        query = f"""
            UPDATE {table_name}
            SET {set_clause}
            WHERE id = %s
        """

        with connection.cursor() as cursor:
            cursor.execute(query, values)

        return {"message": "Record updated successfully"}
    

    def get_groups(self):
        return list(Group.objects.all().values("id", "name"))
    

    def get_permissions_for_group(self, group_id):
        try:
            group = Group.objects.get(id=group_id)
        except Group.DoesNotExist:
            return {"error": "Group not found"}

        permissions = group.permissions.all()
        return list(permissions.values("id", "name", "codename"))
    
    def get_all_permissions(self):
        permissions = Permission.objects.all()
        return list(permissions.values("id", "name", "codename"))

    def update_group_permissions(self, group_id, permission_ids):
        try:
            group = Group.objects.get(id=group_id)
        except Group.DoesNotExist:
            return {"error": "Group not found"}

        if permission_ids is None:
            return {"error": "permission_ids is required"}

        if not isinstance(permission_ids, list):
            return {"error": "permission_ids must be a list"}

        normalized_ids = []
        for permission_id in permission_ids:
            if isinstance(permission_id, bool):
                return {"error": "permission_ids must contain integers"}
            if not isinstance(permission_id, int):
                return {"error": "permission_ids must contain integers"}
            normalized_ids.append(permission_id)

        permissions = Permission.objects.filter(id__in=normalized_ids)
        found_ids = set(permissions.values_list("id", flat=True))
        missing_ids = sorted(set(normalized_ids) - found_ids)

        if missing_ids:
            return {"error": f"Invalid permission id(s): {missing_ids}"}

        group.permissions.set(permissions)
        return {
            "message": "Group permissions updated successfully",
            "group_id": group.id,
            "permission_ids": sorted(found_ids),
        }

    def set_group_permission_assignment(self, group_id, permission_id, assigned):
        try:
            group = Group.objects.get(id=group_id)
        except Group.DoesNotExist:
            return {"error": "Group not found"}

        try:
            permission = Permission.objects.get(id=permission_id)
        except Permission.DoesNotExist:
            return {"error": "Permission not found"}

        if not isinstance(assigned, bool):
            return {"error": "assigned must be a boolean"}

        if assigned:
            group.permissions.add(permission)
            action = "assigned"
        else:
            group.permissions.remove(permission)
            action = "removed"

        return {
            "message": f"Permission {action} successfully",
            "group_id": group.id,
            "permission_id": permission.id,
            "assigned": assigned,
        }
