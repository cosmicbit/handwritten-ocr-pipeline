from django.db import connection
from ..models import *
from auth2.models import *

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
    