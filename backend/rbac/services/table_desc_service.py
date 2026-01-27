from django.db import connection

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