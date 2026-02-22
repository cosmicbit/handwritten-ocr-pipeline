from django.db import connection


class TableDataRepository:
    def insert_row(self, table_name, payload):
        columns = list(payload.keys())
        values = [payload[column] for column in columns]

        quoted_table = connection.ops.quote_name(table_name)

        if columns:
            quoted_columns = [connection.ops.quote_name(column) for column in columns]
            placeholders = ", ".join(["%s"] * len(columns))
            query = f"""
                INSERT INTO {quoted_table} ({", ".join(quoted_columns)})
                VALUES ({placeholders})
                RETURNING *
            """
            params = values
        else:
            query = f"""
                INSERT INTO {quoted_table}
                DEFAULT VALUES
                RETURNING *
            """
            params = []

        with connection.cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()
            column_names = [column[0] for column in cursor.description]

        return dict(zip(column_names, row))

