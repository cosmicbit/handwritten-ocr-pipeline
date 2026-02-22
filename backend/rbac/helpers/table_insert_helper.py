from django.db import connection


class TableInsertValidationError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class TableInsertHelper:
    def table_exists(self, table_name):
        return table_name in connection.introspection.table_names()

    def get_table_columns_metadata(self, table_name):
        query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                is_identity
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
            ORDER BY ordinal_position
        """
        with connection.cursor() as cursor:
            cursor.execute(query, [table_name])
            rows = cursor.fetchall()

        metadata = []
        for row in rows:
            column_name, data_type, is_nullable, column_default, is_identity = row
            metadata.append(
                {
                    "column_name": column_name,
                    "data_type": data_type,
                    "is_nullable": is_nullable,
                    "column_default": column_default,
                    "is_identity": is_identity,
                    "is_serial": bool(column_default and "nextval(" in column_default),
                }
            )
        return metadata

    def validate_payload(self, table_name, payload):
        if not isinstance(payload, dict):
            raise TableInsertValidationError("Request body must be a JSON object")

        if not self.table_exists(table_name):
            raise TableInsertValidationError("No matching table found")

        columns_metadata = self.get_table_columns_metadata(table_name)
        if not columns_metadata:
            raise TableInsertValidationError("Table has no insertable columns")

        valid_columns = {column["column_name"] for column in columns_metadata}
        unknown_columns = sorted(set(payload.keys()) - valid_columns)
        if unknown_columns:
            raise TableInsertValidationError(
                f"Unknown column(s): {', '.join(unknown_columns)}"
            )

        missing_required_columns = []
        for column in columns_metadata:
            column_name = column["column_name"]
            nullable = column["is_nullable"] == "YES"
            has_default = column["column_default"] is not None
            auto_generated = column["is_identity"] == "YES" or column["is_serial"]

            if column_name in payload:
                continue

            if not nullable and not has_default and not auto_generated:
                missing_required_columns.append(column_name)

        if missing_required_columns:
            raise TableInsertValidationError(
                "Missing required column(s): " + ", ".join(sorted(missing_required_columns))
            )

        return columns_metadata

