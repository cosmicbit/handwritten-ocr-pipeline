from django.db import IntegrityError

from rbac.helpers.table_insert_helper import (
    TableInsertHelper,
    TableInsertValidationError,
)
from rbac.repositories.table_data_repository import TableDataRepository


class TableDataService:
    def __init__(self):
        self.insert_helper = TableInsertHelper()
        self.table_repository = TableDataRepository()

    def create_table_row(self, table_name, payload):
        try:
            self.insert_helper.validate_payload(table_name=table_name, payload=payload)
            inserted_row = self.table_repository.insert_row(
                table_name=table_name,
                payload=payload,
            )
            return {"error": None, "message": inserted_row}
        except TableInsertValidationError as exc:
            return {"error": exc.message, "message": None}
        except IntegrityError as exc:
            return {"error": self._map_pg_error(exc), "message": None}
        except Exception:
            return {"error": "Failed to insert row", "message": None}

    def _map_pg_error(self, error):
        pg_error = getattr(error, "__cause__", None)
        pg_code = getattr(pg_error, "pgcode", None) or getattr(pg_error, "sqlstate", None)

        if pg_code == "23502":
            return "A required column is missing or null (not_null_violation)"
        if pg_code == "23503":
            return "Referenced record does not exist (foreign_key_violation)"
        if pg_code == "23505":
            return "Duplicate value violates a unique constraint (unique_violation)"

        return "Database integrity error"
