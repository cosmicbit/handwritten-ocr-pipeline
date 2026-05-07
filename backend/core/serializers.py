from decimal import Decimal, InvalidOperation


class RequestValidationError(Exception):
    def __init__(self, errors):
        super().__init__("Invalid request payload")
        self.errors = errors


class BaseRequestSchema:
    required_fields = ()

    def __init__(self, data):
        self.data = data or {}
        self.validated_data = {}
        self.errors = {}

    def is_valid(self):
        self.errors = {}
        self.validated_data = {}

        if not isinstance(self.data, dict):
            self.errors["non_field_errors"] = ["Payload must be a JSON object"]
            return False

        for field in self.required_fields:
            value = self.data.get(field)
            if value in (None, ""):
                self.errors[field] = ["This field is required."]

        if self.errors:
            return False

        try:
            self.validated_data = self.validate()
        except RequestValidationError as exc:
            self.errors = exc.errors
            return False

        return True

    def validate(self):
        return {}

    def raise_for_errors(self):
        if not self.is_valid():
            raise RequestValidationError(self.errors)
        return self.validated_data

    def _parse_int(self, field_name, *, minimum=None):
        value = self.data.get(field_name)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise RequestValidationError({field_name: ["A valid integer is required."]})

        if minimum is not None and parsed < minimum:
            raise RequestValidationError(
                {field_name: [f"Ensure this value is greater than or equal to {minimum}."]}
            )
        return parsed


class TeacherStudentAnswerSheetRequestSchema(BaseRequestSchema):
    required_fields = ("teacher_id", "student_id", "department_id", "subject_id")

    def validate(self):
        return {
            "teacher_id": self._parse_int("teacher_id", minimum=1),
            "student_id": self._parse_int("student_id", minimum=1),
            "department_id": self._parse_int("department_id", minimum=1),
            "subject_id": self._parse_int("subject_id", minimum=1),
        }


class TeacherStudentMarkUpdateRequestSchema(BaseRequestSchema):
    required_fields = ("teacher_id", "student_id", "department_id", "subject_id", "acquired_mark")

    def validate(self):
        validated = {
            "teacher_id": self._parse_int("teacher_id", minimum=1),
            "student_id": self._parse_int("student_id", minimum=1),
            "department_id": self._parse_int("department_id", minimum=1),
            "subject_id": self._parse_int("subject_id", minimum=1),
        }

        raw_mark = self.data.get("acquired_mark")
        try:
            parsed_mark = Decimal(str(raw_mark))
        except (InvalidOperation, TypeError, ValueError):
            raise RequestValidationError({"acquired_mark": ["A valid numeric value is required."]})

        if parsed_mark < 0:
            raise RequestValidationError(
                {"acquired_mark": ["Ensure this value is greater than or equal to 0."]}
            )

        if parsed_mark != parsed_mark.to_integral_value():
            raise RequestValidationError({"acquired_mark": ["A whole number is required."]})

        validated["acquired_mark"] = int(parsed_mark)
        return validated
