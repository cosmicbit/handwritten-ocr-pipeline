import json

from celery import shared_task
from core.services.core_service import CoreService, CoreServiceError

@shared_task(bind=True)
def run_engine_task(self, teacher_pdf_path=None, student_pdf_path=None, marks=None):
    service = CoreService()
    try:
        message, _status = service.trigger_engine_model(teacher_pdf_path, student_pdf_path, marks)
        return message
    except CoreServiceError as exc:
        payload = exc.to_response_body()
        payload["status"] = exc.status
        raise Exception(json.dumps(payload))
    except Exception as exc:
        raise Exception(json.dumps({"error": "Unhandled task error", "details": str(exc), "status": 500}))
