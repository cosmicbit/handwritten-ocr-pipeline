import os
import shutil
import tempfile
import logging

from asgiref.sync import async_to_sync
from celery import shared_task
from channels.layers import get_channel_layer
from django.core.files.base import ContentFile
from django.db import transaction

logger = logging.getLogger("engine.realtime")


def _task_group_name(task_id):
    return f"engine_task_{task_id}"


def _broadcast_task_status(task_id, payload):
    channel_layer = get_channel_layer()
    if not channel_layer:
        logger.warning(
            "ws_publish_skipped no_channel_layer task_id=%s event=%s",
            task_id,
            payload.get("event"),
        )
        return

    try:
        async_to_sync(channel_layer.group_send)(
            _task_group_name(task_id),
            {
                "type": "engine_status",
                "payload": payload,
            },
        )
        logger.info(
            "ws_publish_ok task_id=%s group=%s event=%s stage=%s progress=%s",
            task_id,
            _task_group_name(task_id),
            payload.get("event"),
            payload.get("stage"),
            payload.get("progress"),
        )
    except Exception:
        logger.exception(
            "ws_publish_failed task_id=%s group=%s event=%s",
            task_id,
            _task_group_name(task_id),
            payload.get("event"),
        )
        # Do not fail background evaluation if websocket publish fails.
        return


def _update_task_status(task, stage, progress, message, **extra):
    task_id = task.request.id
    payload = {
        "event": "progress",
        "task_id": task_id,
        "stage": stage,
        "progress": progress,
        "message": message,
        **extra,
    }
    task.update_state(
        state="PROGRESS",
        meta=payload,
    )
    _broadcast_task_status(task_id, payload)


def _normalize_score(score, max_score):
    normalized = int(round(score))
    if normalized < 0:
        return 0
    if normalized > max_score:
        return max_score
    return normalized


def _persist_evaluation(
    subject_id,
    student_id,
    teacher_answer_key_upload_id,
    student_pdf_upload_id,
    marks,
    teacher_answers,
    student_answers,
    student_scores,
    teacher_txt_path=None,
    student_txt_path=None,
):
    if not subject_id or not student_id:
        return None

    from core.models import Student, StudentMark, Subject, TeacherPDFUpload, TeacherSubjectAnswerKey
    from notification.service.notification_service import NotificationService

    subject = Subject.objects.filter(id=subject_id).first()
    student = Student.objects.filter(id=student_id).first()
    if not subject or not student:
        return None

    total_mark = int(round(sum(marks))) if marks else int(round(sum(student_scores))) if student_scores else 0
    acquired_mark = _normalize_score(sum(student_scores), total_mark) if student_scores else 0
    teacher_text = "\n\nNext Answer \n".join(teacher_answers)
    student_text = "\n\nNext Answer \n".join(student_answers)
    if teacher_txt_path and os.path.isfile(teacher_txt_path):
        with open(teacher_txt_path, "r", encoding="utf-8") as teacher_file:
            teacher_text = teacher_file.read()
    if student_txt_path and os.path.isfile(student_txt_path):
        with open(student_txt_path, "r", encoding="utf-8") as student_file:
            student_text = student_file.read()

    with transaction.atomic():
        mark_record, _created = StudentMark.objects.update_or_create(
            subject=subject,
            student=student,
            defaults={
                "total_mark": total_mark,
                "acquired_mark": acquired_mark,
            },
        )

        if teacher_answer_key_upload_id:
            answer_key = TeacherSubjectAnswerKey.objects.filter(id=teacher_answer_key_upload_id).first()
            if answer_key:
                if answer_key.extracted_text_file:
                    answer_key.extracted_text_file.delete(save=False)
                answer_key.extracted_text = teacher_text
                answer_key.extracted_text_file.save(
                    "teacher.txt",
                    ContentFile(teacher_text),
                    save=False,
                )
                answer_key.save(update_fields=["extracted_text", "extracted_text_file"])
        if student_pdf_upload_id:
            student_upload = TeacherPDFUpload.objects.filter(id=student_pdf_upload_id).first()
            if student_upload:
                if student_upload.extracted_text_file:
                    student_upload.extracted_text_file.delete(save=False)
                student_upload.extracted_text = student_text
                student_upload.extracted_text_file.save(
                    "student.txt",
                    ContentFile(student_text),
                    save=False,
                )
                student_upload.save(update_fields=["extracted_text", "extracted_text_file"])

    # Send direct notification to the student user after successful DB commit.
    notification_payload = {
        "title": "Marks Generated",
        "message": (
            f"Your marks for {subject.true_subject.name if subject.true_subject else 'the subject'} "
            f"have been generated: {acquired_mark}/{total_mark}."
        ),
        "type_name": "mark_generated",
        "user_ids": [student.user_id],
    }
    notification = NotificationService().create_notification(
        notification_payload,
        created_by_id=None,
    )
    logger.info(
        "student_notification_created student_user_id=%s notification_id=%s mark_id=%s",
        student.user_id,
        notification.id,
        mark_record.id,
    )

    return mark_record.id


@shared_task(bind=True)
def run_engine_task(
    self,
    teacher_pdf_path=None,
    student_pdf_path=None,
    marks=None,
    subject_id=None,
    student_id=None,
    teacher_answer_key_upload_id=None,
    student_pdf_upload_id=None,
):
    temp_output_dir = None
    task_id = self.request.id
    logger.info("task_started task_id=%s subject_id=%s student_id=%s", task_id, subject_id, student_id)
    try:
        _update_task_status(self, "initializing", 5, "Initializing engine task")
        from core.engine import DEFAULT_STUDENT_PDF, DEFAULT_TEACHER_PDF, run_pipeline

        _update_task_status(self, "validating_input", 15, "Validating input files and marks")
        teacher_path = teacher_pdf_path or DEFAULT_TEACHER_PDF
        student_path = student_pdf_path or DEFAULT_STUDENT_PDF

        if not os.path.isfile(teacher_path):
            raise ValueError(f"Teacher PDF file not found: {teacher_path}")
        if not os.path.isfile(student_path):
            raise ValueError(f"Student PDF file not found: {student_path}")

        if marks is not None:
            if not isinstance(marks, list) or not marks:
                raise ValueError("marks must be a non-empty list")
            if not all(isinstance(mark, (int, float)) for mark in marks):
                raise ValueError("marks must contain only numbers")

        _update_task_status(self, "running_model", 35, "Running OCR and scoring model")
        temp_output_dir = tempfile.mkdtemp(prefix="engine-output-")
        teacher_out_path = os.path.join(temp_output_dir, "teacher.txt")
        student_out_path = os.path.join(temp_output_dir, "student.txt")
        teacher_answers, every_student_answers, every_student_scores = run_pipeline(
            teacher_pdf_path=teacher_path,
            student_pdf_path=student_path,
            marks=marks,
            out_teacher_path=teacher_out_path,
            out_student_path=student_out_path,
        )

        student_answers = every_student_answers[0] if every_student_answers else []
        student_scores = every_student_scores[0] if every_student_scores else []
        _update_task_status(self, "saving_results", 85, "Saving marks and extracted text")
        student_mark_id = _persist_evaluation(
            subject_id=subject_id,
            student_id=student_id,
            teacher_answer_key_upload_id=teacher_answer_key_upload_id,
            student_pdf_upload_id=student_pdf_upload_id,
            marks=marks or [],
            teacher_answers=teacher_answers,
            student_answers=student_answers,
            student_scores=student_scores,
            teacher_txt_path=teacher_out_path,
            student_txt_path=student_out_path,
        )

        payload = {
            "event": "success",
            "task_id": task_id,
            "teacher_pdf_path": teacher_path,
            "student_pdf_path": student_path,
            "subject_id": subject_id,
            "student_id": student_id,
            "teacher_answers_count": len(teacher_answers),
            "student_answers_count": len(student_answers),
            "scores": student_scores,
            "total_score": sum(student_scores) if student_scores else 0,
            "student_mark_id": student_mark_id,
            "teacher_txt": "\n\nNext Answer \n".join(teacher_answers),
            "student_txt": "\n\nNext Answer \n".join(student_answers),
        }
        _update_task_status(self, "finalizing", 95, "Finalizing task result")
        _broadcast_task_status(task_id, payload)
        logger.info("task_succeeded task_id=%s student_mark_id=%s", task_id, student_mark_id)
        return payload
    except ValueError as exc:
        # Let Celery store a proper exception payload for FAILURE state.
        _broadcast_task_status(
            task_id,
            {
                "event": "failure",
                "task_id": task_id,
                "stage": "failed",
                "progress": 100,
                "message": str(exc),
                "status": 400,
            },
        )
        logger.warning("task_failed_validation task_id=%s error=%s", task_id, str(exc))
        raise ValueError(str(exc))
    except ModuleNotFoundError as exc:
        # Dependency issues should fail task cleanly without corrupting backend state.
        _broadcast_task_status(
            task_id,
            {
                "event": "failure",
                "task_id": task_id,
                "stage": "failed",
                "progress": 100,
                "message": f"Missing dependency: {exc.name}",
                "status": 500,
            },
        )
        logger.error("task_failed_dependency task_id=%s dependency=%s", task_id, exc.name)
        raise RuntimeError(f"Missing dependency: {exc.name}") from exc
    except Exception as exc:
        _broadcast_task_status(
            task_id,
            {
                "event": "failure",
                "task_id": task_id,
                "stage": "failed",
                "progress": 100,
                "message": "Unhandled task error",
                "details": str(exc),
                "status": 500,
            },
        )
        logger.exception("task_failed_unhandled task_id=%s", task_id)
        raise RuntimeError(f"Unhandled task error: {exc}") from exc
    finally:
        if temp_output_dir and os.path.isdir(temp_output_dir):
            shutil.rmtree(temp_output_dir, ignore_errors=True)
