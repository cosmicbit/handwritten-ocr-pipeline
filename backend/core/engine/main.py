import os

from .detectors.craft_detector import CraftDetector
from .trocr_recognizer import TrocrRecognizer
from . import utils
from PIL import Image
from sentence_transformers import SentenceTransformer, util

import cv2
import fitz
import numpy as np

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(ENGINE_DIR, "inputs")
OUTPUTS_DIR = os.path.join(ENGINE_DIR, "outputs")
MODELS_DIR = os.path.join(ENGINE_DIR, "models")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(ENGINE_DIR)))
PROJECT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def _first_existing_path(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _first_model_dir_with_file(candidates, required_filename):
    for path in candidates:
        if path and os.path.isdir(path) and os.path.isfile(os.path.join(path, required_filename)):
            return path
    return None

DEFAULT_TEACHER_PDF = os.path.join(INPUTS_DIR, "teacher.pdf")
DEFAULT_STUDENT_PDF = os.path.join(INPUTS_DIR, "student.pdf")
DEFAULT_TEACHER_TXT = os.path.join(OUTPUTS_DIR, "teacher.txt")
DEFAULT_STUDENT_TXT = os.path.join(OUTPUTS_DIR, "student.txt")
DEFAULT_CRAFT_MODEL = os.path.join(MODELS_DIR, "craft_mlt_25k.pth")
DEFAULT_TROCR_MODEL = _first_model_dir_with_file(
    [
        os.path.join(MODELS_DIR, "trocr-base"),
        os.path.join(MODELS_DIR, "trocr-local"),
        os.path.join(PROJECT_MODELS_DIR, "trocr-local"),
    ],
    "config.json",
) or "microsoft/trocr-base-handwritten"
DEFAULT_ANALYSER_MODEL = _first_model_dir_with_file(
    [
        os.path.join(MODELS_DIR, "all-MiniLM-L6-v2"),
        os.path.join(PROJECT_MODELS_DIR, "all-MiniLM-L6-v2"),
    ],
    "modules.json",
) or "sentence-transformers/all-MiniLM-L6-v2"

def merge_boxes_into_lines(boxes, y_threshold=25):
    boxes = sorted(boxes, key=lambda b: b[:,1].mean())
    lines = []

    for box in boxes:
        cy = box[:,1].mean()

        placed = False

        for line in lines:
            ly = np.mean([b[:,1].mean() for b in line])

            if abs(cy - ly) < y_threshold:
                line.append(box)
                placed = True
                break

        if not placed:
            lines.append([box])

    return lines

def pdf_to_numpy_list(pdf_path, zoom=2.0):
    image_list = []
    try:
        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(zoom, zoom)

        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            image_list.append(np.array(img))
        doc.close()

        return image_list

    except Exception as e:
        print(f"Error: {e}")
        return []

def compare(teacher_embedding, student_embedding, mark):
    cosine_score = util.cos_sim(teacher_embedding, student_embedding).item()
    return round((cosine_score * mark), 2)

def get_embedding(model, text):
    return model.encode(text, convert_to_tensor=True)

def phaseThree(teacher_answers, every_students_answers, marks, model=None):
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    if len(teacher_answers) != len(marks):
        print("Marks length is not equal to number of teacher answers")
        return []

    teacher_embeddings = []
    for answer in teacher_answers:
        teacher_embeddings.append(get_embedding(model, answer))
    print("Extracted Embeddings from teachers answers")

    every_students_scores = []
    for student_answers in every_students_answers:

        if len(teacher_answers) != len(student_answers):
            print("Failed to calculate marks of  student: missing answers")
            continue
        
        student_scores = []
        for i in range(len(teacher_answers)):
            answer_embedding = get_embedding(model, student_answers[i])
            score = compare(teacher_embeddings[i], answer_embedding, marks[i])
            student_scores.append(score)
        every_students_scores.append(student_scores)
    return every_students_scores

def getAnswers(pdfPath, craft, recognizer):
    images = pdf_to_numpy_list(pdfPath)
    
    recognized_lines = []
    for image in images:
        boxes = craft.detect(image)
        lines = merge_boxes_into_lines(boxes)

        # draw boxes
        utils.draw_boxes(image, boxes, save_path=os.path.join(OUTPUTS_DIR, "words_output.png"))
        utils.draw_boxes_of_lines(image, lines, save_path=os.path.join(OUTPUTS_DIR, "lines_output.png"))
    
        for i in range(len(lines)):
            line_array = np.concatenate(lines[i], axis=0)
            x_min, y_min = line_array[:,0].min(), line_array[:,1].min()
            x_max, y_max = line_array[:,0].max(), line_array[:,1].max()

            # Crop line for recognition
            line_img = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            crop = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            pil_img = Image.fromarray(crop)
            text = recognizer.recognize(pil_img)
            progress = (i / len(lines)) * 100
            utils.print_progress_bar(progress, 100)
            recognized_lines.append(text)
        utils.print_progress_bar(100, 100)

    answers = []
    answer = ""
    for line in recognized_lines:
        if line.strip().lower() == "end":
            answers.append(answer)
            answer = ""
        else:
            answer = answer + "\n" + line

    
    return answers


def build_models(
    craft_model_path=DEFAULT_CRAFT_MODEL,
    trocr_model_path=DEFAULT_TROCR_MODEL,
    analyser_model_path=DEFAULT_ANALYSER_MODEL,
):
    craft = CraftDetector(craft_model_path)
    recognizer = TrocrRecognizer(trocr_model_path)
    analyser = SentenceTransformer(analyser_model_path)
    return craft, recognizer, analyser


def run_pipeline(
    teacher_pdf_path=DEFAULT_TEACHER_PDF,
    student_pdf_path=DEFAULT_STUDENT_PDF,
    marks=None,
    out_teacher_path=DEFAULT_TEACHER_TXT,
    out_student_path=DEFAULT_STUDENT_TXT,
    craft=None,
    recognizer=None,
    analyser=None,
    progress_callback=None,
):
    def report_progress(stage, progress, message, **extra):
        if progress_callback:
            progress_callback(stage, progress, message, **extra)

    if marks is None:
        marks = [5, 5, 5]

    if not craft or not recognizer or not analyser:
        report_progress("running_model", 40, "Loading OCR and scoring models")
        craft, recognizer, analyser = build_models()

    report_progress("running_model", 50, "Extracting teacher answers")
    teacher_answers = getAnswers(teacher_pdf_path, craft, recognizer)
    utils.save_text(out_teacher_path, "\n\nNext Answer \n".join(teacher_answers))

    report_progress("running_model", 65, "Extracting student answers")
    student1_answer = getAnswers(student_pdf_path, craft, recognizer)
    utils.save_text(out_student_path, "\n\nNext Answer \n".join(student1_answer))

    every_student_answers = [student1_answer]
    report_progress("running_model", 75, "Scoring answers")
    every_student_scores = phaseThree(teacher_answers, every_student_answers, marks, analyser)
    report_progress("running_model", 80, "OCR and scoring finished")
    return teacher_answers, every_student_answers, every_student_scores

def main():
    marks = [5, 5, 5]
    _teacher_answers, _every_student_answers, every_student_scores = run_pipeline(
        teacher_pdf_path=DEFAULT_TEACHER_PDF,
        student_pdf_path=DEFAULT_STUDENT_PDF,
        marks=marks,
        out_teacher_path=DEFAULT_TEACHER_TXT,
        out_student_path=DEFAULT_STUDENT_TXT,
    )

    for i in range(len(every_student_scores)):
        print("Student ", i, ": ", end="")
        student_scores = every_student_scores[i]
        for j in range(len(student_scores)):
            print(student_scores[j],"/", marks[j], end=" ")
        print()

if __name__ == "__main__":
    main()
