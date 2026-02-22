from detectors.craft_detector import CraftDetector
from trocr_recognizer import TrocrRecognizer

import utils



from PIL import Image
from sentence_transformers import SentenceTransformer, util
import nltk
import re
import numpy as np
import cv2
import fitz
import io

teacher = "inputs/teacher.pdf"
student = "inputs/student.pdf"
out_teacher = "outputs/teacher.txt"
out_student = "outputs/student.txt"

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

def get_embedding(model, str):
    return model.encode(str, convert_to_tensor=True)

def phaseThree(teacher_answers, every_students_answers, marks, model = SentenceTransformer('all-MiniLM-L6-v2')):
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
        utils.draw_boxes(image, boxes, save_path="outputs/words_output.png")
        utils.draw_boxes_of_lines(image, lines, save_path="outputs/lines_output.png")
    
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

def main():
    craft = CraftDetector("models/craft_mlt_25k.pth")
    recognizer = TrocrRecognizer("models/trocr-base")
    analyser = SentenceTransformer("models/all-MiniLM-L6-v2")

    teacher_answers = getAnswers(teacher, craft, recognizer)
    utils.save_text(out_teacher, "\n\nNext Answer \n".join(teacher_answers))

    student1_answer = getAnswers(student, craft, recognizer)
    utils.save_text(out_student, "\n\nNext Answer \n".join(student1_answer))

    marks = [5, 5, 5]
    every_student_answers = []
    every_student_answers.append(student1_answer)

    every_student_scores = phaseThree(teacher_answers, every_student_answers, marks, analyser)
    for i in range(len(every_student_scores)):
        print("Student ", i, ": ", end="")
        student_scores = every_student_scores[i]
        for j in range(len(student_scores)):
            print(student_scores[j],"/", marks[j], end=" ")
        print()

if __name__ == "__main__":
    main()