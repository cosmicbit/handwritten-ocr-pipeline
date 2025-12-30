@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating venv...
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt
pip install -e .

echo Running pipeline...
python -m handwriting_ocr_pipeline.main

pause
