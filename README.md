After cloning the repo and switching to the handwritten-ocr branch follow these instructions.

## 1. Create virtual environment

```bash
python -m venv .venv
```

## 2. Activate the virtual environment

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

## 4. Install the dependencies

```bash
pip install -r requirements.txt
```
```bash
pip install -e .
```

## 5. Run the main file

```bash
python -m handwriting_ocr_pipeline.main
```
You can see the output in the Outputs/ folder
