After cloning the repo and follow these instructions.

## 1. Enter the directory
```bash
cd handwritten-ocr-pipeline
```

## 2. Create virtual environment

```bash
python -m venv .venv
```

## 3. Activate the virtual environment

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
## 5. Enter the core directory

```bash
cd core
```

## 6. Run the main file

```bash
python main.py
```
You can see the output in the Outputs/ folder
