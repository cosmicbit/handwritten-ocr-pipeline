After cloning the repo and switching to the handwritten-ocr branch follow these instructions.

## 1. Install Conda

**Linux / macOS**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

## 2. Create virtual environment

```bash
conda create -p .venv python=3.10 -y
```

## 3. Activate the virtual environment

**Linux / macOS**
```bash
conda activate ./.venv
```

**Windows**
```bash
conda activate .\.venv
```

## 4. Install the dependencies

```bash
pip install -r requirements.txt
```

## 5. Run the main file

```bash
python main.py
```
You can see the output in the Outputs/ folder
