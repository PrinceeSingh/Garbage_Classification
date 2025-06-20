# Garbage Classification

A computer vision project that classifies images of garbage into different categories (e.g., plastic, metal, glass, paper).

## Features

- Data loading and preprocessing
- Model training (CNN)
- Evaluation and prediction
- Simple web interface (Flask)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in `data/train` and `data/test` directories.
2. Train the model:
    ```bash
    python src/train.py
    ```
3. Run the app:
    ```bash
    python app.py
    ```
