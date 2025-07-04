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

    ## Dataset

This project uses the [Garbage Classification dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) from Kaggle.

### How to Download

1. Sign up/log in to [Kaggle](https://kaggle.com).
2. Install the Kaggle CLI:
   ```
   pip install kaggle
   ```
3. Get your API token from your Kaggle account ([instructions here](https://github.com/Kaggle/kaggle-api#api-credentials)) and place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<YourUsername>\.kaggle\` (Windows).
4. Download the dataset:
   ```
   kaggle datasets download mostafaabla/garbage-classification
   unzip garbage-classification.zip -d data/
   ```
5. The dataset will be available in the `data/` directory.

> **Note**: The dataset is not included in this repository due to its size and Kaggle's Terms of Use.

See [`data/README.md`](data/README.md) for more info.

## Screenshot

Here is how the app looks:

![App Screenshot](assets/screenshot(150).png)
