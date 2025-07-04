from flask import Flask, request, render_template_string
from src.data_loader import get_data_generators
from src.predict import predict_image
import os

# Prepare class labels from training directory
train_dir = os.path.join('data', 'train')
sample_gen = next(os.walk(train_dir))[1]
class_labels = sorted(sample_gen)

# Map class names to indices
class_indices = {cls: idx for idx, cls in enumerate(class_labels)}

HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <title>Garbage Classification</title>
    <link rel="icon" href="https://cdn-icons-png.flaticon.com/512/679/679922.png">
    <style>
      body { background: #f8f9fa; }
      .container { max-width: 500px; margin-top: 60px; }
      .card { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
      .prediction { font-size: 1.2rem; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="text-center mb-4">
        <img src="https://cdn-icons-png.flaticon.com/512/679/679922.png" width="60" alt="Garbage Icon">
        <h2 class="mt-2">Garbage Classification</h2>
        <p class="text-muted">Upload an image to classify its garbage type</p>
      </div>
      <div class="card p-4">
        <form method="post" enctype="multipart/form-data">
          <div class="mb-3">
            <input class="form-control" type="file" name="file" required>
          </div>
          <button class="btn btn-primary w-100" type="submit">Classify</button>
        </form>
        {% if prediction %}
          <div class="alert alert-success mt-4 text-center prediction">
            <strong>Prediction:</strong> {{ prediction }}<br>
            <span class="text-secondary">Confidence: {{ confidence }}%</span>
          </div>
        {% endif %}
      </div>
      <footer class="text-center mt-4 text-muted" style="font-size:0.9rem;">&copy; 2024 Garbage Classifier</footer>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = f"temp_{file.filename}"
            file.save(filepath)
            label, conf = predict_image(
                filepath,
                model_path='models/garbage_classifier.h5',
                img_size=(224,224),
                class_indices=class_indices
            )
            os.remove(filepath)
            prediction = label
            confidence = round(float(conf) * 100, 2)
    return render_template_string(HTML, prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)