from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import os

# Load model
model = load_model("models/garbage_classifier.h5")  # or garbage_classifier.h5

# Map class indices to labels
# Use the same order as during training
class_names = sorted(os.listdir("data/train"))  # assumes you didn't shuffle or rename
idx_to_label = {i: label for i, label in enumerate(class_names)}

# Load and preprocess image
img_path = "paper255.jpg"  # change to your test image path
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
predicted_class = idx_to_label[np.argmax(pred)]
print(f"Predicted class: {predicted_class}")