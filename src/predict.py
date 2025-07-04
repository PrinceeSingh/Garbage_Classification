import os
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.utils import load_img, img_to_array

def predict_image(image_path, model_path, img_size=(224, 224), class_indices=None):
    """
    Predict the class of an image using a trained model.
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the trained model file
        img_size (tuple): Size to resize the image to (width, height)
        class_indices (dict): Dictionary mapping class names to indices
    
    Returns:
        tuple: (predicted_class, confidence)
    """
    # Load the model
    model = load_model(model_path)
    
    # Load and preprocess the image
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Convert index back to class name
    if class_indices:
        # Reverse the mapping to get class name from index
        class_names = {v: k for k, v in class_indices.items()}
        predicted_class = class_names.get(predicted_class_idx, f"Class_{predicted_class_idx}")
    else:
        predicted_class = f"Class_{predicted_class_idx}"
    
    return predicted_class, confidence 