from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("Red Blood Cell.h5")

# Define class labels
classes = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

# Folder to store uploaded images
UPLOAD_FOLDER = 'images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allow Flask to serve files from the "images" directory
@app.route('/images/<filename>')
def serve_uploaded_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    # Save uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess image
    img = load_img(file_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = classes[predicted_index]
    confidence = float(np.max(predictions[0]) * 100)

    # Create URL for displaying image
    image_url = f'/images/{file.filename}'

    # Render result
    return render_template("result.html",
                           image_file=image_url,
                           prediction=predicted_class,
                           confidence=round(confidence, 2))

if __name__ == "__main__":
    app.run(debug=True)
