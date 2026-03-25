import os
import json
import numpy as np
from flask import Flask, request, render_template, url_for
from PIL import Image
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join('model', 'food_model.h5')
CALORIES_PATH = 'calories.json'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
IMG_SIZE = (224, 224)

# Must be in ALPHABETICAL ORDER — Keras sorts class folders alphabetically
CLASSES = ['caesar_salad', 'club_sandwich', 'french_fries', 'hamburger', 'pizza']

# Display-friendly names for the UI
DISPLAY_NAMES = {
    'caesar_salad': 'Caesar Salad',
    'club_sandwich': 'Club Sandwich',
    'french_fries': 'French Fries',
    'hamburger': 'Hamburger',
    'pizza': 'Pizza'
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────
# Load Model & Calorie Database
# ─────────────────────────────────────────────
print("🔄 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Please run 'python train_model.py' first to train the model.")
    model = None

with open(CALORIES_PATH, 'r') as f:
    calories_db = json.load(f)
print(f"✅ Calorie database loaded: {calories_db}")


# ─────────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────────
def predict_image(img_path):
    """
    Preprocesses an image and runs model inference.
    Returns: (food_label, confidence_percent, display_name)
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0            # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension → (1, 224, 224, 3)

    predictions = model.predict(img_array, verbose=0)  # Shape: (1, 5)
    class_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100.0

    food_label = CLASSES[class_idx]
    display_name = DISPLAY_NAMES.get(food_label, food_label.replace('_', ' ').title())
    return food_label, confidence, display_name


# ─────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None

    if request.method == 'POST':
        if model is None:
            error = "Model not loaded. Please run train_model.py first."
            return render_template('index.html', result=result, error=error)

        if 'file' not in request.files or request.files['file'].filename == '':
            error = "No file selected. Please choose an image."
            return render_template('index.html', result=result, error=error)

        file = request.files['file']
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        ext = os.path.splitext(file.filename)[1].lower()

        if ext not in allowed_extensions:
            error = "Invalid file type. Please upload a JPG, PNG, or WebP image."
            return render_template('index.html', result=result, error=error)

        # Save uploaded file
        filename = 'upload' + ext
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        try:
            food_label, confidence, display_name = predict_image(save_path)
            calories = calories_db.get(food_label, 'N/A')

            result = {
                'food': display_name,
                'food_label': food_label,
                'calories': calories,
                'confidence': round(confidence, 2),
                'image': url_for('static', filename=f'uploads/{filename}')
            }
        except Exception as e:
            error = f"Prediction error: {str(e)}"

    return render_template('index.html', result=result, error=error)


# ─────────────────────────────────────────────
# Start Server
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🌐 Starting Flask server...")
    print("   Open your browser at: http://127.0.0.1:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
