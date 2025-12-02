"""
Flask REST API for Image Mood Classification - Professional Frontend
"""
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
from tensorflow import keras
import os
from utils import (load_image_from_bytes, extract_hsv_histogram,
                  extract_lbp_features, extract_dominant_colors)

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Load models at startup
print("Loading models...")
try:
    rf_model = joblib.load('trainedModels/mood_classifier_rf.pkl')
    label_encoder = joblib.load('trainedModels/label_encoder.pkl')
    cnn_model = keras.models.load_model('trainedModels/mood_classifier_cnn.h5')
    cnn_class_names = joblib.load('trainedModels/cnn_class_names.pkl')
    print("✓ Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    rf_model = None
    label_encoder = None
    cnn_model = None
    cnn_class_names = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_ml(image_bytes):
    """Predict mood using machine learning model"""
    if rf_model is None or label_encoder is None:
        raise ValueError("ML model not loaded")
    
    img_resized, img_hsv, img_gray = load_image_from_bytes(image_bytes)
    
    hsv_hist = extract_hsv_histogram(img_hsv)
    lbp_hist = extract_lbp_features(img_gray)
    feature_vector = np.concatenate([hsv_hist, lbp_hist]).reshape(1, -1)
    
    dominant_colors = extract_dominant_colors(img_resized)
    
    prediction = rf_model.predict(feature_vector)[0]
    probabilities = rf_model.predict_proba(feature_vector)[0]
    
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probabilities))
    
    prob_dict = {label_encoder.classes_[i]: float(probabilities[i]) 
                 for i in range(len(probabilities))}
    
    return predicted_class, confidence, prob_dict, dominant_colors


def predict_cnn(image_bytes):
    """Predict mood using CNN model"""
    if cnn_model is None or cnn_class_names is None:
        raise ValueError("CNN model not loaded")
    
    img_resized, _, _ = load_image_from_bytes(image_bytes)
    
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    probabilities = cnn_model.predict(img_batch, verbose=0)[0]
    prediction = np.argmax(probabilities)
    
    predicted_class = cnn_class_names[prediction]
    confidence = float(probabilities[prediction])
    
    prob_dict = {cnn_class_names[i]: float(probabilities[i]) 
                 for i in range(len(probabilities))}
    
    return predicted_class, confidence, prob_dict


def ensemble_predict(image_bytes):
    """Ensemble prediction combining ML and CNN models"""
    ml_class, ml_conf, ml_probs, dominant_colors = predict_ml(image_bytes)
    cnn_class, cnn_conf, cnn_probs = predict_cnn(image_bytes)
    
    ml_weight = 0.4
    cnn_weight = 0.6
    
    all_classes = list(set(list(ml_probs.keys()) + list(cnn_probs.keys())))
    ensemble_probs = {}
    
    for cls in all_classes:
        ml_prob = ml_probs.get(cls, 0.0)
        cnn_prob = cnn_probs.get(cls, 0.0)
        ensemble_probs[cls] = ml_weight * ml_prob + cnn_weight * cnn_prob
    
    final_class = max(ensemble_probs, key=ensemble_probs.get)
    final_confidence = ensemble_probs[final_class]
    
    color_palette = [
        {"r": int(color[0]), "g": int(color[1]), "b": int(color[2])}
        for color in dominant_colors
    ]
    
    return {
        "ensemble": {
            "predicted_mood": final_class,
            "confidence": round(float(final_confidence), 4),
            "probabilities": {k: round(v, 4) for k, v in ensemble_probs.items()}
        },
        "ml_model": {
            "predicted_mood": ml_class,
            "confidence": round(ml_conf, 4),
            "probabilities": {k: round(v, 4) for k, v in ml_probs.items()}
        },
        "cnn_model": {
            "predicted_mood": cnn_class,
            "confidence": round(cnn_conf, 4),
            "probabilities": {k: round(v, 4) for k, v in cnn_probs.items()}
        },
        "dominant_colors": color_palette
    }


# Professional HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Mood Classifier - AI-Powered Emotion Recognition</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container-main {
            width: 100%;
            max-width: 900px;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
            animation: fadeInDown 0.8s ease-out;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.95;
            font-weight: 300;
        }

        .card-main {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            padding: 40px;
            backdrop-filter: blur(10px);
            animation: fadeInUp 0.8s ease-out;
        }

        .upload-section {
            margin-bottom: 30px;
        }

        .upload-box {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #f5f7ff 0%, #f0f4ff 100%);
        }

        .upload-box:hover {
            border-color: #764ba2;
            background: linear-gradient(135deg, #f0f4ff 0%, #e8ecff 100%);
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.1);
        }

        .upload-box i {
            font-size: 3rem;
            color: #667eea;
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }

        .upload-box:hover i {
            color: #764ba2;
            transform: scale(1.1);
        }

        .upload-box p {
            margin: 0;
            font-size: 1rem;
            color: #555;
            font-weight: 500;
        }

        .upload-box small {
            color: #999;
            display: block;
            margin-top: 10px;
        }

        #imageInput {
            display: none;
        }

        .btn-predict {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            padding: 15px 40px;
            font-size: 1.05rem;
            font-weight: 600;
            border-radius: 10px;
            width: 100%;
            transition: all 0.3s ease;
            cursor: pointer;
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }

        .btn-predict:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
            color: white;
        }

        .btn-predict:active {
            transform: translateY(-1px);
        }

        .btn-predict:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .result-section {
            margin-top: 30px;
            display: none;
            animation: fadeIn 0.6s ease-out;
        }

        .mood-display {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        }

        .mood-display h2 {
            font-size: 2rem;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .confidence-bar {
            background: rgba(255, 255, 255, 0.2);
            height: 8px;
            border-radius: 10px;
            margin: 15px 0;
            overflow: hidden;
        }

        .confidence-fill {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            height: 100%;
            border-radius: 10px;
            transition: width 0.8s ease;
        }

        .confidence-text {
            font-size: 0.95rem;
            opacity: 0.9;
            margin-top: 10px;
        }

        .probabilities-section {
            margin-bottom: 20px;
        }

        .probabilities-section h3 {
            font-size: 1.1rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
        }

        .probability-item {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }

        .probability-label {
            min-width: 120px;
            font-weight: 500;
            color: #555;
            font-size: 0.95rem;
        }

        .probability-bar {
            flex: 1;
            background: #e8ecff;
            height: 6px;
            border-radius: 5px;
            margin: 0 15px;
            overflow: hidden;
        }

        .probability-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            border-radius: 5px;
            transition: width 0.6s ease;
        }

        .probability-value {
            min-width: 50px;
            text-align: right;
            font-weight: 600;
            color: #667eea;
            font-size: 0.9rem;
        }

        .colors-section h3 {
            font-size: 1.1rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 15px;
        }

        .color-palette {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 15px;
        }

        .color-box {
            border-radius: 10px;
            height: 100px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease;
            cursor: pointer;
            border: 3px solid transparent;
        }

        .color-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.25);
            border-color: #667eea;
        }

        .models-section {
            margin-top: 25px;
            padding-top: 25px;
            border-top: 2px solid #e8ecff;
        }

        .model-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        .model-card {
            background: #f8faff;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            transition: all 0.3s ease;
        }

        .model-card:hover {
            background: #f0f4ff;
            transform: translateX(5px);
        }

        .model-card h4 {
            font-size: 0.9rem;
            font-weight: 600;
            color: #667eea;
            margin-bottom: 8px;
        }

        .model-card p {
            font-size: 0.85rem;
            color: #666;
            margin: 0;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .error-alert {
            background: #fff5f5;
            border: 2px solid #fc8181;
            color: #c53030;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            display: none;
        }

        .preview-image {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin-top: 15px;
            display: none;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        @media (max-width: 768px) {
            .card-main {
                padding: 25px;
            }

            .header h1 {
                font-size: 1.8rem;
            }

            .mood-display h2 {
                font-size: 1.5rem;
            }

            .color-palette {
                grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="container-main">
        <div class="header">
            <h1><i class="fas fa-brain"></i> Image Mood Classifier</h1>
            <p>AI-Powered Emotional Tone Recognition</p>
        </div>

        <div class="card-main">
            <!-- Upload Section -->
            <div class="upload-section">
                <div class="upload-box" onclick="document.getElementById('imageInput').click()">
                    <i class="fas fa-cloud-upload-alt"></i>
                    <p>Click to upload or drag and drop</p>
                    <small>Supported formats: JPG, PNG, BMP (Max 16MB)</small>
                </div>
                <input type="file" id="imageInput" accept="image/*">
                <img id="previewImage" class="preview-image" alt="Preview">
            </div>

            <!-- Upload Button -->
            <button class="btn-predict" id="predictBtn" onclick="predictMood()">
                <i class="fas fa-magic"></i> Analyze Image
            </button>

            <!-- Loading Indicator -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing image...</p>
            </div>

            <!-- Error Alert -->
            <div class="error-alert" id="errorAlert"></div>

            <!-- Results Section -->
            <div class="result-section" id="resultSection">
                <!-- Main Prediction -->
                <div class="mood-display">
                    <h2 id="moodEmoji"></h2>
                    <div id="moodName" style="font-size: 1.5rem; margin-bottom: 15px;"></div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="confidenceFill"></div>
                    </div>
                    <div class="confidence-text">
                        Confidence: <strong id="confidenceValue">0%</strong>
                    </div>
                </div>

                <!-- Probabilities -->
                <div class="probabilities-section">
                    <h3>Mood Probabilities</h3>
                    <div id="probabilitiesContainer"></div>
                </div>

                <!-- Dominant Colors -->
                <div class="colors-section">
                    <h3>Dominant Colors</h3>
                    <div class="color-palette" id="colorPalette"></div>
                </div>

                <!-- Model Details -->
                <div class="models-section">
                    <h3 style="font-size: 1.1rem; font-weight: 600; color: #333; margin-bottom: 15px;">
                        Model Analysis
                    </h3>
                    <div class="model-info" id="modelInfo"></div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const moodEmojis = {
            'Calm': '😌',
            'Energetic': '⚡',
            'Warm': '🔥',
            'Dark': '🌑',
            'Soft': '☁️'
        };

        // Drag and drop
        const uploadBox = document.querySelector('.upload-box');
        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = '#764ba2';
            uploadBox.style.background = 'linear-gradient(135deg, #e8ecff 0%, #dfe8ff 100%)';
        });

        uploadBox.addEventListener('dragleave', () => {
            uploadBox.style.borderColor = '#667eea';
            uploadBox.style.background = 'linear-gradient(135deg, #f5f7ff 0%, #f0f4ff 100%)';
        });

        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('imageInput').files = files;
                handleImageSelect();
            }
        });

        // Image preview
        document.getElementById('imageInput').addEventListener('change', handleImageSelect);

        function handleImageSelect() {
            const file = document.getElementById('imageInput').files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('previewImage').src = e.target.result;
                    document.getElementById('previewImage').style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        }

        function predictMood() {
            const fileInput = document.getElementById('imageInput');
            if (!fileInput.files.length) {
                showError('Please select an image first');
                return;
            }

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultSection').style.display = 'none';
            document.getElementById('errorAlert').style.display = 'none';

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                showError('Error: ' + error.message);
            })
            .finally(() => {
                document.getElementById('loading').style.display = 'none';
            });
        }

        function displayResults(data) {
            const ensemble = data.ensemble;
            const mood = ensemble.predicted_mood;
            const confidence = ensemble.confidence;

            // Main mood display
            document.getElementById('moodEmoji').textContent = moodEmojis[mood] || '🎨';
            document.getElementById('moodName').textContent = mood;
            document.getElementById('confidenceValue').textContent = (confidence * 100).toFixed(1) + '%';
            document.getElementById('confidenceFill').style.width = (confidence * 100) + '%';

            // Probabilities
            let probabilitiesHTML = '';
            Object.entries(ensemble.probabilities).forEach(([mood, prob]) => {
                probabilitiesHTML += `
                    <div class="probability-item">
                        <div class="probability-label">${mood}</div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${prob * 100}%"></div>
                        </div>
                        <div class="probability-value">${(prob * 100).toFixed(1)}%</div>
                    </div>
                `;
            });
            document.getElementById('probabilitiesContainer').innerHTML = probabilitiesHTML;

            // Colors
            let colorsHTML = '';
            data.dominant_colors.forEach(color => {
                colorsHTML += `<div class="color-box" style="background-color: rgb(${color.r}, ${color.g}, ${color.b});" title="RGB(${color.r}, ${color.g}, ${color.b})"></div>`;
            });
            document.getElementById('colorPalette').innerHTML = colorsHTML;

            // Model info
            const mlData = data.ml_model;
            const cnnData = data.cnn_model;
            const modelHTML = `
                <div class="model-card">
                    <h4>ML Model (Random Forest)</h4>
                    <p><strong>${mlData.predicted_mood}</strong></p>
                    <p>${(mlData.confidence * 100).toFixed(1)}% confidence</p>
                </div>
                <div class="model-card">
                    <h4>CNN Model (MobileNetV2)</h4>
                    <p><strong>${cnnData.predicted_mood}</strong></p>
                    <p>${(cnnData.confidence * 100).toFixed(1)}% confidence</p>
                </div>
                <div class="model-card">
                    <h4>Ensemble Prediction</h4>
                    <p><strong>${ensemble.predicted_mood}</strong></p>
                    <p>${(ensemble.confidence * 100).toFixed(1)}% confidence</p>
                </div>
            `;
            document.getElementById('modelInfo').innerHTML = modelHTML;

            document.getElementById('resultSection').style.display = 'block';
        }

        function showError(message) {
            const errorAlert = document.getElementById('errorAlert');
            errorAlert.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
            errorAlert.style.display = 'block';
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Render upload page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    """Predict mood from uploaded image"""
    if rf_model is None or cnn_model is None:
        return jsonify({
            'error': 'Models not loaded. Please train models first.'
        }), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp'}), 400
    
    try:
        image_bytes = file.read()
        result = ensemble_predict(image_bytes)
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    models_loaded = (rf_model is not None and cnn_model is not None)
    
    return jsonify({
        'status': 'healthy' if models_loaded else 'models_not_loaded',
        'models_loaded': models_loaded
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
