import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import joblib
import tensorflow as tf
from tensorflow import keras
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}
FEATURE_SIZE = 192  

try:
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))
    cnn_model = keras.models.load_model(os.path.join(BASE_DIR, 'CNN/cnn-model.keras'))
    dnn_model = keras.models.load_model(os.path.join(BASE_DIR, 'DNN/DNN-Model.keras'))
except Exception as e:
    print(f"Error loading models: {e}")
    raise RuntimeError("Failed to load required model files")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, duration=5) 
        

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfccs)
        delta2_mfcc = librosa.feature.delta(mfccs, order=2)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        mel = librosa.feature.melspectrogram(y=y, sr=sr)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        features = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(delta_mfcc, axis=1),
            np.mean(delta2_mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(spectral_contrast, axis=1),
            np.mean(tonnetz, axis=1)
        ])

        if len(features) > FEATURE_SIZE:
            features = features[:FEATURE_SIZE]
        elif len(features) < FEATURE_SIZE:
            features = np.pad(features, (0, FEATURE_SIZE - len(features)))
            
        return features
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros(FEATURE_SIZE)

def predict_audio(file_path):
    try:
        features = extract_features(file_path).reshape(1, -1)
        features_scaled = scaler.transform(features)

        cnn_input = features_scaled.reshape(1, features_scaled.shape[1], 1, 1)
        cnn_pred = cnn_model.predict(cnn_input, verbose=0)
        cnn_label_num = np.argmax(cnn_pred)

        dnn_pred = dnn_model.predict(features_scaled, verbose=0)
        dnn_label_num = np.argmax(dnn_pred)

        labels = label_encoder.inverse_transform([cnn_label_num, dnn_label_num])
        
        return {
            "CNN": "Real" if labels[0] == 1 else "Fake",
            "DNN": "Real" if labels[1] == 1 else "Fake",
            "confidence": {
                "CNN": float(np.max(cnn_pred)),
                "DNN": float(np.max(dnn_pred))
            }
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
        
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        result = predict_audio(file_path)
        if not result:
            return jsonify({'error': 'Prediction failed'}), 500
            
        return jsonify(result)
        
    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': True})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
