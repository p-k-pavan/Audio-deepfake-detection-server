from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from tensorflow import keras
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models and preprocessors
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
cnn_model = keras.models.load_model('CNN/cnn-model.keras')
dnn_model = keras.models.load_model('DNN/DNN-Model.keras')

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfccs)
        delta2_mfcc = librosa.feature.delta(mfccs, order=2)
        mfcc_features = np.hstack((
            np.mean(mfccs, axis=1),
            np.mean(delta_mfcc, axis=1),
            np.mean(delta2_mfcc, axis=1)
        ))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_features = np.mean(chroma, axis=1)

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_features = np.mean(mel, axis=1)

        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_features = np.mean(spectral_contrast, axis=1)

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_features = np.mean(tonnetz, axis=1)

        combined = np.hstack((
            mfcc_features,
            chroma_features,
            mel_features,
            contrast_features,
            tonnetz_features
        ))

        return combined
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros(192)

def predict_audio(file_path):
    try:
        features = extract_features(file_path).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # CNN Prediction (Primary)
        cnn_input = features_scaled.reshape(1, features_scaled.shape[1], 1, 1)
        cnn_pred = cnn_model.predict(cnn_input, verbose=0)
        cnn_label_num = np.argmax(cnn_pred)
        cnn_label = label_encoder.inverse_transform([cnn_label_num])[0]

        # DNN Prediction (Secondary)
        dnn_pred = dnn_model.predict(features_scaled, verbose=0)
        dnn_label_num = np.argmax(dnn_pred)
        dnn_label = label_encoder.inverse_transform([dnn_label_num])[0]

        # Convert labels to human-readable format
        label_map = {0: "Fake", 1: "Real"}  # Adjust based on your label encoder
        
        return {
            "CNN": label_map.get(int(cnn_label_num), str(cnn_label)),
            "DNN": label_map.get(int(dnn_label_num), str(dnn_label))
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

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        result = predict_audio(file_path)
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Clean up the uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return jsonify({'result': result})
    except Exception as e:
        print("Error during prediction:", e)
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
