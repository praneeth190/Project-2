import torch
import librosa
import numpy as np
from model1 import EmotionANN
import json
import sys
import os
import pickle

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    return np.hstack([mfcc, chroma, spectral_contrast, tonnetz, rms])


def predict_emotion(audio_path, model, label_encoder, scaler):
    """
    Preprocesses an audio file, loads the model, and predicts the emotion.
    Returns the predicted emotion label.
    """
    # Load and preprocess audio
    features = extract_features(audio_path)
    features = scaler.transform([features])  # Apply the same scaler
    features_tensor = torch.tensor(features, dtype=torch.float32)
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        output = model(features_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    # Map label to emotion
    predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_emotion


if __name__ == '__main__':
    # Example Usage (Replace placeholders)
    MODEL_PATH = os.path.join("models", "emotion_model_pran.pth")  # Replace with your model path
    LABEL_ENCODER_PATH = os.path.join("models","label_encoder.pkl")
    SCALER_PATH = os.path.join("models","scaler.pkl")
    # Use CUDA if available, else CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    input_size = 39 # Update with the correct input size (Check the training)
    num_classes = 5 # Update with correct num_classes (Check the training)
    trained_model = EmotionANN.load_model(MODEL_PATH, input_size, num_classes)

    with open(LABEL_ENCODER_PATH,'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(SCALER_PATH,'rb') as f:
        scaler = pickle.load(f)
        
    if len(sys.argv) != 2:
      print("Usage: python predict.py <audio_file_path>")
      sys.exit(1)

    audio_path = sys.argv[1]  # Get audio path from command line argument

    predicted_emotion = predict_emotion(audio_path, trained_model, label_encoder, scaler)

    # Output JSON string
    result = json.dumps({"emotion": predicted_emotion})
    print(result)