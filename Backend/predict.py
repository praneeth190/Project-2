import torch
import librosa
import numpy as np
from model import load_model
import json
import sys
import os

# Load audio file
def load_audio(file_path, sr=16000):
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr

# Trim audio to exactly 8 seconds
def trim_to_8_seconds(audio, sr, max_duration=8):
    target_length = sr * max_duration
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        audio = np.pad(audio, (0, target_length - len(audio)))
    return audio


# Extract audio features
def extract_features(audio, sr, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    features = np.concatenate((mfcc, chroma, contrast, mel), axis=0)
    return features.T


def predict_emotion(audio_path, model, emotion_labels, device, sr=16000, max_duration=8):
    """
    Preprocesses an audio file, loads the model, and predicts the emotion.
    Returns the predicted emotion label.
    """
    # Load and preprocess audio
    audio, sr = load_audio(audio_path, sr=sr)
    audio = trim_to_8_seconds(audio, sr, max_duration)
    features = extract_features(audio, sr)

    # Convert to tensor and average across time steps
    input_tensor = torch.tensor(features.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
    
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    
    # Map label to emotion
    label_to_emotion = {v: k for k, v in emotion_labels.items()}
    predicted_emotion = label_to_emotion[predicted_label]
    return predicted_emotion


if __name__ == '__main__':
    # Example Usage (Replace placeholders)
    MODEL_PATH = os.path.join("models","emotion_model.pth") # Replace with your model path
    # Use CUDA if available, else CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_model, emotion_labels = load_model(MODEL_PATH, DEVICE)
    audio_path = sys.argv[1]  # Get audio path from command line argument

    predicted_emotion = predict_emotion(audio_path, trained_model, emotion_labels, DEVICE)

    # Output JSON string
    result = json.dumps({"emotion": predicted_emotion})
    print(result)