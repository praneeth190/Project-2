import torch
import torch.nn as nn

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmotionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def load_model(model_path, device):
    """Loads your model with saved weights"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint['hidden_dim']
    output_dim = checkpoint['output_dim']
    model = EmotionClassifier(input_dim, hidden_dim, output_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set model to evaluation mode for inference
    emotion_labels = checkpoint['emotion_labels']
    return model, emotion_labels