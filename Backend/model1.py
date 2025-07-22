import torch
import torch.nn as nn

class EmotionANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EmotionANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    @classmethod
    def load_model(cls, path, input_size, num_classes):
        model = cls(input_size, num_classes)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()  # Set to evaluation mode
        return model

if __name__ == '__main__':
    # This part can be removed for your local prediction setup. 
    # It's only useful if you need to train the model using this file as a script.
    pass