import torch.nn as nn

class HalalHaramClassifier(nn.Module):
    def __init__(self):
        super(HalalHaramClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 128 * 128, 2)  # Adjust based on input size
        )

    def forward(self, x):
        return self.model(x)
