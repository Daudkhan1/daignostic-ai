from torch import nn
from torchvision.models import efficientnet_b0


class MitoticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.en = efficientnet_b0()
        in_features_en = self.en.classifier[-1].in_features
        self.en.classifier = nn.Identity()
        self.fc = nn.Linear(in_features_en, 1000)
        self.fc1 = nn.Linear(1000, 3)
        self.fc2 = nn.Linear(1000, 4)

    def forward(self, x):
        x = x.float()
        x = self.en(x)
        x = self.fc(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2
