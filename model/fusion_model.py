# model/fusion_classifier.py
import torch.nn as nn

class FusionClassifier(nn.Module):
    def __init__(self, input_dim=6144, num_classes=27):
        super(FusionClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
