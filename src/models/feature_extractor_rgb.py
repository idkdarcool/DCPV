import torch
import torch.nn as nn
import torch.nn.functional as F

class CifarFeatureExtractor(nn.Module):
    """
    Standard CNN for 32x32 RGB images (CIFAR-10, SVHN).
    """
    def __init__(self, output_dim=128):
        super(CifarFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        return self.extract_features(x)

    def extract_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CifarWrapper(nn.Module):
    """Adds a classification head for pre-training."""
    def __init__(self, feature_extractor, num_classes=10):
        super(CifarWrapper, self).__init__()
        self.fe = feature_extractor
        self.head = nn.Linear(128, num_classes)
        
    def forward(self, x):
        feats = self.fe.extract_features(x)
        return self.head(feats)
    
    def extract_features(self, x):
        return self.fe.extract_features(x)
