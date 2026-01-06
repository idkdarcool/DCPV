import torch
import torch.nn as nn
import torch.nn.functional as F

class MnistFeatureExtractor(nn.Module):
    """
    CNN for MNIST feature extraction.
    Architecture: Conv -> Pool -> Conv -> Pool -> FC -> FC
    """
    def __init__(self, dropout_rate=0.5):
        super(MnistFeatureExtractor, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv_layer2 = nn.Conv2d(32, 64, kernel_size=5)
        self.drop_layer = nn.Dropout2d(p=dropout_rate)
        
        self.fc_layer1 = nn.Linear(1024, 128)
        self.fc_layer2 = nn.Linear(128, 10)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.extract_features(x)
        x = F.dropout(x, p=self.dropout_rate, training=True)
        x = self.fc_layer2(x)
        return F.log_softmax(x, dim=1)

    def extract_features(self, x):
        x = F.relu(F.max_pool2d(self.conv_layer1(x), 2))
        x = F.relu(F.max_pool2d(self.drop_layer(self.conv_layer2(x)), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc_layer1(x))
        return x
