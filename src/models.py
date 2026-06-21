import torch
import torch.nn as nn

class KeywordSpottingCNN(nn.Module):
    def __init__(self, num_classes=13):
        super(KeywordSpottingCNN, self).__init__()
        
        # Block 1: 1 Channel (Mel-Spec) -> 16 Channels
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 16 Channels -> 32 Channels
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: 32 Channels -> 64 Channels
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling: mathematically averages any spatial dimension down to 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final linear classifier: 64 features -> 13 classes
        self.classifier = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # x is expected to have shape [Batch, 1, Mels, Time]
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # GAP collapses the spatial dimensions [Batch, 64, H, W] -> [Batch, 64, 1, 1]
        x = self.gap(x)
        
        # Remove the dummy 1x1 spatial dimensions to get a clean 1D vector: [Batch, 64]
        x = torch.flatten(x, 1)
        
        # Pass through the final classifier: [Batch, 13]
        logits = self.classifier(x)
        
        return logits
