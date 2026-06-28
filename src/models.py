import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import models

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



class MobileNetV2Custom(nn.Module):

    def __init__(self,num_classes = 13,pretrained=True,freeze_base=True):

        super().__init__()

        # the mobilenet backbone is {all layers except the final MLP classifier}
        # 1280 -> num_classes
        self.backbone = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)

        # change final classifier matrix to match our 13 classes
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)


        # Unfreeze only classifier
        if freeze_base :
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

            # Freeze only features
            for param in self.backbone.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Expected input shape from your dataset: [Batch, 1, 64, Time]
        
        # The Input Hack: Duplicate the single grayscale channel 3 times.
        # .repeat(batch_multiplier, channel_multiplier, height_multiplier, width_multiplier)
        x = x.repeat(1, 3, 1, 1) 
        # Shape is now mathematically identical to an RGB image: [Batch, 3, 64, Time]
        
        # Pass through the modified network
        logits = self.backbone(x)
        return logits 
                