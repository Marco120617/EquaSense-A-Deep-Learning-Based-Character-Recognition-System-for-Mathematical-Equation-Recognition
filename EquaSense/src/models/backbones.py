import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(DenseNetFeatureExtractor, self).__init__()
        # Using a lightweight pre-trained DenseNet121 configuration
        # Modifying the input layer to handle 1-channel grayscale images
        self.backbone = models.densenet121(pretrained=False)
        self.backbone.features.conv0 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # Extract features without passing through the final classification head
        self.feature_extractor = self.backbone.features

    def forward(self, x):
        # Input size expected: (batch_size, 1, height, width)
        features = self.feature_extractor(x)
        return features