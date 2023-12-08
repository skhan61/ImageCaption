import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet152_Weights


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Load pre-trained ResNet-152 using the new API
        weights = ResNet152_Weights.IMAGENET1K_V1  # or use ResNet152_Weights.DEFAULT
        resnet = models.resnet152(weights=weights)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        return self.bn(features)



# # Create a dummy input tensor of size [batch_size, channels, width, height]
# dummy_input = torch.randn(32, 3, 224, 224)  # Batch size of 32, 3 color channels, 224x224 image size

# # Instantiate the EncoderCNN
# embed_size = 256  # Example embedding size
# encoder = EncoderCNN(embed_size)

# # Pass the dummy input through the encoder
# output_features = encoder(dummy_input)

# print("Output features size:", output_features.size())  # Should be [32, embed_size]