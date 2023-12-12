from torchvision import models
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet152_Weights

from models.group_cnn_layer import GroupConvLayer


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, group_elements=None):
        super(EncoderCNN, self).__init__()
        # Load pre-trained ResNet-152 using the new API
        weights = ResNet152_Weights.IMAGENET1K_V1  # or use ResNet152_Weights.DEFAULT
        resnet = models.resnet152(weights=weights)
        modules = list(resnet.children())[:-1]

        # Freeze the ResNet layers to prevent backpropagation (if not fine-tuning)
        for param in resnet.parameters():
            param.requires_grad = False

        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        return self.bn(features)


class ModifiedEncoderCNN(nn.Module):
    def __init__(self, embed_size, group_conv_layer):
        super(ModifiedEncoderCNN, self).__init__()
        self.group_conv = group_conv_layer  # Use the provided GroupConvLayer instance

        # Load pre-trained ResNet-152
        weights = ResNet152_Weights.IMAGENET1K_V1
        resnet = models.resnet152(weights=weights)
        # Exclude the first /conv layer and the last fully connected layer
        modules = list(resnet.children())[1:-1]

        # Freeze the ResNet layers to prevent backpropagation (if not fine-tuning)
        for param in resnet.parameters():
            param.requires_grad = False

        self.resnet = nn.Sequential(*modules)

        # Adjust the number of output channels to match ResNet's input channels
        self.adjust_channels = nn.Conv2d(1, 64, kernel_size=1)

        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        group_features = self.group_conv(images)
        features = torch.mean(group_features, dim=1, keepdim=True)

        # Adjust the channel size to match ResNet's expected input
        features = self.adjust_channels(features)

        with torch.no_grad():
            features = self.resnet(features)

        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        return self.bn(features)


# class ModifiedEncoderCNN(nn.Module):
#     def __init__(self, embed_size, group_elements):
#         super(ModifiedEncoderCNN, self).__init__()
#         self.group_conv = GroupConvLayer(
#             in_channels=3, out_channels=64,
#             kernel_size=7, group=group_elements,
#             learnable_rotations=True)

#         # Load pre-trained ResNet-152
#         weights = ResNet152_Weights.IMAGENET1K_V1
#         resnet = models.resnet152(weights=weights)
#         # Exclude the first /conv layer and the last fully connected layer
#         modules = list(resnet.children())[1:-1]

#         # Freeze the ResNet layers to prevent backpropagation (if not fine-tuning)
#         for param in resnet.parameters():
#             param.requires_grad = False

#         self.resnet = nn.Sequential(*modules)

#         # Adjust the number of output channels to match ResNet's input channels
#         self.adjust_channels = nn.Conv2d(1, 64, kernel_size=1)

#         self.linear = nn.Linear(resnet.fc.in_features, embed_size)
#         self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

#     def forward(self, images):
#         group_features = self.group_conv(images)
#         features = torch.mean(group_features, dim=1, keepdim=True)

#         # Adjust the channel size to match ResNet's expected input
#         features = self.adjust_channels(features)

#         with torch.no_grad():
#             features = self.resnet(features)

#         features = features.reshape(features.size(0), -1)
#         features = self.linear(features)
#         return self.bn(features)

    # def forward(self, images):
    #     group_features = self.group_conv(images)
    #     features = torch.mean(group_features, dim=1)  # Aggregate features

    #     with torch.no_grad():  # Prevent gradient tracking
    #         features = self.resnet(features)

    #     features = features.reshape(features.size(0), -1)
    #     features = self.linear(features)
    #     return self.bn(features)


# class ModifiedEncoderCNN(nn.Module):
#     def __init__(self, embed_size, group_elements):
#         super(ModifiedEncoderCNN, self).__init__()

#         # Define GroupConvLayer with appropriate channel sizes and group elements
#         self.group_conv = GroupConvLayer(
#             in_channels=3, out_channels=64, kernel_size=7, group=group_elements)

#         # Load pre-trained ResNet-152, excluding the first conv layer and fully connected layers
#         weights = ResNet152_Weights.IMAGENET1K_V1
#         resnet = models.resnet152(weights=weights)

#         # Exclude the first conv layer and fully connected layers
#         modules = list(resnet.children())[1:-1]
#         self.resnet = nn.Sequential(*modules)

#         # Redefine the linear layer to match the output of the modified ResNet architecture
#         self.linear = nn.Linear(resnet.fc.in_features, embed_size)
#         self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

#     def forward(self, images):
#         # First, pass images through the GroupConvLayer
#         group_features = self.group_conv(images)

#         # Process the output of GroupConvLayer as needed (e.g., reshape, aggregate)
#         # For simplicity, let's assume we aggregate the group features by averaging
#         features = torch.mean(group_features, dim=1)

#         # Continue with the modified ResNet layers
#         with torch.no_grad():
#             features = self.resnet(features)
#         features = features.reshape(features.size(0), -1)
#         features = self.linear(features)
#         return self.bn(features)


# # Define group elements (e.g., rotation angles)
# group_elements = [0, 90, 180, 270]

# # Create a dummy input tensor of size [batch_size, channels, width, height]
# dummy_input = torch.randn(32, 3, 224, 224)  # Batch size of 32, 3 color channels, 224x224 image size

# # Instantiate the ModifiedEncoderCNN with group elements
# embed_size = 256  # Example embedding size
# encoder = ModifiedEncoderCNN(embed_size, group_elements)

# # Pass the dummy input through the encoder
# output_features = encoder(dummy_input)

# # Print the shape of the output tensor
# print("Output features size:", output_features.size())  # Should be [32, embed_size]

# # Define group elements (e.g., rotation angles)
# group_elements = [0, 90, 180, 270]

# # Create a dummy input tensor of size [batch_size, channels, width, height]
# dummy_input = torch.randn(32, 3, 224, 224)  # Batch size of 32, 3 color channels, 224x224 image size

# # Instantiate the ModifiedEncoderCNN with group elements
# embed_size = 256  # Example embedding size
# encoder = EncoderCNN(embed_size)

# # Pass the dummy input through the encoder
# output_features = encoder(dummy_input)

# # Print the shape of the output tensor
# print("Output features size:", output_features.size())  # Should be [32, embed_size]
