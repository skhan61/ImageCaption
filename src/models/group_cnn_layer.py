import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, group):
        super(GroupConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, padding=kernel_size//2)
        self.group = group  # e.g., [0, 90, 180, 270] for rotations

    def transform_input(self, x, group_element):
        # Convert degrees to radians
        radians = group_element * math.pi / 180

        # Convert radians to a tensor
        radians = torch.tensor(radians, dtype=torch.float32)

        # Create rotation matrix
        rotation_matrix = torch.tensor([
            [torch.cos(radians), -torch.sin(radians), 0],
            [torch.sin(radians), torch.cos(radians), 0]
        ], dtype=torch.float32)

        # Repeat the rotation matrix for each image in the batch
        batch_size = x.size(0)
        rotation_matrix = rotation_matrix.repeat(batch_size, 1, 1)

        # Create the grid for rotation
        grid = F.affine_grid(rotation_matrix, x.size(), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True)

    def forward(self, x):
        outputs = []
        for group_element in self.group:
            transformed_x = self.transform_input(x, group_element)
            conv_output = self.conv(transformed_x)
            outputs.append(conv_output)
        out = torch.stack(outputs, dim=1)  # Stack along a new dimension
        return out


# from models.group_cnn_layer import GroupConvLayer

# # Create a dummy input - a single image with 3 channels (RGB) and 64x64 pixels
# dummy_input = torch.randn(1, 3, 64, 64)  # Batch size 1


# # Initialize the GroupConvLayer
# group_elements = [0, 90, 180, 270]  # Example group elements (rotation angles in degrees)
# group_conv_layer = GroupConvLayer(in_channels=3, out_channels=6, \
#     kernel_size=3, group=group_elements)

# # Forward pass
# output = group_conv_layer(dummy_input)

# # Print the shape of the output tensor
# print("Output shape:", output.shape)
