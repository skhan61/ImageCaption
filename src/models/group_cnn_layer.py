import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 group, learnable_rotations=False):
        super(GroupConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, padding=kernel_size // 2)
        self.group = group
        self.learnable_rotations = learnable_rotations
        self.rotation_matrices = self._precompute_rotation_matrices(group)

        if self.learnable_rotations:
            # Replace fixed rotation angles with learnable parameters
            self.rotation_angles = nn.Parameter(torch.tensor(
                [g * math.pi / 180 for g in group], dtype=torch.float32))

    def _precompute_rotation_matrices(self, group):
        matrices = []
        for group_element in group:
            radians = group_element * math.pi / 180
            rotation_matrix = self._create_rotation_matrix(radians)
            matrices.append(rotation_matrix)
        # Stack and convert to a parameter
        return nn.Parameter(torch.stack(matrices), requires_grad=False)

    @staticmethod
    def _create_rotation_matrix(radians):
        # Check if 'radians' is already a tensor. If not, convert it to a tensor.
        if not isinstance(radians, torch.Tensor):
            radians_tensor = torch.tensor(radians, dtype=torch.float32)
        else:
            # If it's already a tensor, use it directly.
            radians_tensor = radians

        return torch.tensor([
            [torch.cos(radians_tensor), -torch.sin(radians_tensor), 0],
            [torch.sin(radians_tensor), torch.cos(radians_tensor), 0]
        ], dtype=torch.float32)

    def transform_input(self, x, rotation_matrix):
        batch_size = x.size(0)
        # Repeat rotation matrix for each batch element and reshape
        rotation_matrix = rotation_matrix.repeat(
            batch_size, 1, 1).view(batch_size, 2, 3)
        grid = F.affine_grid(rotation_matrix, x.size(), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True)

    def forward(self, x):
        outputs = []

        if self.learnable_rotations:
            # Update rotation matrices if they are learnable
            updated_matrices = [self._create_rotation_matrix(
                angle).to(x.device) for angle in self.rotation_angles]
        else:
            updated_matrices = self.rotation_matrices

        for rotation_matrix in updated_matrices:
            transformed_x = self.transform_input(x, rotation_matrix)
            conv_output = self.conv(transformed_x)
            outputs.append(conv_output)

        # Concatenate along a new dimension for memory efficiency
        out = torch.cat(outputs, dim=1)
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GroupConvLayerWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 group, learnable_rotations=False):
        super(GroupConvLayerWithAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, padding=kernel_size // 2)
        self.group = group
        self.learnable_rotations = learnable_rotations
        self.rotation_matrices = self._precompute_rotation_matrices(group)
        self.attention_blocks = nn.ModuleList(
            [SqueezeExcitation(out_channels) for _ in group])

        if self.learnable_rotations:
            self.rotation_angles = nn.Parameter(torch.tensor(
                [g * math.pi / 180 for g in group], dtype=torch.float32))

    def _precompute_rotation_matrices(self, group):
        matrices = []
        for group_element in group:
            radians = group_element * math.pi / 180
            rotation_matrix = self._create_rotation_matrix(radians)
            matrices.append(rotation_matrix)
        return nn.Parameter(torch.stack(matrices), requires_grad=False)

    @staticmethod
    def _create_rotation_matrix(radians):
        if not isinstance(radians, torch.Tensor):
            radians_tensor = torch.tensor(radians, dtype=torch.float32)
        else:
            radians_tensor = radians
        return torch.tensor([[torch.cos(radians_tensor), -torch.sin(radians_tensor), 0],
                             [torch.sin(radians_tensor), torch.cos(radians_tensor), 0]],
                            dtype=torch.float32)

    def transform_input(self, x, rotation_matrix):
        batch_size = x.size(0)
        rotation_matrix = rotation_matrix.repeat(
            batch_size, 1, 1).view(batch_size, 2, 3)
        grid = F.affine_grid(rotation_matrix, x.size(), align_corners=True)
        return F.grid_sample(x, grid, align_corners=True)

    def forward(self, x):
        outputs = []
        updated_matrices = self.rotation_matrices if not self.learnable_rotations else \
            [self._create_rotation_matrix(angle).to(
                x.device) for angle in self.rotation_angles]

        for i, rotation_matrix in enumerate(updated_matrices):
            transformed_x = self.transform_input(x, rotation_matrix)
            conv_output = self.conv(transformed_x)
            attention_output = self.attention_blocks[i](conv_output)
            outputs.append(attention_output)

        out = torch.cat(outputs, dim=1)
        return out


# import torch
# from models.group_cnn_layer import GroupConvLayerWithAttention

# # Create a dummy input
# dummy_input = torch.randn(1, 3, 64, 64)  # Batch size 1

# # Initialize the GroupConvLayerWithAttention
# group_elements = [0, 90, 180, 270]  # Rotation angles in degrees
# group_conv_layer = GroupConvLayerWithAttention(in_channels=3, out_channels=6,
#                                                kernel_size=3, group=group_elements,
#                                                learnable_rotations=True,
#                                                # Add any additional parameters for attention here
#                                                )

# # Forward pass
# output = group_conv_layer(dummy_input)

# # Check tensor sizes
# assert output.size(0) == dummy_input.size(0), "Mismatch in batch size"
# assert output.size(1) != 0, "Output channels are zero"
# assert output.size(2) == 64 and output.size(3) == 64, "Mismatch in spatial dimensions"

# # Print the shape of the output tensor
# print("Output shape:", output.shape)


# from models.group_cnn_layer import GroupConvLayer, GroupConvLayerWithAttention

# # Create a dummy input - a single image with 3 channels (RGB) and 64x64 pixels
# dummy_input = torch.randn(1, 3, 64, 64)  # Batch size 1


# # Initialize the GroupConvLayer
# group_elements = [0, 90, 180, 270]  # Example group elements (rotation angles in degrees)
# group_conv_layer = GroupConvLayer(in_channels=3, out_channels=6, \
#     kernel_size=3, group=group_elements, learnable_rotations=True)

# # Forward pass
# output = group_conv_layer(dummy_input)

# # Print the shape of the output tensor
# print("Output shape:", output.shape)
