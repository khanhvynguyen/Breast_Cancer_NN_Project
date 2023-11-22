import torch
import torch.nn as nn


class MyCNN(nn.Module):  # Module: base class for all neural network modules
    def __init__(
        self,
        input_dim: int,
        num_kernel_conv1: int,
        num_kernel_conv2: int,
        num_kernel_conv3: int,
        n_classes: int,
        kernel_size: int = 3,
    ):
        super().__init__()  ## ke thua tu class cha

        self.conv_1 = nn.Conv2d(
            in_channels=3,  ### number of channels of original images
            out_channels=num_kernel_conv1,
            kernel_size=(kernel_size, kernel_size),
            stride=1,
            padding=1,
        )
        self.conv_2 = nn.Conv2d(
            in_channels=num_kernel_conv1,
            out_channels=num_kernel_conv2,
            kernel_size=(kernel_size, kernel_size),
            stride=1,
            padding=1,
        )
        self.conv_3 = nn.Conv2d(
            in_channels=num_kernel_conv2,
            out_channels=num_kernel_conv3,
            kernel_size=(kernel_size, kernel_size),
            stride=1,
            padding=1,
        )
        output_size = input_dim
        self.linear = nn.Linear(
            in_features=num_kernel_conv3 * output_size * output_size, out_features=n_classes
        )
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        ## x -> conv1 -> relu_1 -> conv2 -> relu_2 -> conv3 -> relu_3 ->(flatten) -> linear -> softmax
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.conv_3(x)
        x = self.relu_3(x)

        b = x.shape[0]
        x = x.reshape(b, -1)  ## flatten
        x = self.linear(x)

        ## return output
        return x
