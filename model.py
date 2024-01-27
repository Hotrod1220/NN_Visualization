import torch
from torch import nn 

class Model(nn.Module):
    """
    Model used to classify MNIST digits.
    """
    def __init__(self):
        """
        Initializes the convolution neural network layers.
        """
        super().__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.output = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        """
        Callable function used to pass input through the neural network.

        Args:
            x: Tensor, input image.
        """
        x = self.convolutional(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x
