import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedConv2d(nn.Conv2d):
    """
    Implémentation d'une convolution masquée
    """
    def __init__(self, in_channels, out_channels, kernel_size, mask_type='A',
                 stride=1, bias=True):
        if isinstance(kernel_size, tuple):
            kernel_size = max(kernel_size)
        padding = kernel_size // 2

        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=bias)
        mask = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        center = kernel_size // 2

        mask[:, :, center, center + (mask_type == 'B'):] = 0  # Masquer pixels actuels (A) ou futurs (B) horizontalement
        mask[:, :, center + 1:] = 0  # Masquer pixels futurs verticalement

        self.register_buffer('mask', mask)

    def forward(self, x):
        masked_weight = self.weight * self.mask
        return F.conv2d(x, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
