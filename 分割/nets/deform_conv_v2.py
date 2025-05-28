import torch
import torch.nn as nn
import torchvision.ops as ops

class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1):
        super(DeformConv2d, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.regular_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        offset = self.offset_conv(x)
        x = ops.deform_conv2d(x, offset, self.regular_conv.weight, bias=self.bias, stride=self.regular_conv.stride, padding=self.regular_conv.padding, dilation=self.regular_conv.dilation)
        return x
