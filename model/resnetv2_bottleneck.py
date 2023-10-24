import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from functools import partial


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
        self,
        in_channel,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=False,
        eps=1e-6,
    ):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channel,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1),
            None,
            None,
            training=True,
            momentum=0.0,
            eps=self.eps,
        ).reshape_as(self.weight)
        x = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class ResnetV2Bottleneck(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs=None,
        bottle_ratio=0.25,
        stride=1,
        dilation=1,
        first_dilation=None,
        groups=1,
        act_layer=None,
        conv_layer=None,
        norm_layer=None,
        drop_path_rate=0.0,
    ):
        super().__init__()
        first_dilation = first_dilation or dilation
        act_layer = act_layer or nn.ReLU
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(nn.GroupNorm)
        out_chs = out_chs or in_chs
        mid_chs = make_divisible(out_chs * bottle_ratio)

        self.downsample = conv_layer(in_chs, out_chs, 1, stride=stride, bias=False) if stride > 1 else None

        self.preact = norm_layer(in_chs, in_chs)
        self.preact_act = act_layer(inplace=True)
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.norm1 = norm_layer(4, mid_chs)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm2 = norm_layer(4, mid_chs)
        self.conv3 = conv_layer(mid_chs, out_chs, 1)
        self.norm3 = norm_layer(4, out_chs)
        self.drop_path = StochasticDepth(drop_path_rate, "row") if drop_path_rate > 0 else nn.Identity()
        self.act3 = act_layer(inplace=True)

    def zero_init_last(self):
        if getattr(self.norm3, "weight", None) is not None:
            nn.init.zeros_(self.norm3.weight)

    def forward(self, x):
        x = self.preact_act(self.preact(x))

        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        # residual
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.drop_path(x)
        x = self.act3(x + shortcut)
        return x