import torch.nn as nn

from collections import OrderedDict


class VGGBlock(nn.Sequential):
    '''SuperPoint backbone 的基本块：Conv -> ReLU(可选) -> BatchNorm。

    子模块名固定为 conv / activation / bn，与 rpautrat 转换出的 PyTorch 权重
    （superpoint_v6_from_tf.pth）键名一致，便于直接 load_state_dict。
    '''

    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1) // 2
        conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        bn = nn.BatchNorm2d(c_out, eps=0.001)
        super().__init__(OrderedDict([
            ('conv', conv),
            ('activation', activation),
            ('bn', bn),
        ]))
