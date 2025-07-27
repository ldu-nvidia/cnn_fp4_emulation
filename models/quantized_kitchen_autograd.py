import torch
import torch.nn as nn
from typing import Optional

# Kitchen quantization autograd API
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent / "kitchen"))

from kitchen.quantization_autograd import (
    QParams,
    quantized_gemm,   # not used directly for conv but import for completeness
)
from kitchen.quantization_autograd import QuantizeOpInt8Int4MXFP4MXFP6MXFP8EmulationRefV2 as QuantOp
from kitchen.quantization import ScalingType, TensorType
from kitchen import utils as k_utils


class AutoQuantConv2d(nn.Module):
    """Conv2d that fake-quantises weight and input using Kitchen autograd NVFP4."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)

        # shared quantizer + params (NVFP4 ~ FP4 E2M1, 1x16 tiling)
        self.qop = QuantOp()
        self.qparams_x = QParams(
            scaling_type=ScalingType.PER_1D_BLOCK,
            quant_dtype=k_utils.Fp4Formats.E2M1,
            pow_2_scales=True,
            quant_tile_shape=(1, 16),
            tensor_type=TensorType.X,
        )
        self.qparams_w = QParams(
            scaling_type=ScalingType.PER_1D_BLOCK,
            quant_dtype=k_utils.Fp4Formats.E2M1,
            pow_2_scales=True,
            quant_tile_shape=(1, 16),
            tensor_type=TensorType.W,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantise activation and weight; returns FP tensors with quant error + STE
        qx = self.qop.quantize(x, self.qparams_x, return_identity=True, return_transpose=False).data
        qw = self.qop.quantize(self.conv.weight, self.qparams_w, return_identity=True, return_transpose=False).data
        return nn.functional.conv2d(qx, qw, bias=None, stride=self.conv.stride,
                                     padding=self.conv.padding, dilation=self.conv.dilation,
                                     groups=self.conv.groups)


class AutoQuantConvTranspose2d(nn.Module):
    """ConvTranspose2d with NVFP4 fake quant."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.qop = QuantOp()
        self.qparams_x = QParams(
            scaling_type=ScalingType.PER_1D_BLOCK,
            quant_dtype=k_utils.Fp4Formats.E2M1,
            pow_2_scales=True,
            quant_tile_shape=(1, 16),
            tensor_type=TensorType.X,
        )
        self.qparams_w = QParams(
            scaling_type=ScalingType.PER_1D_BLOCK,
            quant_dtype=k_utils.Fp4Formats.E2M1,
            pow_2_scales=True,
            quant_tile_shape=(1, 16),
            tensor_type=TensorType.W,
        )

    def forward(self, x):
        qx = self.qop.quantize(x, self.qparams_x, return_identity=True).data
        qw = self.qop.quantize(self.tconv.weight, self.qparams_w, return_identity=True).data
        return nn.functional.conv_transpose2d(qx, qw, bias=None, stride=2)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            AutoQuantConv2d(in_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(groups, out_ch),  # not quantised
            nn.ReLU(inplace=False),
            AutoQuantConv2d(out_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.block(x)


class UNetNVFP4(nn.Module):
    """UNet with NVFP4 fake-quant (all conv layers except GroupNorm and out)."""

    def __init__(self, task: str, in_channels=3, out_channels=80, num_instances=10):
        super().__init__()
        self.scale = 0.25
        ch = lambda v: int(self.scale * v)

        # Encoder
        self.enc1 = ConvBlock(in_channels, ch(64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(ch(64), ch(128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(ch(128), ch(256))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(ch(256), ch(512))
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(ch(512), ch(1024))

        # Decoder
        self.upconv4 = AutoQuantConvTranspose2d(ch(1024), ch(512))
        self.dec4 = ConvBlock(ch(1024), ch(512))

        self.upconv3 = AutoQuantConvTranspose2d(ch(512), ch(256))
        self.dec3 = ConvBlock(ch(512), ch(256))

        self.upconv2 = AutoQuantConvTranspose2d(ch(256), ch(128))
        self.dec2 = ConvBlock(ch(256), ch(128))

        self.upconv1 = AutoQuantConvTranspose2d(ch(128), ch(64))
        self.dec1 = ConvBlock(ch(128), ch(64))

        # Output conv (full precision)
        if task == "semantic":
            self.out_conv = nn.Conv2d(ch(64), out_channels, kernel_size=1)
        elif task == "instance":
            self.out_conv = nn.Conv2d(ch(64), num_instances, kernel_size=1)
        else:
            raise ValueError("Unsupported task")

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        return self.out_conv(dec1)
